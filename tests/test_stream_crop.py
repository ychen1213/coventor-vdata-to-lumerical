"""Streaming ROI decoding preserves records and validates the entire source."""
import base64
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from vdata_standalone.reader import decode_vdata
from vdata_standalone.planes import decode_plane_equations


def _block(data):
    packed = zlib.compress(data)
    return b"!zlib!" + struct.pack("!II", len(data), len(packed)) + packed


def _payloads(data):
    # Record boundaries intentionally do not coincide with zlib or XML nodes.
    cuts = (0, 1, 2, 17, 31, len(data))
    return [_block(data[a:b]) for a, b in zip(cuts, cuts[1:]) if b > a]


def _document(raw, planes):
    root = ET.Element("VoxelXML")
    props = ET.SubElement(root, "Properties")
    ET.SubElement(props, "VoxelSize", Units="nm", Value="20")
    ET.SubElement(props, "NumVoxels", **dict(zip(("NX", "NY", "NZ"), map(str, raw.shape[:3]))))
    ET.SubElement(props, "ModelOrigin", Units="um", X="16", Y="-14", Z="-.5")
    palette = ET.SubElement(root, "Materials")
    ET.SubElement(palette, "material", MaterialID="2", name="Si")
    ET.SubElement(palette, "material", MaterialID="3", name="Oxide")
    for tag, data in (("VoxelData", raw.tobytes()), ("PlaneEquations", planes.tobytes())):
        parts = _payloads(data)
        for combined in (b"".join(parts[:3]), b"".join(parts[3:])):
            ET.SubElement(root, tag).text = base64.b64encode(combined).decode("ascii")
    return ET.tostring(root, encoding="utf-8")


class StreamCropTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.source = self.directory / "synthetic.vdata"
        rng = np.random.default_rng(53)
        self.raw = np.empty((3, 4, 5, 3), dtype=np.uint8)
        self.raw[..., 0] = rng.integers(0, 129, size=self.raw.shape[:3], dtype=np.uint8)
        self.raw[..., 1] = 2
        self.raw[..., 2] = 255
        self.raw[-1, -1, -1] = [129, 3, 3]
        self.planes = np.zeros(self.raw.shape[:3] + (4,), dtype="<f4")
        self.planes[..., 0] = 1
        self.planes[..., 3] = np.arange(60).reshape(3, 4, 5) / 64
        self.planes[-1, -1, -1] = [0, 0, 0, 2]
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.source.write_bytes(_document(self.raw, self.planes))
        return self.source

    def test_varied_xyz_crops_exact_in_memory_and_mmap(self):
        crops = [((0, 3), (0, 4), (0, 5)), ((1, 3), (1, 3), (0, 5)),
                 ((0, 3), (0, 4), (1, 4)), ((1, 2), (2, 4), (2, 4)),
                 ((0, 1), (0, 1), (0, 1)), ((2, 3), (3, 4), (4, 5))]
        full_raw, _ = decode_vdata(self.source)
        full_planes, _ = decode_plane_equations(self.source)
        before = self.source.read_bytes()
        for number, crop in enumerate(crops):
            selected = tuple(slice(lo, hi) for lo, hi in crop)
            for decoder, reference in ((decode_vdata, full_raw),
                                       (decode_plane_equations, full_planes)):
                for mapped in (False, True):
                    with self.subTest(crop=crop, decoder=decoder.__name__, mapped=mapped):
                        path = self.directory / f"{decoder.__name__}_{number}_{mapped}.npy" if mapped else None
                        result, _ = decoder(self.source, path, crop_indices=crop)
                        self.assertEqual(result.tobytes(), reference[selected].tobytes())
                        self.assertTrue(result.flags.c_contiguous)
                        if mapped:
                            np.testing.assert_array_equal(np.load(path), reference[selected])
                            result._mmap.close()
        self.assertEqual(self.source.read_bytes(), before)

    def test_source_and_region_metadata_are_unambiguous(self):
        crop = ((1, 3), (1, 4), (2, 4))
        raw, meta = decode_vdata(self.source, crop_indices=crop)
        planes, plane_meta = decode_plane_equations(self.source, crop_indices=crop)
        for info in (meta, plane_meta):
            self.assertEqual(info["dimensions_xyz"], [3, 4, 5])
            np.testing.assert_allclose(info["origin_m"], [16e-6, -14e-6, -.5e-6])
            self.assertEqual(info["crop_dimensions_xyz"], [2, 3, 2])
            self.assertEqual(info["crop_indices_xyz"], [[1, 3], [1, 4], [2, 4]])
            self.assertTrue(info["crop_applied"])
            region = info["region_metadata"]
            self.assertEqual(region["dimensions_xyz"], [2, 3, 2])
            np.testing.assert_allclose(region["origin_m"], [16.02e-6, -13.98e-6, -.46e-6])
            np.testing.assert_allclose(region["bounds_m"][1], [16.06e-6, -13.92e-6, -.42e-6])
            self.assertEqual(region["voxel_count"], 12)
        self.assertEqual(meta["expected_raw_bytes"], self.raw.nbytes)
        self.assertEqual(meta["decoded_raw_bytes"], self.raw.nbytes)
        self.assertEqual(meta["stored_raw_bytes"], raw.nbytes)
        self.assertEqual(meta["raw_shape"], [2, 3, 2, 3])
        self.assertEqual(meta["material_ids_in_records"], [2, 3, 255])
        self.assertEqual(plane_meta["plane_shape"], [2, 3, 2, 4])
        self.assertEqual(plane_meta["decoded_bytes"], self.planes.nbytes)
        self.assertEqual(plane_meta["stored_bytes"], planes.nbytes)
        self.assertEqual(plane_meta["nonzero_normal_count"], 59)
        self.assertEqual(plane_meta["raw_plane_sha256"], hashlib.sha256(self.planes.tobytes()).hexdigest())
        self.assertEqual(plane_meta["region_plane_sha256"], hashlib.sha256(planes.tobytes()).hexdigest())

    def test_chunk_block_and_node_boundaries_do_not_change_bytes(self):
        crop = ((1, 3), (0, 3), (1, 5))
        selected = tuple(slice(lo, hi) for lo, hi in crop)
        # Seven-byte output chunks split both 3-byte and 16-byte records.
        with patch("vdata_standalone.reader._CHUNK", 7):
            raw, _ = decode_vdata(self.source, crop_indices=crop)
            planes, _ = decode_plane_equations(self.source, crop_indices=crop)
        self.assertEqual(raw.tobytes(), self.raw[selected].tobytes())
        self.assertEqual(planes.tobytes(), self.planes[selected].tobytes())

    def test_bad_voxel_outside_roi_is_rejected_and_output_removed(self):
        crop = ((0, 1), (0, 1), (0, 1))
        target = self.directory / "failed.npy"
        for record in ([64, 2, 9], [130, 2, 2], [129, 2, 3]):
            with self.subTest(record=record):
                self.raw[-1, -1, -1] = record
                self.write()
                with self.assertRaises(ValueError):
                    decode_vdata(self.source, target, crop_indices=crop)
                self.assertFalse(target.exists())

    def test_bad_plane_outside_roi_is_rejected_and_output_removed(self):
        crop = ((0, 1), (0, 1), (0, 1))
        target = self.directory / "failed_planes.npy"
        for plane in ([2, 0, 0, .25], [1, 0, 0, float("nan")],
                      [0, 0, 0, float("inf")]):
            with self.subTest(plane=plane):
                self.planes[-1, -1, -1] = plane
                self.write()
                with self.assertRaises(ValueError):
                    decode_plane_equations(self.source, target, crop_indices=crop)
                self.assertFalse(target.exists())

    def test_crop_cannot_bypass_full_source_size_limit(self):
        for decoder, size in ((decode_vdata, self.raw.nbytes),
                              (decode_plane_equations, self.planes.nbytes)):
            with self.subTest(decoder=decoder.__name__):
                target = self.directory / f"cap_{decoder.__name__}.npy"
                with self.assertRaises(ValueError):
                    decoder(self.source, target, crop_indices=((0, 1),) * 3,
                            max_raw_bytes=size - 1)
                self.assertFalse(target.exists())

    def test_existing_outputs_and_source_are_never_overwritten(self):
        for decoder in (decode_vdata, decode_plane_equations):
            for target in (self.source, self.directory / "existing.npy"):
                with self.subTest(decoder=decoder.__name__, target=target.name):
                    if target != self.source:
                        target.write_bytes(b"preserve this file")
                    before = target.read_bytes()
                    with self.assertRaises(FileExistsError):
                        decoder(self.source, target, crop_indices=((0, 1),) * 3)
                    self.assertEqual(target.read_bytes(), before)

    def test_open_memmap_failure_cleans_only_new_reservation(self):
        for decoder in (decode_vdata, decode_plane_equations):
            target = self.directory / f"allocation_{decoder.__name__}.npy"
            with self.subTest(decoder=decoder.__name__):
                with patch("numpy.lib.format.open_memmap", side_effect=OSError("allocation failed")):
                    with self.assertRaisesRegex(OSError, "allocation failed"):
                        decoder(self.source, target, crop_indices=((0, 1),) * 3)
                self.assertFalse(target.exists())
                target.write_bytes(b"original")
                with patch("numpy.lib.format.open_memmap", side_effect=AssertionError("must not run")):
                    with self.assertRaises(FileExistsError):
                        decoder(self.source, target, crop_indices=((0, 1),) * 3)
                self.assertEqual(target.read_bytes(), b"original")

    def test_invalid_crop_is_rejected_before_output_creation(self):
        invalid = [(), ((0, 1),), ((0, 1),) * 4, ((0, 1, 2), (0, 1), (0, 1)),
                   ((0, 0), (0, 1), (0, 1)), ((-1, 1), (0, 1), (0, 1)),
                   ((0, 4), (0, 1), (0, 1)), ((1.0, 2), (0, 1), (0, 1)),
                   ((False, 1), (0, 1), (0, 1)), ((0, 1), None, (0, 1)),
                   ((np.bool_(False), 1), (0, 1), (0, 1)),
                   (slice(0, 1), slice(0, 1), slice(0, 1))]
        target = self.directory / "invalid.npy"
        for decoder in (decode_vdata, decode_plane_equations):
            for crop in invalid:
                with self.subTest(decoder=decoder.__name__, crop=crop):
                    with self.assertRaisesRegex(ValueError, "crop_indices"):
                        decoder(self.source, target, crop_indices=crop)
                    self.assertFalse(target.exists())

    def test_none_allocates_only_the_roi_shape_and_numpy_indices_work(self):
        crop = tuple((np.int64(0), np.int64(1)) for _ in range(3))
        original_empty = np.empty
        for decoder, channels in ((decode_vdata, 3), (decode_plane_equations, 4)):
            with self.subTest(decoder=decoder.__name__):
                with patch("numpy.empty", wraps=original_empty) as allocate:
                    result, _ = decoder(self.source, crop_indices=crop)
                self.assertEqual(result.shape, (1, 1, 1, channels))
                allocate.assert_called_once()
                self.assertEqual(allocate.call_args.args[0], (1, 1, 1, channels))

    def test_default_region_matches_entire_source(self):
        for decoder in (decode_vdata, decode_plane_equations):
            with self.subTest(decoder=decoder.__name__):
                _, meta = decoder(self.source)
                self.assertFalse(meta["crop_applied"])
                self.assertEqual(meta["region_metadata"]["dimensions_xyz"], meta["dimensions_xyz"])
                self.assertEqual(meta["region_metadata"]["origin_m"], meta["origin_m"])


if __name__ == "__main__":
    unittest.main()
