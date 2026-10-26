"""Synthetic verification of independently decoded plane records."""
import base64
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from vdata_standalone.planes import (decode_plane_equations,
                                     plane_center_field,
                                     plane_halfspace_fraction)


def block(data):
    compressed = zlib.compress(data)
    return b"!zlib!" + struct.pack("!II", len(data), len(compressed)) + compressed


def document(raw, planes, *, payload=None):
    root = ET.Element("VoxelXML")
    props = ET.SubElement(root, "Properties")
    ET.SubElement(props, "VoxelSize", Units="nm", Value="20")
    ET.SubElement(props, "NumVoxels", **dict(zip(("NX", "NY", "NZ"), map(str, raw.shape[:3]))))
    ET.SubElement(props, "ModelOrigin", Units="um", X="16", Y="-14", Z="-.5")
    materials = ET.SubElement(root, "Materials")
    ET.SubElement(materials, "material", MaterialID="2", name="Si")
    ET.SubElement(root, "VoxelData").text = base64.b64encode(block(raw.tobytes())).decode()
    if planes is not None or payload is not None:
        if payload is None:
            data = planes.tobytes()
            payload = block(data[:19]) + block(data[19:])
        ET.SubElement(root, "PlaneEquations").text = base64.b64encode(payload).decode()
    return ET.tostring(root, encoding="utf-8")


class PlaneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory / "sample.vdata"
        self.raw = np.empty((2, 3, 4, 3), dtype=np.uint8)
        self.raw[...] = [32, 2, 255]
        self.planes = np.zeros((2, 3, 4, 4), dtype="<f4")
        self.planes[...] = [1, 0, 0, .25]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, planes=None, **kwargs):
        self.path.write_bytes(document(self.raw, self.planes if planes is None else planes, **kwargs))
        return self.path

    def test_decode_exact_float_bits_axes_source_unchanged(self):
        self.planes[1, 2, 3] = [0, -1, 0, -.25]
        path = self.write()
        before = path.read_bytes()
        result, meta = decode_plane_equations(path)
        self.assertEqual(result.dtype, np.dtype("<f4"))
        self.assertEqual(result.tobytes(), self.planes.tobytes())
        self.assertEqual(meta["compressed_blocks"], 2)
        self.assertEqual(meta["decoded_bytes"], self.planes.nbytes)
        self.assertEqual(meta["nonzero_normal_count"], 24)
        np.testing.assert_allclose(meta["origin_m"], [16e-6, -14e-6, -.5e-6])
        self.assertAlmostEqual(meta["voxel_size_m"], 20e-9)
        self.assertEqual(path.read_bytes(), before)

    def test_missing_planes_rejected(self):
        self.path.write_bytes(document(self.raw, None))
        with self.assertRaisesRegex(ValueError, "no PlaneEquations"):
            decode_plane_equations(self.path)

    def test_nonfinite_wrong_endian_and_nonunit_rejected(self):
        for bad in (self.planes.astype(">f4"), self.planes * 2, self.planes * float("nan")):
            with self.subTest(dtype=bad.dtype):
                with self.assertRaises(ValueError):
                    decode_plane_equations(self.write(bad))

    def test_zero_plane_records_preserved(self):
        self.planes[0, 0, 0] = [0, 0, 0, 2]
        result, meta = decode_plane_equations(self.write())
        self.assertEqual(result.tobytes(), self.planes.tobytes())
        self.assertEqual(meta["nonzero_normal_count"], 23)

    def test_byte_cap_and_corrupt_lengths(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            decode_plane_equations(self.write(), max_raw_bytes=20)
        for data in (self.planes.tobytes()[:-1], self.planes.tobytes() + b"extra"):
            with self.assertRaises(ValueError):
                decode_plane_equations(self.write(payload=block(data)))
        with self.assertRaises(ValueError):
            decode_plane_equations(self.write(payload=block(self.planes.tobytes())[:-3]))

    def test_mmap_no_overwrite_cleanup(self):
        path = self.write()
        target = self.directory / "planes.npy"
        planes, _ = decode_plane_equations(path, target)
        self.assertIsInstance(planes, np.memmap)
        np.testing.assert_array_equal(planes, self.planes)
        planes._mmap.close()
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            decode_plane_equations(path, target)
        self.assertEqual(target.read_bytes(), before)
        broken_target = self.directory / "bad.npy"
        self.write(payload=block(self.planes.tobytes()[:-1]))
        with self.assertRaises(ValueError):
            decode_plane_equations(path, broken_target)
        self.assertFalse(broken_target.exists())

    def test_unverified_encoding_rejected(self):
        content = document(self.raw, self.planes).replace(b"<PlaneEquations>", b'<PlaneEquations encoding="binary">')
        self.path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "encoding"):
            decode_plane_equations(self.path)

    def test_halfspace_exact_axis_diagonal_and_complement(self):
        n = np.array([[1, 0, 0], [-1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], float)
        d = np.array([.25, -.75, .5, .5, 1.5])
        expected = [.25, .25, .125, .5 ** 3 / 6, .5]
        result = plane_halfspace_fraction(n, d)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result + plane_halfspace_fraction(-n, -d), 1, rtol=0, atol=1e-14)
        np.testing.assert_allclose(plane_halfspace_fraction(n * 100, d * 100), result, rtol=0, atol=1e-14)

    def test_halfspace_near_axis_and_full_empty(self):
        n = np.array([[1, 1e-12, -1e-12], [1, 0, 0], [1, 1, 1]])
        values = plane_halfspace_fraction(n, np.array([.3, -1, 4]))
        np.testing.assert_allclose(values, [.3, 0, 1], atol=1e-10)
        with self.assertRaises(ValueError):
            plane_halfspace_fraction(np.zeros((1, 3)), np.zeros(1))

    def test_center_field_keeps_original_offset_and_distance_units(self):
        field, report = plane_center_field(self.raw, self.planes, 2, 20e-9)
        np.testing.assert_allclose(field, 5e-9, rtol=0, atol=1e-15)
        self.assertFalse(report["plane_offsets_modified"])
        self.assertEqual(report["mixed_fraction_error"]["maximum"], 0)
        self.planes[..., 3] = .75
        field, report = plane_center_field(self.raw, self.planes, 2, 20e-9)
        np.testing.assert_allclose(field, -5e-9, rtol=0, atol=1e-15)
        self.assertEqual(report["mixed_fraction_error"]["maximum"], .5)
        self.assertEqual(report["mixed_fraction_error_best_of_either_orientation"]["maximum"], 0)

    def test_material_side_reversal_and_zero_normal_fallback(self):
        self.raw[...] = [32, 255, 2]
        field, _ = plane_center_field(self.raw, self.planes, 2, 20e-9)
        np.testing.assert_allclose(field, -5e-9, atol=1e-15)
        self.planes[...] = [0, 0, 0, 2]
        field, report = plane_center_field(self.raw, self.planes, 2, 20e-9)
        np.testing.assert_allclose(field, -5e-9, atol=1e-15)
        self.assertEqual(report["mixed_voxels_without_planes"], 24)

    def test_pure_cell_signs_and_multimaterial_rejection(self):
        self.raw[0] = [129, 2, 2]
        self.raw[1] = [129, 255, 255]
        field, report = plane_center_field(self.raw, self.planes, 2, 20e-9)
        self.assertTrue(np.all(field[0] < 0))
        self.assertTrue(np.all(field[1] > 0))
        self.raw[1, 1, 1] = [32, 2, 3]
        with self.assertRaisesRegex(ValueError, "multi-material"):
            plane_center_field(self.raw, self.planes, 2, 20e-9)

    def test_mesh_integration_planar_interface_is_closed_and_keeps_coordinate(self):
        from vdata_standalone.meshing import mesh_material
        raw = np.empty((4, 5, 6, 3), dtype=np.uint8)
        raw[...] = [129, 255, 255]
        raw[0] = [129, 2, 2]
        raw[1] = [32, 2, 255]
        planes = np.zeros(raw.shape[:-1] + (4,), dtype=np.float32)
        planes[..., 0] = 1
        planes[..., 3] = 1.25 - np.arange(4)[:, None, None]
        meta = {"dimensions_xyz": [4, 5, 6], "origin_m": [10e-6, 20e-6, -1e-6],
                "voxel_size_m": 20e-9, "has_plane_equations": True}
        vertices, faces, report = mesh_material(raw, meta, 2, surface="plane", planes=planes)
        self.assertTrue(report["watertight_edges"])
        self.assertTrue(report["consistent_winding"])
        self.assertEqual(report["degenerate_faces"], 0)
        self.assertTrue(report["plane_equations_used"])
        self.assertAlmostEqual(float(vertices[:, 0].max()), 1.25 * 20e-9, delta=1e-15)
        np.testing.assert_allclose(report["translation_m"], meta["origin_m"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
