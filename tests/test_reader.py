"""Synthetic, vendor-independent tests for the verified VoxelXML reader."""
import base64
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from vdata_standalone.reader import decode_vdata, fraction_for_material, inspect_vdata


def block(raw, *, raw_size=None, packed=None):
    packed = zlib.compress(raw) if packed is None else packed
    return b"!zlib!" + struct.pack("!II", len(raw) if raw_size is None else raw_size,
                                  len(packed)) + packed


def document(raw, shape=(2, 3, 4), *, chunks=None, unit="um", step="0.02",
             origin_unit="um", extra=""):
    root = ET.Element("VoxelXML")
    props = ET.SubElement(root, "Properties")
    ET.SubElement(props, "VoxelSize", Units=unit, Value=step)
    ET.SubElement(props, "NumVoxels", **dict(zip(("NX", "NY", "NZ"), map(str, shape))))
    ET.SubElement(props, "ModelOrigin", Units=origin_unit, X="16", Y="-14", Z="-.5")
    ET.SubElement(props, "ModelFile", name="Z:/does-not-exist/wafer.zam")
    palette = ET.SubElement(root, "Materials")
    ET.SubElement(palette, "material", name="default")
    ET.SubElement(palette, "material", MaterialID="2", name="Si", red="108", green="153", blue="190", type="Dielectric")
    ET.SubElement(palette, "material", MaterialID="3", name="Oxide")
    if chunks is None:
        chunks = [block(raw)]
    for payload in chunks:
        ET.SubElement(root, "VoxelData").text = base64.b64encode(payload).decode("ascii")
    if extra:
        root.append(ET.fromstring(extra))
    return ET.tostring(root, encoding="utf-8")


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.source = self.directory / "synthetic.vdata"
        self.raw = np.empty((2, 3, 4, 3), dtype=np.uint8)
        self.raw[..., 0] = 129
        self.raw[..., 1:] = 2
        self.raw[0, 0, 0] = [0, 2, 255]
        self.raw[0, 1, 2] = [32, 2, 3]
        self.raw[1, 0, 1] = [64, 3, 2]
        self.raw[1, 2, 3] = [128, 3, 255]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, content=None, **kwargs):
        self.source.write_bytes(document(self.raw.tobytes(), **kwargs) if content is None else content)
        return self.source

    def test_axis_order_metadata_and_read_only_input(self):
        source = self.write(extra="<DopantData>not-decoded</DopantData>")
        before = source.read_bytes()
        inspected = inspect_vdata(source)
        self.assertFalse(inspected["record_validation_performed"])
        raw, meta = decode_vdata(source)
        np.testing.assert_array_equal(raw, self.raw)
        self.assertEqual(raw.shape, (2, 3, 4, 3))
        self.assertEqual(meta["dimensions_xyz"], [2, 3, 4])
        self.assertAlmostEqual(meta["voxel_size_m"], 20e-9)
        np.testing.assert_allclose(meta["origin_m"], [16e-6, -14e-6, -.5e-6])
        self.assertEqual(meta["origin_kind"], "lower_corner")
        self.assertEqual(meta["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(meta["expected_raw_bytes"], 72)
        self.assertFalse(meta["external_references_followed"])
        self.assertIn("DopantData", meta["optional_fields"])
        self.assertEqual(meta["optional_fields_decoded"], [])
        self.assertTrue(meta["record_validation_performed"])
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(next(m for m in meta["materials"] if m["id"] == 2)["color_rgb"], [108, 153, 190])
        self.assertTrue(next(m for m in meta["materials"] if m["id"] == 255)["implicit"])

    def test_multiple_blocks_and_multiple_nodes_unaligned_records(self):
        data = self.raw.tobytes()
        payloads = [block(data[:1]) + block(data[1:17]), block(data[17:])]
        raw, meta = decode_vdata(self.write(chunks=payloads))
        np.testing.assert_array_equal(raw, self.raw)
        self.assertEqual(meta["compressed_blocks"], 3)

    def test_fraction_codes_including_128_and_129(self):
        raw, _ = decode_vdata(self.write())
        silicon = fraction_for_material(raw, 2)
        oxide = fraction_for_material(raw, 3)
        air = fraction_for_material(raw, 255)
        self.assertEqual(silicon.dtype, np.float32)
        self.assertEqual(silicon[0, 0, 0], 0)
        self.assertEqual(silicon[0, 1, 2], .25)
        self.assertEqual(oxide[0, 1, 2], .75)
        self.assertEqual(silicon[1, 0, 1], .5)
        self.assertEqual(oxide[1, 2, 3], 1)
        np.testing.assert_array_equal(silicon + oxide + air, np.ones((2, 3, 4), dtype=np.float32))

    def test_length_units_and_planes_flag(self):
        for unit, size in [("um", ".02"), ("µm", ".02"), ("μm", ".02"),
                           ("nm", "20"), ("Å", "200"), ("m", "2e-8")]:
            with self.subTest(unit=unit):
                _, meta = decode_vdata(self.write(unit=unit, step=size,
                                                origin_unit="nm", extra="<PlaneEquations />"))
                self.assertAlmostEqual(meta["voxel_size_m"], 20e-9)
                self.assertAlmostEqual(meta["origin_m"][0], 16e-9)
                self.assertTrue(meta["has_plane_equations"])

    def test_memmap_is_exact_and_refuses_overwrite(self):
        target = self.directory / "records.npy"
        source = self.write()
        mapped, meta = decode_vdata(source, target)
        self.assertIsInstance(mapped, np.memmap)
        np.testing.assert_array_equal(np.load(target), self.raw)
        mapped._mmap.close()
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            decode_vdata(source, target)
        self.assertEqual(target.read_bytes(), before)
        source_before = source.read_bytes()
        with self.assertRaises(FileExistsError):
            decode_vdata(source, source)
        self.assertEqual(source.read_bytes(), source_before)

    def test_limits_before_output_allocation(self):
        target = self.directory / "records.npy"
        with self.assertRaisesRegex(ValueError, "max_raw_bytes"):
            decode_vdata(self.write(), target, max_raw_bytes=71)
        self.assertFalse(target.exists())
        for cap in (0, -1, .5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                decode_vdata(self.source, max_raw_bytes=cap)

    def test_invalid_grid_and_palette(self):
        valid = document(self.raw.tobytes())
        mutations = [
            (b'<VoxelXML>', b'<Other>'),
            (b'Value="0.02"', b'Value="nan"'),
            (b'Value="0.02"', b'Value="-1"'),
            (b'Units="um"', b'Units="furlong"'),
            (b'NX="2"', b'NX="0"'),
            (b'X="16"', b'X="inf"'),
            (b'MaterialID="3"', b'MaterialID="2"'),
            (b'red="108"', b'red="256"'),
            (b'<VoxelData>', b'<VoxelData encoding="unknown">'),
            (b'<VoxelXML>', b'<VoxelXML format="other">'),
        ]
        for old, new in mutations:
            with self.subTest(new=new), self.assertRaises(ValueError):
                decode_vdata(self.write(valid.replace(old, new)))

    def test_xml_doctype_and_entity_are_refused(self):
        content = b'<!DOCTYPE VoxelXML [<!ENTITY injected "unsafe">]>' + document(self.raw.tobytes())
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            inspect_vdata(self.write(content))

    def test_unknown_material_and_bad_fraction(self):
        for record in ([130, 2, 2], [255, 2, 2], [129, 2, 3], [64, 2, 9]):
            mutated = self.raw.copy()
            mutated[0, 0, 0] = record
            with self.subTest(record=record), self.assertRaises(ValueError):
                decode_vdata(self.write(document(mutated.tobytes())))

    def test_corrupt_compressed_formats_and_cleanup(self):
        data = self.raw.tobytes()
        invalid_payloads = [
            b"!other" + block(data)[6:],
            block(data)[:-1],
            block(data) + b"x",
            block(data[:-3]),
            block(data, raw_size=len(data) - 1),
            block(data, raw_size=len(data) + 1),
            block(data, packed=zlib.compress(data) + b"x"),
            block(data, packed=zlib.compress(data) + zlib.compress(b"x")),
            block(data, packed=zlib.compress(data)[:-1]),
            block(data, packed=b"not zlib"),
        ]
        target = self.directory / "incomplete.npy"
        for number, payload in enumerate(invalid_payloads):
            with self.subTest(number=number), self.assertRaises(ValueError):
                decode_vdata(self.write(chunks=[payload]), target)
            self.assertFalse(target.exists())

    def test_decompression_bomb_is_bounded_by_header(self):
        payload = block(b"\0" * (8 * 1024 * 1024), raw_size=72)
        with self.assertRaisesRegex(ValueError, "declared raw size"):
            decode_vdata(self.write(chunks=[payload]))

    def test_base64_invalid_and_padding_followed_by_data(self):
        valid = document(self.raw.tobytes())
        root = ET.fromstring(valid)
        node = root.find("VoxelData")
        for text in ("!!!", "AAAAA", "eA==eA==", "eA==" + " " * 65536 + "eA=="):
            node.text = text
            with self.subTest(text=text[:12]), self.assertRaises(ValueError):
                decode_vdata(self.write(ET.tostring(root, encoding="utf-8")))

    def test_large_compressed_stream_crosses_decoder_chunks(self):
        rng = np.random.default_rng(53)
        raw = np.empty((2, 100, 200, 3), dtype=np.uint8)
        raw[..., 0] = rng.integers(0, 129, size=raw.shape[:-1], dtype=np.uint8)
        raw[..., 1] = 2
        raw[..., 2] = 3
        decoded, _ = decode_vdata(self.write(document(raw.tobytes(), shape=raw.shape[:-1])))
        np.testing.assert_array_equal(decoded, raw)


if __name__ == "__main__":
    unittest.main()
