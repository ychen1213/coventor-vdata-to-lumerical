"""End-to-end checks using generated, non-vendor data."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from vdata_standalone.cli import main


def write_fixture(path, with_second_material=False):
    raw = np.empty((8, 9, 7, 3), dtype=np.uint8)
    raw[..., 0] = 129
    raw[..., 1:] = 255
    raw[2:6, 2:7, 1:6, 1:] = 2
    if with_second_material:
        raw[2:6, 2:7, 1:3, 1:] = 3
    root = ET.Element("VoxelXML")
    p = ET.SubElement(root, "Properties")
    ET.SubElement(p, "VoxelSize", Units="um", Value="0.02")
    ET.SubElement(p, "NumVoxels", NX="8", NY="9", NZ="7")
    ET.SubElement(p, "ModelOrigin", Units="um", X="16", Y="-14", Z="-0.5")
    ET.SubElement(p, "ModelFile", name="missing/source.zam")
    mats = ET.SubElement(root, "Materials")
    ET.SubElement(mats, "Material", MaterialID="2", name="/Silicon/Si_Xtal")
    ET.SubElement(mats, "Material", MaterialID="3", name="UnknownDielectric")
    packed = zlib.compress(raw.tobytes())
    b = b"!zlib!" + struct.pack("!II", raw.nbytes, len(packed)) + packed
    ET.SubElement(root, "VoxelData").text = base64.b64encode(b).decode("ascii")
    path.write_bytes(ET.tostring(root))
    return raw


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic.vdata"
        self.raw = write_fixture(self.source)
        self.original = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)

    def test_decode_roundtrip_in_unicode_output(self):
        out = self.root / "独立 转换"
        self.assertEqual(self.run_cli(["decode", str(self.source), "--output", str(out)]), 0)
        with np.load(out / "voxels.npz") as archive:
            np.testing.assert_array_equal(archive["raw"], self.raw)
        meta = json.loads((out / "voxel_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["dimensions_xyz"], [8, 9, 7])
        np.testing.assert_allclose(meta["origin_m"], [16e-6, -14e-6, -.5e-6])
        m = json.loads((out / "conversion_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(m["source_unchanged"])
        self.assertEqual(m["vendor_api_calls"], [])
        self.assertFalse((out / ".decode-cache.npy").exists())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original)

    def test_crop_is_exact_records_and_physical_origin(self):
        out = self.root / "crop"
        b = [16.02, 16.14, -13.98, -13.84, -.48, -.38]
        self.assertEqual(self.run_cli(["decode", str(self.source), "--output", str(out), "--crop-um", *map(str,b)]), 0)
        with np.load(out / "voxels.npz") as archive:
            np.testing.assert_array_equal(archive["raw"], self.raw[1:7,1:8,1:6])
        meta = json.loads((out / "voxel_metadata.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(meta["origin_m"], [16.02e-6,-13.98e-6,-.48e-6])

    def test_stl_script_and_coordinate_manifest(self):
        out = self.root / "surface"
        self.assertEqual(self.run_cli(["convert", str(self.source), "--output", str(out)]), 0)
        m = json.loads((out / "conversion_manifest.json").read_text(encoding="utf-8"))
        obj = m["objects"][0]
        np.testing.assert_allclose(obj["translation_m"], [16e-6,-14e-6,-.5e-6])
        stl = out / obj["stl"]
        b = stl.read_bytes()
        face_count = struct.unpack("<I", b[80:84])[0]
        self.assertGreater(face_count, 0)
        self.assertEqual(len(b), 84 + 50 * face_count)
        script = (out / "import_lumerical.lsf").read_text(encoding="utf-8")
        self.assertIn("stlimport(", script)
        self.assertIn("Si (Silicon) - Palik", script)
        self.assertFalse(list(out.glob("*.fsp")))
        self.assertEqual(len(m["objects"]),1)

    def test_nonempty_output_is_preserved(self):
        out = self.root / "existing"
        out.mkdir()
        sentinel = out / "sentinel.txt"
        sentinel.write_text("preserve",encoding="utf-8")
        self.assertEqual(self.run_cli(["convert",str(self.source),"--output",str(out)]),2)
        self.assertEqual(sentinel.read_text(),"preserve")

    def test_fractional_crop_refused_before_output(self):
        out = self.root / "badcrop"
        b=[16.01,16.14,-14,-13.82,-.5,-.36]
        self.assertEqual(self.run_cli(["decode",str(self.source),"--output",str(out),"--crop-um",*map(str,b)]),2)
        self.assertFalse(out.exists())

    def test_unknown_optical_material_blocks_script(self):
        write_fixture(self.source,with_second_material=True)
        out=self.root/"multi"
        self.assertEqual(self.run_cli(["convert",str(self.source),"--output",str(out)]),0)
        m=json.loads((out/"conversion_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(m["objects"]),2)
        script=(out/"import_lumerical.lsf").read_text(encoding="utf-8")
        self.assertIn("Import blocked",script)
        self.assertNotIn("stlimport(",script)

    def test_multi_material_qem_rejected(self):
        write_fixture(self.source,with_second_material=True)
        out=self.root/"rejected"
        self.assertEqual(self.run_cli(["convert",str(self.source),"--output",str(out),"--reduction","0.45"]),2)
        m=json.loads((out/"conversion_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["status"],"failed")
        self.assertFalse((out/".decode-cache.npy").exists())

    def test_material_mapping_regeneration_without_original_input(self):
        write_fixture(self.source,with_second_material=True)
        out=self.root/"mapped"
        self.assertEqual(self.run_cli(["convert",str(self.source),"--output",str(out)]),0)
        mapping=out/"my_map.json"
        mapping.write_text(json.dumps({"materials":{"id:3":"My verified dielectric"}}),encoding="utf-8")
        self.source.unlink()
        self.assertEqual(self.run_cli(["lsf",str(out),"--material-map",str(mapping)]),0)
        script=(out/"import_lumerical.lsf").read_text(encoding="utf-8")
        self.assertIn("stlimport(",script)
        self.assertIn("My verified dielectric",script)
        self.assertNotIn("Import blocked",script)

    def test_compact_restore_stl_matches_standard_without_source(self):
        standard, compact = self.root / "standard", self.root / "compact"
        self.assertEqual(self.run_cli(["convert", str(self.source), "--output", str(standard)]), 0)
        self.assertEqual(self.run_cli(["convert", str(self.source), "--output", str(compact), "--storage", "compact"]), 0)
        self.assertEqual(list(compact.rglob("*.stl")), [])
        self.assertEqual(len(list(compact.rglob("*.vmesh"))), 1)
        self.assertIn("Import blocked", (compact / "import_lumerical.lsf").read_text())
        self.source.unlink()
        self.assertEqual(self.run_cli(["export-stl", str(compact)]), 0)
        self.assertEqual((standard / "meshes/material_0002.stl").read_bytes(),
                         (compact / "meshes/material_0002.stl").read_bytes())
        self.assertNotIn("Import blocked", (compact / "import_lumerical.lsf").read_text())
        self.assertEqual(self.run_cli(["export-stl", str(compact)]), 0)

    def test_compact_planar_mat_preserves_vertices_faces_and_translation(self):
        from scipy.io import loadmat
        from vdata_standalone.storage import read_mesh_archive
        out = self.root / "planar"
        self.assertEqual(self.run_cli(["convert", str(self.source), "--output", str(out), "--storage", "compact"]), 0)
        self.source.unlink()
        self.assertEqual(self.run_cli(["export-planar", str(out)]), 0)
        vertices, faces, translation, _ = read_mesh_archive(out / "meshes/material_0002.vmesh")
        mat = loadmat(out / "meshes/material_0002_planar.mat")
        self.assertEqual(mat["vdata_vertices"].tobytes(), vertices.tobytes())
        np.testing.assert_array_equal(mat["vdata_facets"][:, 0, :].T - 1, faces)
        np.testing.assert_array_equal(mat["vdata_translation_m"].ravel(), translation)
        script = (out / "import_lumerical_planar.lsf").read_text()
        self.assertIn("addplanarsolid;", script)
        self.assertLess(script.index('set("vertices"'), script.index('set("facets"'))
        self.assertNotIn("stlimport", script)
        self.assertEqual(list(out.rglob("*.stl")), [])
        self.assertEqual(self.run_cli(["export-planar", str(out)]), 0)

    def test_pack_existing_npz_and_refuse_replacement(self):
        from vdata_standalone.storage import read_mesh_archive
        out = self.root / "standard"
        self.assertEqual(self.run_cli(["convert", str(self.source), "--output", str(out)]), 0)
        archive = self.root / "new.vmesh"
        src = out / "meshes/material_0002.npz"
        self.assertEqual(self.run_cli(["pack-mesh", str(src), "--output", str(archive)]), 0)
        vertices, faces, translation, _ = read_mesh_archive(archive)
        with np.load(src) as old:
            self.assertEqual(vertices.tobytes(), old["vertices_local_m"].tobytes())
            self.assertEqual(faces.tobytes(), old["faces"].tobytes())
            self.assertEqual(translation.tobytes(), old["translation_m"].tobytes())
        before = archive.read_bytes()
        self.assertEqual(self.run_cli(["pack-mesh", str(src), "--output", str(archive)]), 2)
        self.assertEqual(before, archive.read_bytes())

    def test_missing_plane_mode_refused_without_creating_output(self):
        out = self.root / "missingplanes"
        self.assertEqual(self.run_cli(["convert", str(self.source), "--output", str(out), "--surface", "plane"]), 2)
        self.assertFalse(out.exists())

    def test_modified_archive_refused_before_stl_expansion(self):
        out = self.root / "changed"
        self.assertEqual(self.run_cli(["convert", str(self.source), "--output", str(out), "--storage", "compact"]), 0)
        archive = out / "meshes/material_0002.vmesh"
        with archive.open("ab") as f:
            f.write(b"modified")
        self.assertEqual(self.run_cli(["export-stl", str(out)]), 2)
        self.assertEqual(list(out.rglob("*.stl")), [])


if __name__=="__main__":
    unittest.main()
