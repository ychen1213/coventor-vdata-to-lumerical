"""Polygon MAT interoperability and transactional CLI checks, without CAD."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from hdf5storage import loadmat

from test_cli import write_fixture
from vdata_standalone import cli, coplanar_export
from vdata_standalone.storage import read_mesh_archive


def cube():
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float) * 1e-6
    ids = np.array([[0,3,2,1],[0,1,5,4],[1,2,6,5],
                    [2,3,7,6],[3,0,4,7],[4,5,6,7]], np.int32).ravel()
    return v, np.arange(0, 25, 4, dtype=np.int64), ids, np.array([16,-14,-.5])*1e-6


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def converted(tmp_path, multi=False):
    source = tmp_path / "source.vdata"
    write_fixture(source, with_second_material=multi)
    folder = tmp_path / "converted"
    assert cli.main(["convert", str(source), "--output", str(folder), "--storage", "compact"]) == 0
    return folder


def test_numeric_csr_mat_roundtrip_has_no_dense_facet_padding(tmp_path):
    v, offsets, ids, t = cube()
    target = tmp_path / "cube.mat"
    report = coplanar_export.write_coplanar_mat(target, v, offsets, ids, t)
    loaded = loadmat(target)
    assert loaded["vdata_poly_vertices"].tobytes() == v.tobytes()
    np.testing.assert_array_equal(loaded["vdata_poly_flat"].ravel() - 1, ids)
    np.testing.assert_array_equal(loaded["vdata_poly_offsets"].ravel() - 1, offsets)
    np.testing.assert_array_equal(loaded["vdata_poly_translation_m"].ravel(), t)
    assert loaded["vdata_poly_flat"].shape == (24, 1)
    assert report["facets"] == 6 and not report["cad_import_executed"]


@pytest.mark.parametrize("bad", ["float_ids", "negative", "overflow", "short_loop", "unordered", "nan"])
def test_invalid_csr_refused_without_file(tmp_path, bad):
    v, offsets, ids, t = cube()
    if bad == "float_ids": ids = ids.astype(float)
    elif bad == "negative": ids[0] = -1
    elif bad == "overflow": ids[0] = len(v)
    elif bad == "short_loop": offsets[1] = 2
    elif bad == "unordered": offsets[1], offsets[2] = offsets[2], offsets[1]
    else: v[0,0] = np.nan
    with pytest.raises(ValueError):
        coplanar_export.write_coplanar_mat(tmp_path / "bad.mat", v, offsets, ids, t)
    assert not list(tmp_path.iterdir())


def test_writer_preserves_existing_or_concurrently_created_file(tmp_path, monkeypatch):
    v, offsets, ids, t = cube()
    target = tmp_path / "existing.mat"
    target.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        coplanar_export.write_coplanar_mat(target, v, offsets, ids, t)
    assert target.read_bytes() == b"preserve"
    raced = tmp_path / "raced.mat"
    original = coplanar_export.os.link
    def race(src, dst):
        Path(dst).write_bytes(b"other writer")
        original(src, dst)
    monkeypatch.setattr(coplanar_export.os, "link", race)
    with pytest.raises(FileExistsError):
        coplanar_export.write_coplanar_mat(raced, v, offsets, ids, t)
    assert raced.read_bytes() == b"other writer"
    assert {p.name for p in tmp_path.iterdir()} == {"existing.mat", "raced.mat"}


def test_export_without_source_and_after_moving_directory(tmp_path):
    out = converted(tmp_path)
    (tmp_path / "source.vdata").unlink()
    assert cli.main(["export-coplanar", str(out)]) == 0
    manifest = json.loads((out / "conversion_manifest.json").read_text(encoding="utf8"))
    obj = manifest["objects"][0]
    v, faces, t, _ = read_mesh_archive(out / obj["compact_mesh"])
    loaded = loadmat(out / obj["coplanar_mat"])
    assert loaded["vdata_poly_vertices"].tobytes() == v.tobytes()
    np.testing.assert_array_equal(loaded["vdata_poly_translation_m"].ravel(), t)
    assert len(loaded["vdata_poly_offsets"]) - 1 < len(faces)
    assert not list(out.rglob("*.stl"))
    assert not list(out.rglob("*.fsp"))
    before = (out / obj["coplanar_mat"]).read_bytes()
    moved = tmp_path / "移动后的 转换"
    shutil.move(str(out), str(moved))
    assert cli.main(["export-coplanar", str(moved)]) == 0
    assert (moved / obj["coplanar_mat"]).read_bytes() == before
    script = (moved / "import_lumerical_coplanar.lsf").read_text(encoding="utf8")
    assert moved.as_posix() in script and out.as_posix() not in script
    assert "addplanarsolid(vdata_poly_vertices,vdata_poly_facets);" in script
    assert "transpose(vdata_poly_flat" in script
    assert 'set("facets"' not in script and "stlimport(" not in script


def test_one_command_from_vdata_and_invalid_options_preflight(tmp_path):
    source = tmp_path / "synthetic.vdata"
    write_fixture(source)
    out = tmp_path / "one_command"
    assert cli.main(["convert", str(source), "--output", str(out), "--storage", "compact", "--coplanar"]) == 0
    assert (out / "import_lumerical_coplanar.lsf").is_file()
    assert json.loads((out / "coplanar_report.json").read_text())["status"] == "complete"
    invalid = tmp_path / "invalid"
    assert cli.main(["convert", str(source), "--output", str(invalid), "--coplanar-min-faces", "1"]) == 2
    assert not invalid.exists()


@pytest.mark.parametrize("damage", ["source", "unrecorded_target", "outside", "duplicate"])
def test_entire_batch_is_preflighted_before_any_export(tmp_path, damage):
    out = converted(tmp_path, multi=True)
    manifest_path = out / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "source":
        with (out / "meshes/material_0003.vmesh").open("ab") as stream:
            stream.write(b"modified")
    elif damage == "unrecorded_target":
        (out / "meshes/material_0003_coplanar.mat").write_bytes(b"user data")
    elif damage == "outside":
        manifest["objects"][1]["compact_mesh"] = "../../outside.vmesh"
        manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    else:
        manifest["objects"][1]["material_id"] = manifest["objects"][0]["material_id"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    before = snapshot(out)
    assert cli.main(["export-coplanar", str(out)]) == 2
    assert snapshot(out) == before


def test_mid_batch_failure_rolls_back_and_can_retry(tmp_path, monkeypatch):
    out = converted(tmp_path, multi=True)
    before = snapshot(out)
    original = coplanar_export.write_coplanar_mat
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("Simulated second material failure")
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(coplanar_export, "write_coplanar_mat", fail_second)
        assert cli.main(["export-coplanar", str(out)]) == 2
    assert len(calls) == 2 and snapshot(out) == before
    assert cli.main(["export-coplanar", str(out)]) == 0
    assert "Import blocked" in (out / "import_lumerical_coplanar.lsf").read_text()


def test_script_failure_restores_existing_outputs(tmp_path, monkeypatch):
    out = converted(tmp_path)
    before = snapshot(out)
    def fail(objects, folder, **kwargs):
        (folder / "import_lumerical_coplanar.lsf").write_text("partial")
        raise OSError("Simulated script write failure")
    with monkeypatch.context() as patch:
        patch.setattr(coplanar_export, "generate_coplanar_lsf", fail)
        assert cli.main(["export-coplanar", str(out)]) == 2
    assert snapshot(out) == before


@pytest.mark.parametrize("name", ["meshes/material_0002_coplanar.mat", "coplanar_report.json"])
def test_modified_previous_export_cannot_be_silently_adopted(tmp_path, name):
    out = converted(tmp_path)
    assert cli.main(["export-coplanar", str(out)]) == 0
    with (out / name).open("ab") as stream:
        stream.write(b"tamper")
    before = snapshot(out)
    assert cli.main(["export-coplanar", str(out)]) == 2
    assert snapshot(out) == before


def test_compaction_preserves_each_facet_coordinate_sequence(tmp_path):
    v, offsets, ids, t = cube()
    unused = np.array([[.25,.25,.25],[.75,.75,.75]]) * 1e-6
    v = np.vstack((unused[:1], v[:4], unused[1:], v[4:]))
    old_to_new = np.array([1,2,3,4,6,7,8,9])
    ids = old_to_new[ids]
    compacted, remapped, report = coplanar_export.compact_vertex_table(v, ids)
    assert len(compacted) == 8 and report["removed_unreferenced_vertices"] == 2
    assert v[ids].tobytes() == compacted[remapped].tobytes()
    target = tmp_path / "compact.mat"
    coplanar_export.write_coplanar_mat(target, compacted, offsets, remapped, t)
    loaded = loadmat(target)
    new_ids = loaded["vdata_poly_flat"].ravel().astype(np.int64) - 1
    assert loaded["vdata_poly_vertices"][new_ids].tobytes() == v[ids].tobytes()


def test_compaction_keeps_baseline_and_reuses_its_audited_polygons(tmp_path, monkeypatch):
    from vdata_standalone import coplanar
    out = converted(tmp_path)
    assert cli.main(["export-coplanar", str(out)]) == 0
    baseline_mat = (out / "meshes/material_0002_coplanar.mat").read_bytes()
    baseline_script = (out / "import_lumerical_coplanar.lsf").read_bytes()
    def must_not_remerge(*args, **kwargs):
        raise AssertionError("Audited baseline should be reused")
    with monkeypatch.context() as patch:
        patch.setattr(coplanar, "merge_coplanar", must_not_remerge)
        assert cli.main(["export-coplanar", str(out), "--compact-vertices"]) == 0
        assert cli.main(["export-coplanar", str(out), "--compact-vertices"]) == 0
    assert (out / "meshes/material_0002_coplanar.mat").read_bytes() == baseline_mat
    assert (out / "import_lumerical_coplanar.lsf").read_bytes() == baseline_script
    old = loadmat(out / "meshes/material_0002_coplanar.mat")
    new = loadmat(out / "meshes/material_0002_coplanar_compact.mat")
    before_ids = old["vdata_poly_flat"].ravel().astype(np.int64)-1
    after_ids = new["vdata_poly_flat"].ravel().astype(np.int64)-1
    assert old["vdata_poly_vertices"][before_ids].tobytes() == new["vdata_poly_vertices"][after_ids].tobytes()
    np.testing.assert_array_equal(old["vdata_poly_offsets"], new["vdata_poly_offsets"])
    report = json.loads((out / "coplanar_compact_report.json").read_text())
    assert report["objects"][0]["reused_audited_baseline_polygons"]


def test_cli_entry_and_compact_only_one_command(tmp_path):
    entry = Path(cli.__file__).resolve().parents[1] / "run.py"
    source = tmp_path / "source.vdata"
    write_fixture(source)
    out = tmp_path / "compact_only"
    completed = subprocess.run([sys.executable, str(entry), "convert", str(source),
        "--output", str(out), "--storage", "compact", "--coplanar", "--compact-vertices"],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (out / "import_lumerical_coplanar_compact.lsf").is_file()
    assert not (out / "import_lumerical_coplanar.lsf").exists()
    repeated = subprocess.run([sys.executable, str(entry), "export-coplanar", str(out), "--compact-vertices"],
                              capture_output=True, text=True)
    assert repeated.returncode == 0, repeated.stderr
    invalid = tmp_path / "invalid_compact"
    assert cli.main(["convert", str(source), "--output", str(invalid), "--compact-vertices"]) == 2
    assert not invalid.exists()


@pytest.mark.parametrize("corrupt,repeat", [("", "_compact"), ("_compact", "")])
def test_other_variant_corruption_is_not_adopted(tmp_path, corrupt, repeat):
    out = converted(tmp_path)
    assert cli.main(["export-coplanar", str(out)]) == 0
    assert cli.main(["export-coplanar", str(out), "--compact-vertices"]) == 0
    (out / ("meshes/material_0002_coplanar" + corrupt + ".mat")).write_bytes(b"corrupted")
    before = snapshot(out)
    command = ["export-coplanar", str(out)] + (["--compact-vertices"] if repeat else [])
    assert cli.main(command) == 2
    assert snapshot(out) == before


def test_keyboard_interrupt_rolls_back_new_artifacts_and_allows_retry(tmp_path, monkeypatch):
    from vdata_standalone import coplanar
    out = converted(tmp_path, multi=True)
    before = snapshot(out)
    original = coplanar.merge_coplanar
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(coplanar, "merge_coplanar", interrupted)
        with pytest.raises(KeyboardInterrupt):
            cli.main(["export-coplanar", str(out)])
    assert len(calls) == 2 and snapshot(out) == before
    assert cli.main(["export-coplanar", str(out)]) == 0


def test_source_changed_during_merge_is_not_rehashed_as_valid(tmp_path, monkeypatch):
    from vdata_standalone import coplanar
    out = converted(tmp_path)
    original = coplanar.merge_coplanar
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with (out / "meshes/material_0002.vmesh").open("ab") as stream:
            stream.write(b"concurrent source edit")
        return result
    manifest = (out / "conversion_manifest.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(coplanar, "merge_coplanar", changed)
        assert cli.main(["export-coplanar", str(out)]) == 2
    assert not list(out.rglob("*_coplanar.mat"))
    assert (out / "conversion_manifest.json").read_bytes() == manifest
