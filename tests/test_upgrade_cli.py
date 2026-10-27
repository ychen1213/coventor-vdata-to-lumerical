"""Integration boundaries for streamed ROI and geometry-preserving postprocess."""
import json

import numpy as np
import pytest

from test_cli import write_fixture
from vdata_standalone import cli
from vdata_standalone.meshing import mesh_material
from vdata_standalone.reader import inspect_vdata
from vdata_standalone.storage import read_mesh_archive


def test_crop_decodes_region_once_and_preserves_original_mesh(tmp_path):
    source = tmp_path / 'input.vdata'
    original = write_fixture(source)
    out = tmp_path / 'crop'
    crop = [[1, 7], [1, 8], [1, 6]]
    expected_v, expected_f, expected_report = mesh_material(
        original, inspect_vdata(source), 2, crop_indices=crop)
    assert cli.main(['convert', str(source), '--output', str(out), '--storage', 'compact',
                     '--crop-um', '16.02', '16.14', '-13.98', '-13.84', '-.48', '-.38']) == 0
    actual_v, actual_f, actual_t, _ = read_mesh_archive(out / 'meshes/material_0002.vmesh')
    np.testing.assert_array_equal(actual_v, expected_v)
    np.testing.assert_array_equal(actual_f, expected_f)
    np.testing.assert_array_equal(actual_t, expected_report['translation_m'])
    manifest = json.loads((out / 'conversion_manifest.json').read_text(encoding='utf-8'))
    assert manifest['source_metadata']['dimensions_xyz'] == [8, 9, 7]
    assert manifest['source_metadata']['raw_shape'] == [6, 7, 5, 3]
    assert manifest['crop']['dimensions_xyz'] == [6, 7, 5]
    assert manifest['objects'][0]['reconstruction']['source_crop_indices_xyz'] == crop
    with np.load(out / 'voxels.npz') as data:
        np.testing.assert_array_equal(data['raw'], original[1:7, 1:8, 1:6])
    assert not (out / '.decode-cache.npy').exists()


def test_planar_option_preserves_geometry_audit_and_exports_current_face_count(tmp_path):
    source = tmp_path / 'input.vdata'
    write_fixture(source)
    out = tmp_path / 'optimized'
    assert cli.main(['convert', str(source), '--output', str(out), '--storage', 'compact',
                     '--planar-optimize', '--planar-min-faces', '8']) == 0
    v, f, _, _ = read_mesh_archive(out / 'meshes/material_0002.vmesh')
    manifest = json.loads((out / 'conversion_manifest.json').read_text(encoding='utf-8'))
    obj = manifest['objects'][0]
    assert obj['exported_face_count'] == len(f)
    assert obj['exported_vertex_count'] == len(v)
    assert obj['exported_face_count'] < obj['reconstruction']['faces']
    assert 'planar_optimization' in obj['reconstruction']
    assert cli.main(['export-planar', str(out)]) == 0
    assert cli.main(['export-stl', str(out)]) == 0


def test_optimize_existing_mesh_does_not_need_original_vdata(tmp_path):
    source = tmp_path / 'input.vdata'
    write_fixture(source)
    out = tmp_path / 'original'
    assert cli.main(['convert', str(source), '--output', str(out), '--storage', 'compact']) == 0
    mesh = out / 'meshes/material_0002.vmesh'
    before = mesh.read_bytes()
    source.unlink()
    optimized = tmp_path / 'smaller.vmesh'
    assert cli.main(['optimize-mesh', str(mesh), '--output', str(optimized), '--min-faces', '8']) == 0
    _, old_f, old_t, _ = read_mesh_archive(mesh)
    _, new_f, new_t, _ = read_mesh_archive(optimized)
    assert len(new_f) < len(old_f)
    np.testing.assert_array_equal(old_t, new_t)
    assert mesh.read_bytes() == before
    assert cli.main(['optimize-mesh', str(mesh), '--output', str(optimized)]) == 2


def test_planar_multi_material_refuses_independent_interface_changes(tmp_path):
    source = tmp_path / 'multi.vdata'
    write_fixture(source, with_second_material=True)
    out = tmp_path / 'out'
    assert cli.main(['convert', str(source), '--output', str(out), '--planar-optimize']) == 2
    manifest = json.loads((out / 'conversion_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'failed'
    assert not list(out.rglob('*.stl'))
    assert not (out / '.decode-cache.npy').exists()


def test_bad_planar_threshold_refused_before_output(tmp_path):
    source = tmp_path / 'input.vdata'
    write_fixture(source)
    out = tmp_path / 'out'
    assert cli.main(['convert', str(source), '--output', str(out), '--planar-optimize',
                     '--planar-min-faces', '0']) == 2
    assert not out.exists()


@pytest.mark.parametrize('surface', ['fraction', 'vof'])
def test_active_mask_cli_matches_default(surface, tmp_path):
    source = tmp_path / 'input.vdata'
    write_fixture(source)
    arrays = []
    for name, options in [('normal', []), ('active', ['--active-blocks'])]:
        out = tmp_path / name
        assert cli.main(['convert', str(source), '--output', str(out), '--storage', 'compact',
                         '--surface', surface, *options]) == 0
        arrays.append(read_mesh_archive(out / 'meshes/material_0002.vmesh')[:3])
    for actual, expected in zip(*arrays):
        np.testing.assert_array_equal(actual, expected)
