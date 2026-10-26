"""Numerical and boundary checks for the vendor-independent mesher."""

import struct

import numpy as np
import pytest

from vdata_standalone.meshing import (
    MeshingError, audit_mesh, mesh_material, simplify_mesh, write_binary_stl,
)


def solid(dims=(4, 5, 6), *, material_id=2, spacing=20e-9, origin=(0, 0, 0)):
    raw = np.empty((*dims, 3), dtype=np.uint8)
    raw[..., 0] = 129
    raw[..., 1:] = material_id
    return raw, {"dimensions_xyz": list(dims), "voxel_size_m": spacing,
                 "origin_m": list(origin), "has_plane_equations": False,
                 "materials": [{"id": 2, "name": "Si", "type": "Solid"}]}


def test_box_coordinates_closure_winding_and_volume_warning():
    spacing = np.array([2e-9, 3e-9, 5e-9])
    raw, meta = solid((4, 5, 6), spacing=spacing)
    vertices, faces, report = mesh_material(raw, meta, 2)
    np.testing.assert_allclose(vertices.min(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(vertices.max(axis=0), spacing * [4, 5, 6], rtol=1e-6)
    assert report["watertight_edges"] and report["consistent_winding"]
    assert report["boundary_edges"] == report["nonmanifold_edges"] == 0
    assert report["degenerate_faces"] == 0
    assert report["signed_volume_m3"] > 0
    assert report["euler_characteristic"] == 2
    assert report["volume_m3"] < report["source_fraction_volume_m3"]
    assert report["volume_relative_difference"] < 0
    assert any("not the exact union" in warning for warning in report["warnings"])
    assert vertices.dtype == np.float64 and np.issubdtype(faces.dtype, np.integer)


def test_large_world_origin_never_enters_marching_cubes_coordinates():
    raw, meta = solid()
    vertices0, faces0, report0 = mesh_material(raw, meta, 2)
    meta["origin_m"] = [1e6, -2e6, 3e6]
    vertices1, faces1, report1 = mesh_material(raw, meta, 2)
    np.testing.assert_array_equal(vertices1, vertices0)
    np.testing.assert_array_equal(faces1, faces0)
    assert report1["translation_m"] == meta["origin_m"]
    assert report0["translation_m"] == [0, 0, 0]


def test_fraction_interpolation_and_xyz_axis_order():
    raw, meta = solid((3, 4, 5), spacing=1e-6)
    raw[..., 1] = 2
    raw[..., 2] = 255
    raw[0, ..., 0] = 0
    raw[1, ..., 0] = 32  # 0.25, not converted to a binary occupied voxel
    raw[2, ..., 0] = 96  # 0.75
    vertices, _faces, report = mesh_material(raw, meta, 2)
    assert vertices[:, 0].min() == pytest.approx(2e-6, rel=1e-6)
    assert vertices[:, 0].max() == pytest.approx((2.5 + 1 / 3) * 1e-6, rel=1e-6)
    assert report["source_fraction_volume_m3"] == pytest.approx(20e-18)
    assert vertices[:, 1].max() > vertices[:, 0].max()
    assert vertices[:, 2].max() > vertices[:, 1].max()


def test_crop_is_closed_and_translation_preserved():
    raw, meta = solid((6, 7, 8), spacing=10e-9, origin=(1e-6, -2e-6, 3e-6))
    vertices, _faces, report = mesh_material(raw, meta, 2,
                                             crop_indices=((1, 5), (2, 6), (3, 7)))
    np.testing.assert_allclose(vertices.min(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(vertices.max(axis=0), 40e-9, rtol=1e-6)
    np.testing.assert_allclose(report["translation_m"], np.array(meta["origin_m"]) + np.array([1, 2, 3]) * 10e-9)
    assert report["watertight_edges"]
    assert any("artificial closed cut" in text for text in report["warnings"])
    alternative = mesh_material(raw, meta, 2, crop_indices=((1, 2, 3), (5, 6, 7)))
    np.testing.assert_array_equal(alternative[0], vertices)


@pytest.mark.parametrize("shape", [(7, 8, 9), (8, 9, 10), (3, 4, 5)])
def test_stride_padding_still_closes_boundary(shape):
    raw, meta = solid(shape)
    _vertices, _faces, report = mesh_material(raw, meta, 2, step_size=2)
    assert report["watertight_edges"] and report["consistent_winding"]
    assert any("discard features" in warning for warning in report["warnings"])


def test_empty_material():
    raw, meta = solid(material_id=255)
    vertices, faces, report = mesh_material(raw, meta, 2)
    assert vertices.shape == faces.shape == (0, 3)
    assert report["empty"] and report["volume_m3"] == 0
    assert not report["watertight_edges"]


def test_audit_detects_holes_nonmanifold_and_wrong_winding():
    raw, meta = solid()
    vertices, faces, _ = mesh_material(raw, meta, 2)
    good = audit_mesh(vertices, faces, chunk_size=1)
    assert good["watertight_edges"] and good["consistent_winding"]
    hole = audit_mesh(vertices, faces[1:], chunk_size=7)
    assert hole["boundary_edges"] == 3
    duplicate = audit_mesh(vertices, np.concatenate([faces, faces[:1]]), chunk_size=13)
    assert duplicate["nonmanifold_edges"] == 3
    reversed_face = faces.copy()
    reversed_face[0] = reversed_face[0, [0, 2, 1]]
    wrong = audit_mesh(vertices, reversed_face, chunk_size=9)
    assert wrong["inconsistent_winding_edges"] == 3
    assert wrong["watertight_edges"]


def test_stl_binary_layout_normals_units_and_quantization(tmp_path):
    raw, meta = solid()
    vertices, faces, _ = mesh_material(raw, meta, 2)
    target = tmp_path / "box.stl"
    report = write_binary_stl(target, vertices, faces, chunk_size=7)
    assert target.stat().st_size == 84 + len(faces) * 50
    data = target.read_bytes()
    assert struct.unpack("<I", data[80:84])[0] == len(faces)
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.frombuffer(data, dtype=dtype, offset=84)
    tri = records["vertices"].astype(np.float64)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    np.testing.assert_allclose(records["normal"], normal, atol=1e-7)
    np.testing.assert_allclose(tri, vertices[faces] / 1e-6, rtol=1e-6, atol=1e-9)
    assert report["coordinate_unit"] == "um"
    assert report["max_vertex_float32_quantization_error_m"] < 1e-13
    assert not report["translation_stored_in_stl"]


def test_stl_failed_quantization_preserves_existing_target(tmp_path):
    target = tmp_path / "existing.stl"
    target.write_bytes(b"original")
    # These vertices coincide after float32 rounding in any ordinary unit.
    vertices = np.array([[1, 1, 1], [1 + 1e-10, 1, 1], [1, 1 + 1e-10, 1]])
    with pytest.raises(MeshingError, match="collapsed"):
        write_binary_stl(target, vertices, np.array([[0, 1, 2]]))
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("kwargs", [{"level": 0}, {"level": 1}, {"level": float("nan")},
                                    {"step_size": 0}, {"step_size": 1.5},
                                    {"crop_indices": ((0, 0), (0, 1), (0, 1))},
                                    {"crop_indices": ((0, 8), (0, 1), (0, 1))}])
def test_invalid_extraction_arguments(kwargs):
    raw, meta = solid()
    with pytest.raises(MeshingError):
        mesh_material(raw, meta, 2, **kwargs)


def test_invalid_geometry_arguments():
    with pytest.raises(MeshingError, match="index"):
        audit_mesh(np.zeros((3, 3)), np.array([[0, 1, 8]]))
    with pytest.raises(MeshingError, match="integer"):
        audit_mesh(np.zeros((3, 3)), np.array([[0.0, 1, 2]]))


def test_qem_reduction_validates_both_surfaces_and_reports_sampling():
    pytest.importorskip("fast_simplification")
    pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    raw, meta = solid((12, 13, 14))
    vertices, faces, _ = mesh_material(raw, meta, 2)
    result_vertices, result_faces, report = simplify_mesh(vertices, faces, .45, sample_count=100)
    assert len(result_faces) < len(faces)
    assert report["source_audit"]["watertight_edges"]
    assert report["reduced_audit"]["watertight_edges"]
    assert report["source_audit"]["euler_characteristic"] == report["reduced_audit"]["euler_characteristic"]
    assert report["qem_aggressiveness"] == 3.0
    distance = report["sampled_bidirectional_distance"]
    assert distance["samples_each_direction"] == 100
    assert not distance["is_strict_hausdorff_bound"]
    assert not distance["is_optical_position_accuracy_validation"]
    assert np.isfinite(result_vertices).all()


def test_qem_zero_is_identity():
    raw, meta = solid()
    vertices, faces, _ = mesh_material(raw, meta, 2)
    v2, f2, report = simplify_mesh(vertices, faces, 0, sample_count=4)
    np.testing.assert_array_equal(v2, vertices)
    np.testing.assert_array_equal(f2, faces)
    assert report["geometry_unchanged"]


def test_qem_rejects_nonmanifold_output_with_explicit_counts(monkeypatch):
    simplifier = pytest.importorskip("fast_simplification")
    raw, meta = solid()
    vertices, faces, _report = mesh_material(raw, meta, 2)

    def invalid_result(points, source_faces, **_kwargs):
        return points, np.concatenate([source_faces, source_faces[:1]])

    monkeypatch.setattr(simplifier, "simplify", invalid_result)
    with pytest.raises(MeshingError, match="nonmanifold_edges=3"):
        simplify_mesh(vertices, faces, .45, sample_count=4)
