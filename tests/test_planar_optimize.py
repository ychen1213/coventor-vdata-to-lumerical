"""Boundary and topology invariants for conservative plane retriangulation."""

import numpy as np
import pytest

from vdata_standalone.meshing import MeshingError, audit_mesh
from vdata_standalone.planar_optimize import optimize_planar_mesh, _convex_corners


def grid_box(n=8, *, offset=(0, 0, 0), scale=1e-6):
    """Six tessellated faces with welded edge vertices and outward normals."""
    vertices, ids, faces = [], {}, []

    def vertex(point):
        key = tuple(point)
        if key not in ids:
            ids[key] = len(vertices)
            vertices.append(np.asarray(point) / n)
        return ids[key]

    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, n):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        point = [0, 0, 0]
                        point[axis], point[u], point[v] = side, a, b
                        quad.append(vertex(point))
                    triangles = [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
                    faces.extend(t if side else t[::-1] for t in triangles)
    return (np.asarray(vertices) * scale + np.asarray(offset), np.asarray(faces, np.int32))


def voxel_union_shell(cells, *, scale=1e-6):
    """Exactly axis-aligned watertight boundary, including holes or concavity."""
    cells = set(map(tuple, cells))
    vertices, ids, faces = [], {}, []
    for cell in sorted(cells):
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for side in (0, 1):
                neighbour = list(cell)
                neighbour[axis] += 1 if side else -1
                if tuple(neighbour) in cells:
                    continue
                quad = []
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    point = list(cell)
                    point[axis] += side
                    point[u] += a
                    point[v] += b
                    key = tuple(point)
                    if key not in ids:
                        ids[key] = len(vertices)
                        vertices.append(point)
                    quad.append(ids[key])
                triangles = [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
                faces.extend(t if side else t[::-1] for t in triangles)
    return np.asarray(vertices, float) * scale, np.asarray(faces, np.int32)


def coordinate_faces(vertices, faces):
    return {tuple(map(tuple, vertices[face])) for face in faces}


def assert_valid_same_geometry(v0, f0, vertices, faces, report):
    before, after = audit_mesh(v0, f0), audit_mesh(vertices, faces)
    assert after["watertight_edges"] and after["consistent_winding"]
    assert after["degenerate_faces"] == 0
    assert after["euler_characteristic"] == before["euler_characteristic"]
    np.testing.assert_array_equal(after["bounds_local_m"], before["bounds_local_m"])
    assert after["surface_area_m2"] == pytest.approx(before["surface_area_m2"], rel=2e-13)
    assert after["signed_volume_m3"] == pytest.approx(before["signed_volume_m3"], rel=2e-13)
    assert report["input_components"] == report["output_components"]


@pytest.mark.parametrize("index_dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_subdivided_box_locks_boundary_and_reduces_every_axis(index_dtype):
    v0, f0 = grid_box()
    f0 = f0.astype(index_dtype)
    original_bytes = f0.tobytes()
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert report["changed"] and report["accepted_patches"] == 6
    assert len(faces) == 6 * 4 * 8
    assert {p["axis"] for p in report["patches"]} == {"X", "Y", "Z"}
    boundary = v0[np.count_nonzero((v0 == 0) | (v0 == 1e-6), axis=1) >= 2]
    remaining = set(map(tuple, vertices))
    assert all(tuple(v) in remaining for v in boundary)
    assert_valid_same_geometry(v0, f0, vertices, faces, report)
    assert f0.dtype == index_dtype and f0.tobytes() == original_bytes
    assert np.issubdtype(faces.dtype, np.integer)


@pytest.mark.parametrize("index_dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_default_minimum_and_repeat_are_safe(index_dtype):
    v0, f0 = grid_box(n=7)
    f0 = f0.astype(index_dtype)
    vertices, faces, report = optimize_planar_mesh(v0, f0)
    assert not report["changed"]
    assert faces.dtype == f0.dtype
    np.testing.assert_array_equal(vertices, v0)
    np.testing.assert_array_equal(faces, f0)
    v1, f1 = grid_box(n=9)
    v2, f2, first = optimize_planar_mesh(v1, f1)
    v3, f3, second = optimize_planar_mesh(v2, f2)
    assert first["changed"] and not second["changed"]
    np.testing.assert_array_equal(v2, v3)
    np.testing.assert_array_equal(f2, f3)


def test_unused_interior_vertex_is_compacted_and_not_counted_as_component():
    v0, f0 = grid_box(n=8)
    v0 = np.vstack((v0, [0.5e-6, 0.5e-6, 0.5e-6]))
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert report["changed"]
    assert report["input_components"] == report["output_components"] == 1
    assert np.unique(faces).size == len(vertices)


def test_center_plane_value_is_copied_without_averaging():
    v0, f0 = grid_box(n=8, offset=(1.113e-6, -4.21e-6, 7.312e-6))
    vertices, _faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert report["changed"] and report["accepted_patches"] == 6
    # Original referenced vertices keep order; the six new centers follow.
    for center, patch in zip(vertices[-6:], report["patches"]):
        axis = "XYZ".index(patch["axis"])
        expected = np.float64(patch["coordinate_m"])
        assert center[axis].tobytes() == expected.tobytes()


def test_disconnected_same_planes_are_not_bridged():
    va, fa = grid_box(n=6)
    vb, fb = grid_box(n=6, offset=(3e-6, 0, 0))
    v0, f0 = np.vstack((va, vb)), np.vstack((fa, fb + len(va)))
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert report["accepted_patches"] == 12
    assert report["input_components"] == report["output_components"] == 2
    x = vertices[faces, 0]
    assert not np.any((x.min(axis=1) < 1.5e-6) & (x.max(axis=1) > 1.5e-6))
    assert_valid_same_geometry(v0, f0, vertices, faces, report)


def test_tilted_planes_are_preserved():
    v0, f0 = grid_box()
    angle = 0.271
    rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    v0 = v0 @ rz @ rx
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert not report["changed"]
    np.testing.assert_array_equal(vertices, v0)
    np.testing.assert_array_equal(faces, f0)


def test_curved_nonplanar_triangles_are_preserved():
    v0, f0 = grid_box()
    v0 = v0 - 0.5e-6
    v0 = v0 / np.linalg.norm(v0, axis=1)[:, None] * 1e-6
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert not report["changed"]
    np.testing.assert_array_equal(vertices, v0)
    np.testing.assert_array_equal(faces, f0)


def test_hole_top_and_bottom_are_retained():
    n = 8
    cells = [(x, y, 0) for x in range(n) for y in range(n)
             if not (2 <= x < 6 and 2 <= y < 6)]
    v0, f0 = voxel_union_shell(cells)
    original = coordinate_faces(v0, f0[np.ptp(v0[f0, 2], axis=1) == 0])
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=16)
    current = coordinate_faces(vertices, faces[np.ptp(vertices[faces, 2], axis=1) == 0])
    assert original <= current
    assert report["skipped"].get("not_one_boundary_loop", 0) >= 2
    assert_valid_same_geometry(v0, f0, vertices, faces, report)


def test_concave_top_and_bottom_are_retained():
    cells = [(x, y, 0) for x in range(8) for y in range(8) if x < 3 or y < 3]
    v0, f0 = voxel_union_shell(cells)
    original = coordinate_faces(v0, f0[np.ptp(v0[f0, 2], axis=1) == 0])
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=16)
    current = coordinate_faces(vertices, faces[np.ptp(vertices[faces, 2], axis=1) == 0])
    assert original <= current
    assert report["skipped"].get("not_exactly_convex", 0) >= 2
    assert_valid_same_geometry(v0, f0, vertices, faces, report)


def test_exact_dyadic_notch_and_star_rejected():
    assert _convex_corners(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])) is not None
    # A sub-ulp-looking notch must not be rounded away by an angular tolerance.
    notch = np.array([[0., 0.], [0.5, 2.0 ** -60], [1., 0.], [1., 1.], [0., 1.]])
    assert _convex_corners(notch) is None
    angle = np.arange(5) * (2 * np.pi / 5)
    points = np.column_stack((np.cos(angle), np.sin(angle)))
    assert _convex_corners(points[[0, 2, 4, 1, 3]]) is None


def test_rough_faces_survive_byte_for_byte_in_coordinate_space():
    v0, f0 = grid_box(n=8)
    # Lift only the top face's interior; all four sides and bottom stay planar.
    top_inside = ((v0[:, 2] == 1e-6) & (v0[:, 0] > 0) & (v0[:, 0] < 1e-6)
                  & (v0[:, 1] > 0) & (v0[:, 1] < 1e-6))
    v0[top_inside, 2] += 0.07e-6 * np.sin(v0[top_inside, 0] * 2e6) * np.sin(v0[top_inside, 1] * 3e6)
    rough = f0[np.any(top_inside[f0], axis=1)]
    original = coordinate_faces(v0, rough)
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert report["changed"]
    assert original <= coordinate_faces(vertices, faces)
    assert_valid_same_geometry(v0, f0, vertices, faces, report)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_minimum(bad):
    vertices, faces = grid_box()
    with pytest.raises(MeshingError, match="positive integer"):
        optimize_planar_mesh(vertices, faces, min_faces=bad)


def test_invalid_open_and_wrong_winding_inputs_raise():
    vertices, faces = grid_box()
    with pytest.raises(MeshingError, match="closed"):
        optimize_planar_mesh(vertices, faces[1:])
    faces[0] = faces[0, ::-1]
    with pytest.raises(MeshingError, match="closed"):
        optimize_planar_mesh(vertices, faces)


def test_empty_arrays():
    vertices, faces, report = optimize_planar_mesh(np.empty((0, 3)), np.empty((0, 3), int))
    assert not report["changed"] and vertices.shape == faces.shape == (0, 3)


def test_failed_final_audit_rolls_back(monkeypatch):
    import vdata_standalone.planar_optimize as module
    original_audit = module.audit_mesh
    calls = 0

    def bad_final(vertices, faces):
        nonlocal calls
        report = original_audit(vertices, faces)
        calls += 1
        if calls == 2:
            report["euler_characteristic"] += 2
        return report

    monkeypatch.setattr(module, "audit_mesh", bad_final)
    v0, f0 = grid_box()
    vertices, faces, report = optimize_planar_mesh(v0, f0, min_faces=32)
    assert not report["changed"] and report["rolled_back"]
    assert "Euler" in report["rollback_reason"]
    np.testing.assert_array_equal(vertices, v0)
    np.testing.assert_array_equal(faces, f0)
