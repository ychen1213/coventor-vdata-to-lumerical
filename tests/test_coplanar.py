"""Surface/topology invariants for exact boundary-preserving CAD polygons."""

import json

import numpy as np
import pytest

from vdata_standalone.coplanar import merge_coplanar, _simple_polygon
from vdata_standalone.meshing import MeshingError


def voxel_shell(cells):
    """Weld the outward shell of integer unit cubes, triangulating every square."""
    cells = set(map(tuple, cells))
    vertices, lookup, triangles = [], {}, []
    for cell in sorted(cells):
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for side in (0, 1):
                neighbor = list(cell)
                neighbor[axis] += 1 if side else -1
                if tuple(neighbor) in cells:
                    continue
                quad = []
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    point = list(cell)
                    point[axis] += side
                    point[u] += a
                    point[v] += b
                    point = tuple(point)
                    if point not in lookup:
                        lookup[point] = len(vertices)
                        vertices.append(point)
                    quad.append(lookup[point])
                two = [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
                triangles.extend(t if side else t[::-1] for t in two)
    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int32)


def box(n=2):
    return voxel_shell((i, j, k) for i in range(n) for j in range(n) for k in range(n))


def facets(offsets, indices):
    return [indices[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def signed_volume(vertices, loops):
    # The fan formula computes oriented integral even for a concave polygon;
    # this is independent of the implementation's exact boundary proof.
    result = 0.
    for loop in loops:
        p = vertices[loop]
        result += np.dot(p[0], np.sum(np.cross(p - p[0], np.roll(p - p[0], -1, axis=0)), axis=0)) / 6
    return result


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_box_merges_faces_keeps_every_boundary_point_and_inputs(dtype):
    v, f = box()
    f = f.astype(dtype)
    vb, fb = v.tobytes(), f.tobytes()
    offsets, indices, report = merge_coplanar(v, f)
    loops = facets(offsets, indices)
    assert len(loops) == 6
    assert all(len(loop) == 8 for loop in loops)
    expected = set(np.flatnonzero(np.count_nonzero((v == 0) | (v == 2), axis=1) >= 2))
    assert set(indices) == expected
    assert v.tobytes() == vb and f.tobytes() == fb and f.dtype == dtype
    assert signed_volume(v, loops) == pytest.approx(signed_volume(v, f), rel=2e-14)
    assert report["original_triangles_replaced"] == len(f)
    assert report["audit"]["passed"]
    assert report["coordinate_movement"] == report["coplanarity_tolerance"] == 0
    json.dumps(report, allow_nan=False)


def test_arbitrary_exact_tilt_and_anisotropic_dyadic_scaling():
    v, f = box()
    matrix = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float)
    v = (v @ matrix) * np.array([2.**-21, 2.**-19, 2.**-23])
    v += np.array([2.**-14, -2.**-13, 2.**-15])
    offsets, indices, report = merge_coplanar(v, f)
    assert len(offsets) - 1 == 6
    assert all(len(loop) == 8 for loop in facets(offsets, indices))
    assert report["merged_polygon_facets"] == 6
    assert signed_volume(v, facets(offsets, indices)) == pytest.approx(signed_volume(v, f), rel=1e-13)


def test_one_ulp_departure_is_not_merged_into_flat_top():
    v, f = box()
    center = np.flatnonzero(np.all(v == [1, 1, 2], axis=1))[0]
    v[center, 2] = np.nextafter(v[center, 2], np.inf)
    offsets, indices, report = merge_coplanar(v, f)
    loops = facets(offsets, indices)
    assert len(loops) > 6
    # No facet contains both the raised point and three noncollinear points of
    # the unperturbed top plane, even though the departure is only one ULP.
    for loop in loops:
        if center in loop:
            points = v[loop]
            if len(loop) > 3:
                top = points[points[:, 2] == 2]
                assert len(top) <= 2 or np.linalg.matrix_rank(top - top[0]) <= 1
    assert report["audit"]["passed"]


def test_concave_single_loop_is_supported_without_changing_volume():
    v, f = voxel_shell([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    offsets, indices, report = merge_coplanar(v, f)
    loops = facets(offsets, indices)
    horizontal = [loop for loop in loops if np.ptp(v[loop, 2]) == 0]
    assert len(horizontal) == 2
    assert all(len(loop) == 8 for loop in horizontal)
    assert len(loops) == 8
    assert signed_volume(v, loops) == 3
    assert report["fallback_components"].get("invalid_projected_polygon", 0) == 0


def test_annulus_keeps_original_top_and_bottom_triangles():
    v, f = voxel_shell((i, j, 0) for i in range(3) for j in range(3) if (i, j) != (1, 1))
    offsets, indices, report = merge_coplanar(v, f)
    loops = facets(offsets, indices)
    assert report["fallback_components"]["multiple_boundary_loops"] == 2
    for height in (0, 1):
        original = {tuple(row) for row in f if np.all(v[row, 2] == height)}
        output = {tuple(loop) for loop in loops if np.all(v[loop, 2] == height)}
        assert output == original
    assert signed_volume(v, loops) == pytest.approx(8, rel=2e-14)
    assert report["audit"]["passed"]


def test_disconnected_solids_and_unused_vertices_are_preserved():
    va, fa = box(n=1)
    vb, fb = box(n=1)
    v = np.vstack((va, vb + [4, 0, 0], [[99, 99, 99]]))
    f = np.vstack((fa, fb + len(va)))
    offsets, indices, report = merge_coplanar(v, f)
    assert len(offsets) - 1 == 12
    assert len(v) - 1 not in indices
    assert report["vertex_compaction"] is False
    for loop in facets(offsets, indices):
        assert np.all(loop < len(va)) or np.all(loop >= len(va))


def test_minimum_preserves_original_faces_exactly_and_repeat_is_deterministic():
    v, f = box(n=1)
    offsets, indices, report = merge_coplanar(v, f, min_faces=3)
    np.testing.assert_array_equal(indices.reshape(-1, 3), f)
    assert not report["changed"]
    a, b, _ = merge_coplanar(v, f)
    c, d, _ = merge_coplanar(v, f)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)


@pytest.mark.parametrize("value", [True, False, 0, 1, -1, 2.0, "2"])
def test_bad_minimum_rejected(value):
    v, f = box(n=1)
    with pytest.raises(MeshingError, match="min_faces"):
        merge_coplanar(v, f, min_faces=value)


@pytest.mark.parametrize("problem", ["open", "nonmanifold", "winding", "duplicate_index", "collinear", "nonfinite", "out_of_bounds", "empty"])
def test_invalid_input_rejected(problem):
    v, f = box(n=1)
    if problem == "open":
        f = f[:-1]
    elif problem == "nonmanifold":
        f = np.vstack((f, f[:1]))
    elif problem == "winding":
        f[0] = f[0, ::-1]
    elif problem == "duplicate_index":
        f[0, 1] = f[0, 0]
    elif problem == "collinear":
        v[:] = np.column_stack((v[:, 0], np.zeros((len(v), 2))))
    elif problem == "nonfinite":
        v[0, 0] = np.nan
    elif problem == "out_of_bounds":
        f[0, 0] = len(v)
    elif problem == "empty":
        v, f = v[:0], f[:0]
    with pytest.raises(MeshingError):
        merge_coplanar(v, f)


def test_exact_boundary_simple_check_allows_straight_points_and_rejects_crossings():
    assert _simple_polygon([(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)])
    assert not _simple_polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    assert not _simple_polygon([(0, 0), (2, 0), (1, 0), (2, 2), (0, 2)])
    assert not _simple_polygon([(0, 0), (1, 0), (0, 0), (0, 1)])


def test_large_integer_predicates_do_not_overflow_fixed_width():
    v, f = box(n=1)
    v *= np.array([2.**300, 2.**-300, 2.**100])
    offsets, _indices, report = merge_coplanar(v, f)
    assert len(offsets) - 1 == 6
    assert report["audit"]["passed"]


def test_subnormal_coordinates_are_handled_as_exact_dyadics():
    v, f = box(n=1)
    v[:, 0] *= np.nextafter(0., 1.)
    offsets, _indices, report = merge_coplanar(v, f)
    assert len(offsets) - 1 == 6
    assert max(report["integer_axis_denominator_bits"]) == 1075


def test_non_coplanar_tetrahedron_retains_all_triangles():
    v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int64)
    offsets, indices, report = merge_coplanar(v, f)
    assert not report["changed"]
    np.testing.assert_array_equal(offsets, np.arange(5) * 3)
    np.testing.assert_array_equal(indices.reshape(-1, 3), f)


def test_opposite_normals_are_not_joined_despite_exact_coplanarity():
    # A flat tetrahedron is closed in its edge graph but geometrically overlaps.
    # The optimizer does not promise a global self-intersection check.  It must
    # nevertheless never join the opposing sheets to one zero-area polygon.
    v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int64)
    offsets, indices, report = merge_coplanar(v, f)
    assert report["opposed_coplanar_adjacent_pairs_not_joined"] > 0
    assert len(offsets) - 1 == 2
    assert all(len(loop) == 4 for loop in facets(offsets, indices))
    assert report["audit"]["global_self_intersections_checked"] is False
