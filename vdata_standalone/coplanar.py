"""Replace exact coplanar triangular disks by their unchanged boundary loops.

This is a surface-preserving representation change, not surface reconstruction
or approximate mesh decimation.  Coordinates are never fitted or moved.  Every
merge predicate uses integer arithmetic on the exact dyadic values of the input
float64 coordinates.  Holes and unsupported patch topology retain their original
triangles.  NumPy, SciPy and Shapely are required; no CAD or simulator API is used.
"""

from __future__ import annotations

from collections import Counter
import time
from typing import Any

import numpy as np

from .meshing import MeshingError, _mesh_arrays


def _integer_points(vertices: np.ndarray):
    """One positive, exact power-of-two scale per coordinate axis."""
    axes = []
    denominator_bits = []
    for axis in range(3):
        values, inverse = np.unique(vertices[:, axis], return_inverse=True)
        ratios = [float(value).as_integer_ratio() for value in values]
        denominator = max(d for _n, d in ratios)
        table = [n * (denominator // d) for n, d in ratios]
        axes.append([table[int(i)] for i in inverse])
        denominator_bits.append(denominator.bit_length())
    return list(zip(*axes)), denominator_bits


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _intersects(a, b, c, d):
    """Exact closed-segment intersection in the projected integer plane."""
    ab_c, ab_d = _orient(a, b, c), _orient(a, b, d)
    cd_a, cd_b = _orient(c, d, a), _orient(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    return ((ab_c == 0 and _on_segment(a, b, c))
            or (ab_d == 0 and _on_segment(a, b, d))
            or (cd_a == 0 and _on_segment(c, d, a))
            or (cd_b == 0 and _on_segment(c, d, b)))


def _simple_polygon(points):
    """GEOS broad phase followed by exact intersection tests; never a tolerance.

    Rounding the normalized coordinates is monotone on each axis, so their edge
    bounding boxes cannot lose an overlap of exact boxes.  GEOS validity can
    conservatively reject a very ill-conditioned loop.  Acceptance additionally
    checks every broad-phase candidate with integer orientation predicates.
    """
    import shapely
    from shapely.geometry import Polygon

    n = len(points)
    if n < 3 or len(set(points)) != n:
        return False
    lower = tuple(min(p[k] for p in points) for k in range(2))
    span = tuple(max(p[k] for p in points) - lower[k] for k in range(2))
    if not all(span):
        return False
    xy = np.asarray([[(p[k] - lower[k]) / span[k] for k in range(2)]
                     for p in points], dtype=np.float64)
    polygon = Polygon(xy)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        return False
    # Adjacent edges may share only their common endpoint.  Keep collinear
    # points along a straight boundary, but reject collinear backtracking.
    for i, b in enumerate(points):
        a, c = points[i - 1], points[(i + 1) % n]
        if _orient(a, b, c) == 0:
            if sum((a[k] - b[k]) * (c[k] - b[k]) for k in range(2)) >= 0:
                return False
    if n <= 3:
        return True
    lines = shapely.linestrings(np.stack((xy, np.roll(xy, -1, axis=0)), axis=1))
    tree = shapely.STRtree(lines)
    # Chunk queries to avoid a quadratic-sized temporary for pathological rings.
    for begin in range(0, n, 1024):
        pairs = tree.query(lines[begin:begin + 1024])
        for local_i, raw_j in zip(*pairs):
            i, j = begin + int(local_i), int(raw_j)
            if i >= j or j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _intersects(points[i], points[(i + 1) % n],
                           points[j], points[(j + 1) % n]):
                return False
    return True


def _boundary_loops(a, b):
    if len(a) < 3:
        return None
    order = np.argsort(a)
    aa, bb = a[order], b[order]
    if np.any(aa[1:] == aa[:-1]) or not np.array_equal(aa, np.sort(bb)):
        return None
    following = np.searchsorted(aa, bb)
    unseen = np.ones(len(aa), dtype=np.bool_)
    paths = []
    for start in range(len(aa)):
        if not unseen[start]:
            continue
        path, at = [], start
        while unseen[at]:
            unseen[at] = False
            path.append(int(aa[at]))
            at = int(following[at])
        if at != start or len(path) < 3:
            return None
        paths.append(np.asarray(path, dtype=np.int64))
    return paths


def _adjacency(faces, n_vertices):
    """Sorted edge positions, after strict closure and winding validation."""
    if n_vertices * n_vertices > np.iinfo(np.uint64).max:
        raise MeshingError("too many vertices for exact coplanar edge indexing")
    keys = np.empty(3 * len(faces), dtype=np.uint64)
    for edge in range(3):
        a, b = faces[:, edge], faces[:, (edge + 1) % 3]
        keys[edge::3] = (np.minimum(a, b).astype(np.uint64) * np.uint64(n_vertices)
                         + np.maximum(a, b).astype(np.uint64))
    order = np.argsort(keys)
    keys = keys[order]
    if (len(keys) % 2 or not np.array_equal(keys[::2], keys[1::2])
            or np.any(keys[2::2] == keys[:-2:2])):
        raise MeshingError("coplanar merging requires a closed mesh with exactly two faces per edge")
    left, right = order[::2], order[1::2]
    a, b = left // 3, right // 3
    ea, eb = left % 3, right % 3
    if (not np.array_equal(faces[a, ea], faces[b, (eb + 1) % 3])
            or not np.array_equal(faces[a, (ea + 1) % 3], faces[b, eb])):
        raise MeshingError("coplanar merging requires consistent winding at every shared edge")
    if np.any(a == b):
        raise MeshingError("coplanar merging rejects degenerate repeated-index triangles")
    return order


def _audit_csr(offsets, indices, n_vertices):
    ends = np.roll(indices, -1)
    ends[offsets[1:] - 1] = indices[offsets[:-1]]
    keys = (np.minimum(indices, ends).astype(np.uint64) * np.uint64(n_vertices)
            + np.maximum(indices, ends).astype(np.uint64))
    order = np.argsort(keys)
    keys = keys[order]
    closed = (len(keys) % 2 == 0 and np.array_equal(keys[::2], keys[1::2])
              and not np.any(keys[2::2] == keys[:-2:2]))
    a, b = indices[order], ends[order]
    wound = (closed and np.array_equal(a[::2], b[1::2])
             and np.array_equal(b[::2], a[1::2]))
    return {"closed_two_incidence_edges": bool(closed),
            "consistent_winding": bool(wound),
            "valid_csr": bool(offsets[0] == 0 and offsets[-1] == len(indices)
                              and np.all(np.diff(offsets) >= 3)),
            "zero_length_index_edges": int(np.count_nonzero(indices == ends))}


def merge_coplanar(vertices: Any, faces: Any, *, min_faces: int = 2):
    """Return ``(facet_offsets, facet_indices, report)`` for single-loop facets.

    CSR offsets and vertex IDs are zero-based int64; offsets are half-open.
    Vertex IDs refer to the *unchanged original vertex table*, including unused
    vertices.  Retained triangles appear first in original order, then accepted
    polygons in component order.  Every original patch boundary point is kept.

    The input must be finite, nonempty, closed at edges, consistently wound and
    exactly nondegenerate.  Invalid input raises :class:`MeshingError`.  Within
    valid input, an unsupported patch (holes, non-disk topology, invalid boundary)
    safely retains all its original triangles.  Disconnected solids are allowed.
    Global self-intersections and physical reconstruction accuracy are not tested.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    import shapely

    started = time.perf_counter()
    if (isinstance(min_faces, (bool, np.bool_))
            or not isinstance(min_faces, (int, np.integer)) or min_faces < 2):
        raise MeshingError("min_faces must be an integer of at least 2")
    vertices, faces = _mesh_arrays(vertices, faces)
    if not len(vertices) or not len(faces):
        raise MeshingError("coplanar merging requires a nonempty closed mesh")
    faces = faces.astype(np.int64, copy=False)
    order = _adjacency(faces, len(vertices))
    times = {"input_edge_validation": time.perf_counter() - started}
    at = time.perf_counter()
    points, denominator_bits = _integer_points(vertices)
    times["integer_coordinates"] = time.perf_counter() - at
    at = time.perf_counter()
    left, right = order[::2], order[1::2]
    exact = np.zeros(len(left), dtype=np.bool_)
    opposite_normals = 0
    for j, (raw_a, raw_b) in enumerate(zip(left, right)):
        face_a, edge_a = divmod(int(raw_a), 3)
        face_b, edge_b = divmod(int(raw_b), 3)
        p = points[int(faces[face_a, edge_a])]
        q = points[int(faces[face_a, (edge_a + 1) % 3])]
        c = points[int(faces[face_a, (edge_a + 2) % 3])]
        d = points[int(faces[face_b, (edge_b + 2) % 3])]
        u, v, w = _sub(q, p), _sub(c, p), _sub(d, p)
        normal = _cross(u, v)
        nonzero = [k for k in range(3) if normal[k]]
        if not nonzero:
            raise MeshingError("coplanar merging rejects exactly degenerate triangles")
        if sum(normal[k] * w[k] for k in nonzero):
            continue
        # Both triangles are in the same exact plane.  They must lie on opposite
        # sides of their shared directed edge to have the same surface normal.
        k = nonzero[0]
        second = _cross(u, w)[k]
        if not second:
            raise MeshingError("coplanar merging rejects exactly degenerate triangles")
        if normal[k] * second < 0:
            exact[j] = True
        else:
            opposite_normals += 1
    times["exact_adjacent_plane_predicates"] = time.perf_counter() - at
    at = time.perf_counter()
    fa, fb = left // 3, right // 3
    dtype = np.int32 if len(faces) <= np.iinfo(np.int32).max else np.int64
    graph = coo_matrix((np.ones(int(exact.sum()), dtype=np.uint8),
                        (fa[exact].astype(dtype), fb[exact].astype(dtype))),
                       shape=(len(faces), len(faces))).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    del graph
    sizes = np.bincount(labels, minlength=n_components)
    face_order = np.argsort(labels, kind="stable")
    face_offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    # Every edge between different components becomes two directed boundaries.
    different = labels[fa] != labels[fb]
    positions = order.reshape(-1, 2)[different].reshape(-1)
    bf, be = positions // 3, positions % 3
    boundary_labels = labels[bf]
    boundary_order = np.argsort(boundary_labels, kind="stable")
    ba = faces[bf, be][boundary_order]
    bb = faces[bf, (be + 1) % 3][boundary_order]
    counts = np.bincount(boundary_labels, minlength=n_components)
    boundary_offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    del order, left, right, exact, fa, fb, positions, bf, be
    times["components_and_boundaries"] = time.perf_counter() - at
    at = time.perf_counter()
    replaced = np.zeros(len(faces), dtype=np.bool_)
    polygons = []
    skipped = Counter()
    skipped_faces = Counter()
    exact_plane_points = 0
    max_loop = 3
    for component in range(n_components):
        size = int(sizes[component])
        if size < min_faces:
            skipped["below_min_faces"] += 1
            skipped_faces["below_min_faces"] += size
            continue
        first, last = boundary_offsets[component:component + 2]
        a, b = ba[first:last], bb[first:last]
        loops = _boundary_loops(a, b)
        reason = None
        if loops is None:
            reason = "invalid_boundary_cycles"
        elif len(loops) != 1:
            reason = "multiple_boundary_loops"
        selected = face_order[face_offsets[component]:face_offsets[component + 1]]
        if reason is None:
            loop = loops[0]
            used = np.unique(faces[selected])
            if ((3 * size + len(loop)) % 2
                    or len(used) - (3 * size + len(loop)) // 2 + size != 1):
                reason = "not_topological_disk"
        if reason is None:
            ref = faces[selected[0]]
            p0, p1, p2 = (points[int(i)] for i in ref)
            normal = _cross(_sub(p1, p0), _sub(p2, p0))
            for vertex_id in used:
                p = points[int(vertex_id)]
                if sum((p[k] - p0[k]) * normal[k] for k in range(3)):
                    reason = "component_not_exactly_planar"
                    break
            exact_plane_points += len(used)
        if reason is None:
            drop = max(range(3), key=lambda k: abs(normal[k]))
            keep = [k for k in range(3) if k != drop]
            projected = [tuple(points[int(i)][k] for k in keep) for i in loop]
            if not _simple_polygon(projected):
                reason = "invalid_projected_polygon"
        if reason is None:
            # Compare directed edge multisets independently of cycle traversal.
            original = np.sort(a.astype(np.uint64) * np.uint64(len(vertices))
                               + b.astype(np.uint64))
            boundary = np.sort(loop.astype(np.uint64) * np.uint64(len(vertices))
                               + np.roll(loop, -1).astype(np.uint64))
            if not np.array_equal(original, boundary):
                reason = "directed_boundary_mismatch"
        if reason is not None:
            skipped[reason] += 1
            skipped_faces[reason] += size
            continue
        replaced[selected] = True
        polygons.append(loop)
        max_loop = max(max_loop, len(loop))
    times["patch_geometry_audit"] = time.perf_counter() - at
    at = time.perf_counter()
    retained = faces[~replaced]
    lengths = np.r_[np.full(len(retained), 3, dtype=np.int64),
                    np.asarray([len(loop) for loop in polygons], dtype=np.int64)]
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    indices = np.empty(int(offsets[-1]), dtype=np.int64)
    n_retained = retained.size
    indices[:n_retained] = retained.reshape(-1)
    cursor = n_retained
    for loop in polygons:
        indices[cursor:cursor + len(loop)] = loop
        cursor += len(loop)
    audit = _audit_csr(offsets, indices, len(vertices))
    if (not all(audit[k] for k in ("closed_two_incidence_edges", "consistent_winding", "valid_csr"))
            or audit["zero_length_index_edges"]):
        raise MeshingError("internal error: merged polygon boundary failed final closure audit")
    times["csr_and_final_edge_audit"] = time.perf_counter() - at
    report = {
        "method": "exact_dyadic_coplanar_single_loop_disks",
        "changed": bool(polygons), "input_vertices": int(len(vertices)),
        "input_triangles": int(len(faces)), "output_facets": int(len(lengths)),
        "merged_polygon_facets": int(len(polygons)),
        "original_triangles_replaced": int(replaced.sum()),
        "retained_triangle_facets": int(len(retained)),
        "facet_reduction_percent": float(100 * (1 - len(lengths) / len(faces))),
        "coplanar_components": int(n_components),
        "opposed_coplanar_adjacent_pairs_not_joined": int(opposite_normals),
        "boundary_indices": int(len(indices)), "largest_loop": int(max_loop),
        "min_faces": int(min_faces), "vertex_compaction": False,
        "boundary_simplification": False, "coordinate_movement": 0,
        "coplanarity_tolerance": 0,
        "integer_axis_denominator_bits": denominator_bits,
        "exact_component_plane_points_checked": int(exact_plane_points),
        "fallback_components": dict(skipped), "fallback_triangles": dict(skipped_faces),
        "audit": {**audit, "passed": True,
                  "exact_input_triangle_nondegeneracy": True,
                  "all_merged_patches_exactly_planar": True,
                  "all_merged_patches_topological_disks": True,
                  "all_merged_boundaries_simple_exact": True,
                  "all_directed_patch_boundary_edges_preserved": True,
                  "surface_preservation_basis": "exact plane, consistently oriented disk, identical directed boundary",
                  "global_self_intersections_checked": False,
                  "global_vertex_manifoldness_checked": False,
                  "physical_surface_accuracy_checked": False},
        "index_convention": "zero-based original vertex IDs; half-open CSR offsets",
        "shapely_version": shapely.__version__, "timing_seconds": times,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return offsets, indices, report
