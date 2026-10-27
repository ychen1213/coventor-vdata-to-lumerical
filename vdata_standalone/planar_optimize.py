"""Conservative, boundary-locked optimization of exactly axis-aligned planes.

Only one consistently oriented, connected, convex disk at a time is replaced.
All its boundary vertices and edges are retained.  Curved faces, tilted planes,
holes and concave outlines are deliberately left alone.  This is geometric
retriangulation, not bit-identical mesh storage and not a general QEM reducer.
"""

from __future__ import annotations

from collections import Counter
import time
from typing import Any

import numpy as np

from .meshing import MeshingError, _mesh_arrays, audit_mesh


def _closed_good(report: dict[str, Any]) -> bool:
    return (not report["empty"] and report["watertight_edges"]
            and report["consistent_winding"] and report["degenerate_faces"] == 0
            and np.isfinite(report["surface_area_m2"])
            and np.isfinite(report["signed_volume_m3"]))


def _component_count(faces: np.ndarray, n_vertices: int) -> int:
    """Count used-vertex components with a compact sparse graph, not dicts."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(faces):
        return 0
    dtype = np.int32 if n_vertices <= np.iinfo(np.int32).max else np.int64
    rows = np.concatenate((faces[:, 0], faces[:, 0])).astype(dtype, copy=False)
    cols = np.concatenate((faces[:, 1], faces[:, 2])).astype(dtype, copy=False)
    graph = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                       shape=(n_vertices, n_vertices)).tocsr()
    del rows, cols
    count, _labels = connected_components(graph, directed=False)
    # Unused input vertices are isolated graph nodes, not mesh components.
    used = np.zeros(n_vertices, dtype=np.bool_)
    for begin in range(0, len(faces), 200_000):
        used[faces[begin:begin + 200_000]] = True
    return int(count - (n_vertices - np.count_nonzero(used)))


def _patch_components(faces: np.ndarray, n_vertices: int):
    """Face components and directed boundary edges using sorted uint64 keys."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    keys = np.empty(3 * n_faces, dtype=np.uint64)
    for edge, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        low = np.minimum(faces[:, a], faces[:, b]).astype(np.uint64)
        low *= np.uint64(n_vertices)
        low += np.maximum(faces[:, a], faces[:, b]).astype(np.uint64)
        keys[edge::3] = low
    order = np.argsort(keys, kind="quicksort")
    keys = keys[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(keys)])
    del keys
    if np.any(counts > 2):
        return None
    pair_starts = starts[counts == 2]
    rows = (order[pair_starts] // 3).astype(np.int32)
    cols = (order[pair_starts + 1] // 3).astype(np.int32)
    graph = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                       shape=(n_faces, n_faces)).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    del graph, rows, cols, pair_starts
    positions = order[starts[counts == 1]]
    boundary_face = positions // 3
    edge = positions % 3
    boundary_a = faces[boundary_face, edge]
    boundary_b = faces[boundary_face, (edge + 1) % 3]
    return int(n_components), labels, boundary_face, boundary_a, boundary_b


def _one_boundary_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Preserve the original directed edge orientation, including collinear points."""
    if len(a) < 3:
        return None
    order = np.argsort(a)
    aa, bb = a[order], b[order]
    if np.any(aa[1:] == aa[:-1]) or not np.array_equal(aa, np.sort(bb)):
        return None
    next_position = np.searchsorted(aa, bb)
    path = np.empty(len(a), dtype=np.int64)
    seen = np.zeros(len(a), dtype=np.bool_)
    position = 0
    for i in range(len(a)):
        if seen[position]:
            return None  # Several loops, or a premature return.
        seen[position] = True
        path[i] = aa[position]
        position = int(next_position[position])
    return path if position == 0 else None


def _integer_points(points: np.ndarray) -> list[tuple[int, int]]:
    """Express finite float64 dyadics on one exact integer lattice per axis."""
    axes = []
    for axis in range(2):
        ratios = [float(x).as_integer_ratio() for x in points[:, axis]]
        denominator = max(d for _n, d in ratios)
        axes.append([n * (denominator // d) for n, d in ratios])
    return list(zip(*axes))


def _orient(a, b, c) -> int:
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def _convex_corners(points: np.ndarray) -> list[int] | None:
    """Exact convexity and ordering check; no tolerance can erase a small notch."""
    xy = _integer_points(points)
    if len(set(xy)) != len(xy):
        return None
    corners, sign = [], 0
    for i, point in enumerate(xy):
        previous, following = xy[i - 1], xy[(i + 1) % len(xy)]
        turn = _orient(previous, point, following)
        if not turn:
            # A straight boundary may have samples, but may not backtrack.
            if ((point[0] - previous[0]) * (following[0] - point[0])
                    + (point[1] - previous[1]) * (following[1] - point[1])) <= 0:
                return None
            continue
        this_sign = 1 if turn > 0 else -1
        if sign and sign != this_sign:
            return None
        sign = this_sign
        corners.append(i)
    if len(corners) < 3:
        return None
    # Consistent local turns alone would accept some self-crossing stars.
    # Compare the reduced boundary with an exact monotone-chain convex hull.
    sorted_indices = sorted(range(len(xy)), key=xy.__getitem__)
    lower, upper = [], []
    for indices, chain in ((sorted_indices, lower), (reversed(sorted_indices), upper)):
        for idx in indices:
            while len(chain) >= 2 and _orient(xy[chain[-2]], xy[chain[-1]], xy[idx]) <= 0:
                chain.pop()
            chain.append(idx)
    hull = lower[:-1] + upper[:-1]
    if len(hull) != len(corners):
        return None
    expected = hull if sign > 0 else hull[::-1]
    try:
        offset = expected.index(corners[0])
    except ValueError:
        return None
    if corners != expected[offset:] + expected[:offset]:
        return None
    return corners


def _axis_groups(vertices: np.ndarray, faces: np.ndarray, axis: int,
                 chunk_size: int = 200_000):
    """Find exact planes and orientation; process one axis to bound memory."""
    ids, values, signs = [], [], []
    u, v = (axis + 1) % 3, (axis + 2) % 3
    for start in range(0, len(faces), chunk_size):
        part = faces[start:start + chunk_size]
        coords = vertices[part, axis]
        selected = (coords[:, 0] == coords[:, 1]) & (coords[:, 0] == coords[:, 2])
        indexes = np.flatnonzero(selected)
        if not len(indexes):
            continue
        triangle = vertices[part[indexes]]
        cross = ((triangle[:, 1, u] - triangle[:, 0, u])
                 * (triangle[:, 2, v] - triangle[:, 0, v])
                 - (triangle[:, 1, v] - triangle[:, 0, v])
                 * (triangle[:, 2, u] - triangle[:, 0, u]))
        valid = cross != 0
        ids.append(indexes[valid] + start)
        values.append(coords[indexes[valid], 0])
        signs.append(np.sign(cross[valid]).astype(np.int8))
    if not ids:
        return
    face_ids, plane, direction = np.concatenate(ids), np.concatenate(values), np.concatenate(signs)
    del ids, values, signs
    order = np.lexsort((direction, plane))
    face_ids, plane, direction = face_ids[order], plane[order], direction[order]
    del order
    starts = np.r_[0, np.flatnonzero((plane[1:] != plane[:-1])
                                    | (direction[1:] != direction[:-1])) + 1]
    stops = np.r_[starts[1:], len(face_ids)]
    for start, stop in zip(starts, stops):
        yield face_ids[start:stop], float(plane[start]), int(direction[start])


def optimize_planar_mesh(vertices: Any, faces: Any, *, min_faces: int = 128):
    """Replace safe convex axis-plane disks while locking every boundary edge.

    Returns ``(vertices, faces, report)``.  Input coordinates are never moved.
    A new center uses the mean of strict hull corners and copies its plane
    coordinate exactly.  All unused vertices are compacted after replacement.
    Every original non-replaced face survives with the same coordinates and
    winding.  The routine intentionally accepts only valid closed input meshes;
    invalid input raises :class:`MeshingError`.  A failed final invariant check
    rolls all replacements back, with the reason in the report.

    Plane tests and outline convexity are exact for the supplied float64
    coordinates.  The audit does not prove arbitrary global self-intersection
    freedom.  This routine is for a single material mesh; independently applying
    it to several contacting materials does not ensure conforming interfaces.
    """
    started = time.perf_counter()
    if isinstance(min_faces, bool) or not isinstance(min_faces, (int, np.integer)) or min_faces < 1:
        raise MeshingError("min_faces must be a positive integer")
    vertices, faces = _mesh_arrays(vertices, faces)
    before = audit_mesh(vertices, faces)
    if before["empty"]:
        return vertices.copy(), faces.copy(), {"method": "exact_axis_planar_disks", "changed": False,
                    "accepted_patches": 0, "before": before, "after": before,
                    "input_components": 0, "output_components": 0, "removed_faces": 0,
                    "arrays_changed": False, "geometric_surface_preserved": True,
                    "elapsed_seconds": time.perf_counter() - started}
    if not _closed_good(before):
        raise MeshingError("planar optimization requires a closed, consistently wound, nondegenerate mesh")
    before_components = _component_count(faces, len(vertices))
    if len(vertices) * len(vertices) > np.iinfo(np.uint64).max:
        raise MeshingError("mesh has too many vertices for uint64 edge keys")
    removed = np.zeros(len(faces), dtype=np.bool_)
    centers, replacement_faces, patches = [], [], []
    skipped = Counter()
    candidate_groups = 0
    for axis in range(3):
        for group_ids, plane, direction in _axis_groups(vertices, faces, axis):
            if len(group_ids) < min_faces:
                skipped["below_min_faces"] += 1
                continue
            candidate_groups += 1
            group = faces[group_ids]
            components = _patch_components(group, len(vertices))
            if components is None:
                skipped["nonmanifold_plane_group"] += 1
                continue
            n_components, labels, boundary_face, boundary_a, boundary_b = components
            sizes = np.bincount(labels, minlength=n_components)
            # Boundary-size masks are small; face sorting happens just once for
            # a plane containing disconnected patches, never once per patch.
            if n_components > 1:
                face_order = np.argsort(labels, kind="stable")
                offsets = np.r_[0, np.cumsum(sizes)]
            boundary_labels = labels[boundary_face]
            for component in np.flatnonzero(sizes >= min_faces):
                part_ids = (np.arange(len(group)) if n_components == 1 else
                            face_order[offsets[component]:offsets[component + 1]])
                take = boundary_labels == component
                loop = _one_boundary_loop(boundary_a[take], boundary_b[take])
                if loop is None:
                    skipped["not_one_boundary_loop"] += 1
                    continue
                # A connected orientable triangulated disk has V-E+F == 1.
                # The single boundary + convex checks below also rule out holes.
                f_count = len(part_ids)
                boundary_count = len(loop)
                vertex_count = np.unique(group[part_ids]).size
                edge_count = (3 * f_count + boundary_count) // 2
                if vertex_count - edge_count + f_count != 1:
                    skipped["not_a_disk"] += 1
                    continue
                u, v = (axis + 1) % 3, (axis + 2) % 3
                corners = _convex_corners(vertices[loop][:, [u, v]])
                if corners is None:
                    skipped["not_exactly_convex"] += 1
                    continue
                if boundary_count >= f_count:
                    skipped["no_face_reduction"] += 1
                    continue
                center = np.mean(vertices[loop[np.asarray(corners)]], axis=0, dtype=np.float64)
                center[axis] = plane  # Never average a constant plane coordinate.
                # Recheck the rounded mean is strictly inside every edge using
                # the same exact float64-dyadic predicate.
                xy = _integer_points(np.vstack((vertices[loop][:, [u, v]], center[[u, v]])))
                if any(_orient(xy[i], xy[(i + 1) % boundary_count], xy[-1]) * direction <= 0
                       for i in range(boundary_count)):
                    skipped["center_not_strictly_inside"] += 1
                    continue
                new_center = len(vertices) + len(centers)
                new_faces = np.column_stack((loop, np.roll(loop, -1),
                                             np.full(boundary_count, new_center, dtype=np.int64)))
                # Degenerate fan triangles can occur if an edge is extremely
                # short; global audit would reject them, so keep this disk.
                triangle = np.stack((vertices[loop], vertices[np.roll(loop, -1)],
                                     np.broadcast_to(center, (boundary_count, 3))), axis=1)
                double_area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                                     triangle[:, 2] - triangle[:, 0]), axis=1)
                if np.any(double_area <= before["degenerate_area2_tolerance_m2"]):
                    skipped["fan_degenerate_at_global_scale"] += 1
                    continue
                removed[group_ids[part_ids]] = True
                centers.append(center)
                replacement_faces.append(new_faces)
                patches.append({"axis": "XYZ"[axis], "coordinate_m": plane, "normal_sign": direction,
                                "old_faces": f_count, "new_faces": boundary_count,
                                "locked_boundary_vertices": boundary_count})
    report = {"method": "exact_axis_planar_disks", "changed": bool(patches),
              "arrays_changed": bool(patches), "geometric_surface_preserved": True,
              "min_faces": int(min_faces), "candidate_plane_groups": candidate_groups,
              "accepted_patches": len(patches), "patches": patches,
              "skipped": dict(skipped), "before": before,
              "input_components": before_components, "boundary_coordinates_preserved_exactly": True,
              "self_intersection_checked": False,
              "limitations": ["Strict axis-aligned planes only; concave outlines and holes are retained.",
                              "Center fans can contain long, thin triangles; this is not a triangle-quality optimizer.",
                              "Single-material operation does not guarantee conforming multi-material interfaces."]}
    if not patches:
        report.update(after=before, output_components=before_components,
                      removed_faces=0, elapsed_seconds=time.perf_counter() - started)
        return vertices.copy(), faces.copy(), report
    all_vertices = np.vstack((vertices, np.asarray(centers)))
    # ``uint64`` + ``int64`` otherwise promotes to float64 in NumPy.  Indices
    # have already been checked to be < len(vertices) by _mesh_arrays; array
    # lengths use signed np.intp, so they fit safely in this int64 work type.
    # Cast only the selected copy and never mutate the caller's face array.
    all_faces = np.concatenate([faces[~removed].astype(np.int64, copy=False),
                                *replacement_faces])
    used = np.unique(all_faces)
    index = np.full(len(all_vertices), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    optimized_vertices = all_vertices[used]
    optimized_faces = index[all_faces]
    if len(optimized_vertices) <= np.iinfo(np.int32).max:
        optimized_faces = optimized_faces.astype(np.int32)
    del all_vertices, all_faces, index, used, replacement_faces
    after = audit_mesh(optimized_vertices, optimized_faces)
    after_components = _component_count(optimized_faces, len(optimized_vertices))
    failure = None
    if not _closed_good(after):
        failure = "final closure, winding, or degeneracy invariant failed"
    elif before["euler_characteristic"] != after["euler_characteristic"]:
        failure = "Euler characteristic changed"
    elif before_components != after_components:
        failure = "connected component count changed"
    elif not np.array_equal(before["bounds_local_m"], after["bounds_local_m"]):
        failure = "bounds changed"
    else:
        # Chunked accumulation order changes after retriangulation.  This is a
        # strict numeric check, not a permission to move surface coordinates.
        rel = 8192 * np.finfo(np.float64).eps
        scale = float(np.max(np.ptp(vertices, axis=0)))
        for name, absolute in (("surface_area_m2", rel * scale ** 2),
                               ("signed_volume_m3", rel * scale ** 3)):
            if not np.isclose(before[name], after[name], rtol=rel, atol=absolute):
                failure = f"{name} changed beyond floating-point tolerance"
                break
    if failure:
        report.update(changed=False, rolled_back=True, rollback_reason=failure,
                      arrays_changed=False, accepted_patches=0, patches=[],
                      candidate_patches=patches,
                      candidate_after=after, after=before, output_components=before_components,
                      removed_faces=0, elapsed_seconds=time.perf_counter() - started)
        return vertices.copy(), faces.copy(), report
    report.update(after=after, output_components=after_components,
                  removed_faces=int(len(faces) - len(optimized_faces)),
                  elapsed_seconds=time.perf_counter() - started)
    return optimized_vertices, optimized_faces, report
