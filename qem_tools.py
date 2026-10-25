"""Optional QEM surface reduction and sampled metrology. Python 3.10+.

Coordinates are micrometres. No smoothing is performed. Distances are measured
to triangles in both directions at every vertex, face centroid and edge midpoint.
Their maximum is a sampled observation, NOT a continuous Hausdorff certificate.
"""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import math
import struct
import time

import numpy as np


ALGORITHM_VERSION = "qem-dense-surface-v1"
MIN_QEM_TRIANGLES = 20
AGGRESSIVENESS = 3.0


def read_stl(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(84)
    count = struct.unpack("<I", header[80:84])[0] if len(header) == 84 else 0
    if count and path.stat().st_size == 84 + 50 * count:
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        data = np.memmap(str(path), mode="r", dtype=dtype, offset=84, shape=(count,))
        triangles = data["vertices"].astype(np.float64)
        del data
    else:
        points = []
        with path.open("r", encoding="ascii") as stream:
            for line in stream:
                words = line.split()
                if words and words[0].lower() == "vertex":
                    if len(words) != 4:
                        raise ValueError("Malformed ASCII STL vertex")
                    points.append([float(x) for x in words[1:]])
        if not points or len(points) % 3:
            raise ValueError("STL contains no complete triangles")
        triangles = np.asarray(points, dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(triangles).all():
        raise ValueError("STL contains nonfinite coordinates")
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3).astype(np.int32)


def _face_components(face_count, adjacent_faces):
    parents = np.arange(face_count, dtype=np.int64)
    sizes = np.ones(face_count, dtype=np.int64)

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in adjacent_faces:
        a, b = find(int(first)), find(int(second))
        if a == b:
            continue
        if sizes[a] < sizes[b]:
            a, b = b, a
        parents[b] = a
        sizes[a] += sizes[b]
    roots = np.asarray([find(index) for index in range(face_count)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def audit_mesh(vertices, faces):
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Mesh needs finite (N,3) vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Mesh needs nonempty integer (M,3) triangle indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh has invalid triangle indices")
    if len(np.unique(faces)) != len(vertices):
        raise ValueError("Mesh has unused vertices; compact its indices before reduction")
    tri = vertices[faces].astype(np.float64)
    areas2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    degenerate = int(np.count_nonzero(areas2 == 0))
    if degenerate:
        raise ValueError("Mesh contains {} degenerate triangles".format(degenerate))
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    nonclosed = int(np.count_nonzero(counts != 2))
    if nonclosed:
        raise ValueError("Mesh has {} boundary/nonmanifold edges; a closed input is required".format(nonclosed))
    winding = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    inconsistent = int(np.count_nonzero(winding))
    if inconsistent:
        raise ValueError("Mesh has {} inconsistently oriented edges".format(inconsistent))
    edge_order = np.argsort(inverse, kind="stable")
    face_ids = np.tile(np.arange(len(faces)), 3)
    labels = _face_components(len(faces), face_ids[edge_order].reshape(-1, 2))
    components = []
    for label in range(int(labels.max()) + 1):
        component_faces = faces[labels == label]
        used = np.unique(component_faces)
        component_tri = vertices[component_faces].astype(np.float64)
        centered = component_tri - vertices[used].mean(axis=0)
        volume = float(np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum() / 6)
        if not math.isfinite(volume) or volume == 0:
            raise ValueError("A closed component has zero/nonfinite enclosed signed volume")
        edge_count = len(component_faces) * 3 // 2
        components.append({"triangles": len(component_faces), "vertices": len(used),
                           "edges": edge_count, "euler_characteristic": int(len(used) - edge_count + len(component_faces)),
                           "signed_volume_um3": volume, "orientation_sign": 1 if volume > 0 else -1})
    # Reject separate closed shells touching only at an indexed vertex: an
    # ordinary edge-collapse implementation cannot treat that as a manifold.
    vertex_component = np.unique(np.column_stack((faces.reshape(-1), np.repeat(labels, 3))), axis=0)
    _, vertex_component_counts = np.unique(vertex_component[:, 0], return_counts=True)
    if np.any(vertex_component_counts > 1):
        raise ValueError("Different closed surface components share vertices")
    volume = sum(component["signed_volume_um3"] for component in components)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("Surface does not enclose a positive finite total material volume")
    return {"triangles": len(faces), "vertices": len(vertices), "unique_edges": len(edges),
            "closed_edges_all_two": True, "degenerate_triangles": 0, "inconsistent_edges": 0,
            "closed_components": len(components), "components": components,
            "euler_characteristic": int(len(vertices) - len(edges) + len(faces)),
            "mesh_volume_um3": volume,
            "local_bounds_um": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "self_intersection_test": "not performed",
            "limit": "Edge closure, component and Euler checks do not certify freedom from self-intersections or optical accuracy."}


def check_topology_preserved(before, after):
    if before["closed_components"] != after["closed_components"]:
        raise ValueError("Reduction changed the number of closed components")
    if before["euler_characteristic"] != after["euler_characteristic"]:
        raise ValueError("Reduction changed the Euler characteristic")
    signature = lambda audit: sorted((c["euler_characteristic"], c["orientation_sign"]) for c in audit["components"])
    if signature(before) != signature(after):
        raise ValueError("Reduction changed per-component Euler/orientation signatures")


def write_stl(path, vertices, faces):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing STL: " + str(path))
    tri = vertices[faces].astype(np.float32)
    normals = np.cross(tri[:, 1].astype(float) - tri[:, 0], tri[:, 2].astype(float) - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Float32 STL contains degenerate triangles")
    normals /= lengths[:, None]
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.zeros(len(faces), dtype=dtype)
    records["normal"], records["vertices"] = normals, tri
    with path.open("xb") as stream:
        stream.write(b"VDATA optional QEM; local vertex units um; see conversion manifest".ljust(80, b" "))
        stream.write(struct.pack("<I", len(faces)))
        stream.write(records.tobytes())


def dense_samples(vertices, faces):
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    return np.concatenate((vertices, vertices[faces].mean(axis=1), vertices[edges].mean(axis=1)), axis=0)


def _distances(points, vertices, faces, batch_size=1500):
    # Keep the same independently checked point-to-triangle pipeline used in
    # the research version. trimesh's proximity path also needs scipy + rtree.
    import trimesh
    import scipy.spatial  # noqa: F401 -- fail clearly before a long metrology run
    import rtree  # noqa: F401
    target = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    target.triangles_tree
    values = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), batch_size):
        _, distance, _ = trimesh.proximity.closest_point(target, points[start:start + batch_size])
        values[start:start + batch_size] = distance
    return values


def _distance_stats(values):
    nm = values * 1000
    return {"samples": len(nm), "maximum_nm": float(nm.max()),
            "rms_nm": float(np.sqrt(np.mean(nm ** 2))), "p95_nm": float(np.quantile(nm, .95)),
            "p99_nm": float(np.quantile(nm, .99))}


def require_dependencies():
    try:
        import fast_simplification  # noqa: F401
        import trimesh  # noqa: F401
        import scipy.spatial  # noqa: F401
        import rtree  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("QEM needs numpy, fast-simplification, trimesh, scipy and rtree; install requirements-qem.txt") from exc


def reduce_stl(source, destination, reduction, sampled_max_error_nm=None):
    if not math.isfinite(reduction) or not 0 < reduction < 1:
        raise ValueError("QEM reduction must be strictly between 0 and 1; exact copying is handled by the caller")
    if sampled_max_error_nm is not None and (not math.isfinite(sampled_max_error_nm) or sampled_max_error_nm < 0):
        raise ValueError("Sampled error tolerance must be finite and nonnegative")
    require_dependencies()
    import fast_simplification
    begin = time.perf_counter()
    vertices, faces = read_stl(source)
    before = audit_mesh(vertices, faces)
    if len(faces) < MIN_QEM_TRIANGLES:
        raise ValueError("QEM is disabled for fewer than {} triangles; use --reduction 0 for an exact copy".format(MIN_QEM_TRIANGLES))
    target = int((1 - reduction) * len(faces))
    if target < 4 or target >= len(faces):
        raise ValueError("Requested reduction has an invalid/unattainable target for this small closed mesh")
    simplify_start = time.perf_counter()
    out_vertices, out_faces = fast_simplification.simplify(vertices, faces, target_count=target,
                                                         agg=AGGRESSIVENESS, preserve_border=True)
    simplify_seconds = time.perf_counter() - simplify_start
    if len(out_faces) >= len(faces) or not len(out_faces):
        raise ValueError("QEM did not produce a smaller nonempty mesh; no successful reduction is claimed")
    out_faces = np.asarray(out_faces)
    if (out_faces.ndim != 2 or out_faces.shape[1] != 3 or
            not np.issubdtype(out_faces.dtype, np.integer) or
            out_faces.min() < 0 or out_faces.max() >= len(out_vertices)):
        raise ValueError("QEM returned invalid triangle indices")
    # Audit actual STL float32 values, compacting any coincident quantized
    # vertices, before writing and before distance measurements.
    out_vertices, inverse = np.unique(np.asarray(out_vertices, dtype=np.float32), axis=0, return_inverse=True)
    out_vertices = out_vertices.astype(np.float64)
    out_faces = inverse[np.asarray(out_faces)].astype(np.int32)
    after = audit_mesh(out_vertices, out_faces)
    check_topology_preserved(before, after)
    write_stl(destination, out_vertices, out_faces)
    forward = _distances(dense_samples(vertices, faces), out_vertices, out_faces)
    reverse = _distances(dense_samples(out_vertices, out_faces), vertices, faces)
    if not np.isfinite(forward).all() or not np.isfinite(reverse).all():
        raise ValueError("Point-to-triangle metrology returned nonfinite distances")
    maximum = float(max(forward.max(), reverse.max()) * 1000)
    accepted = sampled_max_error_nm is None or maximum <= sampled_max_error_nm
    versions = {}
    for package in ("numpy", "fast-simplification", "trimesh", "scipy", "rtree"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {"algorithm_version": ALGORITHM_VERSION, "dependency_versions": versions,
            "requested_reduction": reduction, "actual_triangle_reduction": 1 - len(out_faces) / len(faces),
            "aggressiveness": AGGRESSIVENESS, "preserve_border": True, "smoothing": "none",
            "source_audit": before, "output_audit": after,
            "topology_checks_preserved": True,
            "additional_volume_change_percent": 100 * (after["mesh_volume_um3"] / before["mesh_volume_um3"] - 1),
            "source_to_reduced": _distance_stats(forward), "reduced_to_source": _distance_stats(reverse),
            "sampling": "all vertices + all face centroids + all distinct edge midpoints, both directions",
            "distance_target": "triangles, not a point cloud",
            "sampled_symmetric_maximum_nm": maximum,
            "sampled_max_error_tolerance_nm": sampled_max_error_nm,
            "sampled_tolerance_passed": accepted, "accepted": accepted,
            "strict_hausdorff_bound": False, "stl_float32_quantization_included": True,
            "simplify_seconds": simplify_seconds, "elapsed_seconds": time.perf_counter() - begin,
            "stl_bytes": Path(destination).stat().st_size,
            "limitation": "Sampled errors are additional to the input STL approximation. They do not certify continuous Hausdorff distance, CD, LER, self-intersection freedom, FDTD accuracy or solver-memory reduction."}
