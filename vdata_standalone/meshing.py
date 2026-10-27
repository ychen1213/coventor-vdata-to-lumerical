"""Open-source surface extraction for decoded Coventor voxel fractions.

This module uses NumPy and scikit-image only.  No installed process simulator,
native command, licence server, or optical CAD API is used.  Marching cubes is
an explicit reconstruction choice: an occupancy fraction is *not* a uniquely
specified plane or surface.  In particular, padding and interpolation round
the corners of an isolated voxel; this is not the exact union of voxel cubes.

Coordinates returned by :func:`mesh_material` are local SI metres.  Add the
reported ``translation_m`` once to place the surface in world coordinates.
Keeping the original origin out of the float32 marching-cubes coordinates
avoids losing small features when a design has a large world-coordinate offset.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any

import numpy as np


class MeshingError(ValueError):
    """Invalid geometry, or a requested extraction could not be validated."""


def _audit_failure_counts(report: dict[str, Any]) -> str:
    return ", ".join(f"{key}={report.get(key)}" for key in (
        "degenerate_faces", "boundary_edges", "nonmanifold_edges",
        "inconsistent_winding_edges", "euler_characteristic"))


def _xyz(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise MeshingError(f"{name} must contain three finite XYZ values")
    if positive and np.any(arr <= 0):
        raise MeshingError(f"{name} must be strictly positive")
    return arr


def _crop_bounds(crop_indices: Any, dims: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    if crop_indices is None:
        return np.zeros(3, dtype=np.int64), np.asarray(dims, dtype=np.int64)
    bounds = np.asarray(crop_indices)
    if not np.issubdtype(bounds.dtype, np.integer):
        raise MeshingError("crop_indices must contain integers")
    if bounds.shape == (3, 2):
        lo, hi = bounds[:, 0], bounds[:, 1]
    elif bounds.shape == (2, 3):
        lo, hi = bounds[0], bounds[1]
    else:
        raise MeshingError("crop_indices must be three (start, stop) XYZ pairs, or (lo_xyz, hi_xyz)")
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    if np.any(lo < 0) or np.any(hi > np.asarray(dims)) or np.any(lo >= hi):
        raise MeshingError("crop_indices must be nonempty half-open ranges within the voxel array")
    return lo, hi


def _mesh_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise MeshingError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise MeshingError("faces must be an integer array with shape (M, 3)")
    if not np.all(np.isfinite(vertices)):
        raise MeshingError("vertices contain non-finite coordinates")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise MeshingError("face vertex index is out of range")
    return vertices, faces


def _edge_audit(faces: np.ndarray, n_vertices: int, *, chunk_size: int) -> dict[str, int]:
    """Count undirected edge incidences with one compact sortable uint64 array.

    The low bit retains edge direction.  Sorted runs can then detect holes,
    nonmanifold edges and inconsistent winding.  Chunked run reduction avoids
    another whole-mesh edge array and a whole-mesh argsort index array.
    """
    if n_vertices * n_vertices * 2 > np.iinfo(np.uint64).max:
        raise MeshingError("mesh has too many vertices for compact uint64 edge auditing")
    edge_count = len(faces) * 3
    codes = np.empty(edge_count, dtype=np.uint64)
    for begin in range(0, len(faces), chunk_size):
        part = faces[begin:begin + chunk_size]
        for edge, (a_idx, b_idx) in enumerate(((0, 1), (1, 2), (2, 0))):
            a, b = part[:, a_idx], part[:, b_idx]
            encoded = np.minimum(a, b).astype(np.uint64)
            encoded *= np.uint64(n_vertices)
            encoded += np.maximum(a, b).astype(np.uint64)
            encoded *= np.uint64(2)
            encoded += (a > b).astype(np.uint64)
            codes[3 * begin + edge:3 * (begin + len(part)):3] = encoded
    codes.sort(kind="quicksort")
    totals = {"unique_edges": 0, "boundary_edges": 0, "nonmanifold_edges": 0,
              "inconsistent_winding_edges": 0}

    def classify(count: int, parity_sum: int) -> None:
        totals["unique_edges"] += 1
        if count == 1:
            totals["boundary_edges"] += 1
        elif count > 2:
            totals["nonmanifold_edges"] += 1
        elif parity_sum != 1:
            totals["inconsistent_winding_edges"] += 1

    carry_key: int | None = None
    carry_count = carry_parity = 0
    edge_chunk = max(2, chunk_size * 3)
    for begin in range(0, edge_count, edge_chunk):
        part = codes[begin:begin + edge_chunk]
        keys = part >> np.uint64(1)
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        stops = np.r_[starts[1:], len(part)]
        counts = stops - starts
        first_count = int(counts[0])
        first_parity = int(np.sum(part[:first_count] & np.uint64(1), dtype=np.uint64))
        if carry_key is not None:
            if carry_key == int(keys[0]):
                first_count += carry_count
                first_parity += carry_parity
            else:
                classify(carry_count, carry_parity)
        if len(starts) == 1:
            carry_key, carry_count, carry_parity = int(keys[0]), first_count, first_parity
            continue
        classify(first_count, first_parity)
        # Only two-incidence edges need an orientation test.  Runs strictly
        # inside this chunk can be classified with small temporary arrays.
        middle_starts, middle_counts = starts[1:-1], counts[1:-1]
        totals["unique_edges"] += len(middle_counts)
        totals["boundary_edges"] += int(np.count_nonzero(middle_counts == 1))
        totals["nonmanifold_edges"] += int(np.count_nonzero(middle_counts > 2))
        paired = middle_starts[middle_counts == 2]
        totals["inconsistent_winding_edges"] += int(np.count_nonzero(
            (part[paired] & np.uint64(1)) == (part[paired + 1] & np.uint64(1))))
        last_start = int(starts[-1])
        carry_key = int(keys[last_start])
        carry_count = int(counts[-1])
        carry_parity = int(np.sum(part[last_start:] & np.uint64(1), dtype=np.uint64))
    if carry_key is not None:
        classify(carry_count, carry_parity)
    return totals


def audit_mesh(vertices: Any, faces: Any, *, chunk_size: int = 200_000,
               check_topology: bool = True) -> dict[str, Any]:
    """Measure a mesh and check edge closure, winding and degeneracy.

    This audit does not prove absence of self-intersection or equivalence to
    the physical etched surface.  It checks finite coordinates, valid indices,
    nonzero triangles, two triangles per edge, and consistent edge winding.
    A shared vertex with disconnected incident fans is not ruled out either.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise MeshingError("chunk_size must be a positive integer")
    vertices, faces = _mesh_arrays(vertices, faces)
    report: dict[str, Any] = {
        "vertices": len(vertices), "faces": len(faces), "empty": len(faces) == 0,
        "finite_coordinates": True, "valid_indices": True,
        "topology_checked": bool(check_topology),
        "self_intersection_checked": False,
        "vertex_manifoldness_checked": False,
    }
    if not len(vertices) or not len(faces):
        report.update({"surface_area_m2": 0.0, "signed_volume_m3": 0.0,
                       "volume_m3": 0.0, "degenerate_faces": 0,
                       "bounds_local_m": None, "used_vertices": 0,
                       "watertight_edges": False, "consistent_winding": False})
        if check_topology:
            report.update({"unique_edges": 0, "boundary_edges": 0, "nonmanifold_edges": 0,
                           "inconsistent_winding_edges": 0, "euler_characteristic": 0})
        return report
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    reference = (lower + upper) * 0.5
    scale = float(np.max(upper - lower))
    area2_tolerance = np.finfo(np.float64).eps * scale * scale * 32
    area = signed_volume = 0.0
    degenerate = 0
    used = np.zeros(len(vertices), dtype=np.bool_)
    for begin in range(0, len(faces), chunk_size):
        part = faces[begin:begin + chunk_size]
        used[part] = True
        triangles = vertices[part] - reference
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        degenerate += int(np.count_nonzero(area2 <= area2_tolerance))
        area += 0.5 * float(np.sum(area2, dtype=np.float64))
        signed_volume += float(np.einsum("ij,ij->", triangles[:, 0], cross, dtype=np.float64)) / 6.0
    report.update({"surface_area_m2": area, "signed_volume_m3": signed_volume,
                   "volume_m3": abs(signed_volume), "degenerate_faces": degenerate,
                   "bounds_local_m": [lower.tolist(), upper.tolist()],
                   "used_vertices": int(np.count_nonzero(used)),
                   "degenerate_area2_tolerance_m2": area2_tolerance})
    if check_topology:
        report.update(_edge_audit(faces, len(vertices), chunk_size=chunk_size))
        report["euler_characteristic"] = report["used_vertices"] - report["unique_edges"] + len(faces)
        report["watertight_edges"] = report["boundary_edges"] == report["nonmanifold_edges"] == 0
        report["consistent_winding"] = report["inconsistent_winding_edges"] == 0
    return report


def _active_block_mask(field: np.ndarray, level: float, *, step_size: int = 1,
                       block_size: int | tuple[int, int, int] = (32, 32, 8)) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask definitely inactive MC regions without changing a sample value.

    Blocks partition *cell lower corners*, not independent volumes.  A block
    reduction includes the shared upper samples, up to ``step_size`` beyond
    its lower-corner range.  If its min/max straddle or touch the isovalue, the
    full sample box is enabled.  This also retains one cell halo on its lower
    sides.  Equal-to-level samples are deliberately not classified by sign.

    scikit-image 0.26's Cython loop tests ``mask[z+st, y+st, x+st]`` before
    loading a cube.  The input array's XYZ axes are untouched by this internal
    variable naming.  See src/skimage/measure/_marching_cubes_lewiner_cy.pyx
    in the upstream v0.26.0 tag.  Both the lower and upper corners of every
    potentially crossing cell are inside an enabled sample box, so the mask
    remains conservative under corner-based masking as well.  Extraction is
    still one global Lewiner call; there are no new per-block crop caps.
    """
    sizes = np.asarray(block_size)
    if sizes.ndim == 0:
        sizes = np.repeat(sizes, 3)
    if (sizes.shape != (3,) or not np.issubdtype(sizes.dtype, np.integer)
            or np.any(sizes < 1)):
        raise MeshingError("active block_size must contain positive XYZ integers")
    if (isinstance(step_size, bool) or not isinstance(step_size, (int, np.integer))
            or step_size < 1):
        raise MeshingError("active step_size must be a positive integer")
    if field.ndim != 3 or min(field.shape) <= step_size:
        raise MeshingError("active-block field must have three axes larger than step_size")
    started = time.perf_counter()
    mask = np.zeros(field.shape, dtype=np.bool_)
    active = total = 0
    cell_shape = tuple(int(n) - int(step_size) for n in field.shape)
    for x in range(0, cell_shape[0], int(sizes[0])):
        x_end = min(x + int(sizes[0]), cell_shape[0]) + int(step_size)
        for y in range(0, cell_shape[1], int(sizes[1])):
            y_end = min(y + int(sizes[1]), cell_shape[1]) + int(step_size)
            for z in range(0, cell_shape[2], int(sizes[2])):
                z_end = min(z + int(sizes[2]), cell_shape[2]) + int(step_size)
                part = field[x:x_end, y:y_end, z:z_end]
                minimum, maximum = float(part.min()), float(part.max())
                total += 1
                # Uniform level-valued blocks are MC case 0 (no surface).
                # Nonuniform ties stay enabled to preserve Lewiner decisions.
                if minimum <= level <= maximum and minimum != maximum:
                    mask[x:x_end, y:y_end, z:z_end] = True
                    active += 1
    enabled_samples = int(np.count_nonzero(mask))
    return mask, {
        "enabled": True, "block_cells_xyz": sizes.tolist(),
        "total_blocks": total, "active_blocks": active,
        "active_block_fraction": active / total if total else 0.0,
        "mask_enabled_samples": enabled_samples,
        "mask_total_samples": int(mask.size),
        "mask_coverage_fraction": enabled_samples / mask.size,
        "mask_bytes": int(mask.nbytes),
        "preprocessing_seconds": time.perf_counter() - started,
        "shared_upper_samples": int(step_size),
        "lower_cell_halo": 1,
        "geometry_samples_modified": False,
        "global_extraction": True,
    }


def mesh_material(raw: np.ndarray, meta: dict[str, Any], material_id: int, *,
                  level: float = 0.5, crop_indices: Any = None,
                  step_size: int = 1, surface: str = "fraction", planes=None,
                  active_blocks: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct one material's fraction isosurface in local SI metres.

    ``raw`` is the reader's uint8 XYZ array with shape ``(Nx, Ny, Nz, 3)``.
    Fractions are sampled at voxel centres, not at voxel lower corners.
    ``crop_indices`` uses half-open integer XYZ index intervals.  Cropping
    explicitly replaces the material outside the selected box with void and
    therefore creates artificial closed cut faces.  Materials are extracted
    independently; shared interfaces and triple junctions are not guaranteed
    conforming. The default is the historical fraction isosurface. Optional
    vof and plane modes build a distance-like field before zero-level MC;
    neither mode is an exact union of piecewise interface polygons. Optional
    ``active_blocks`` builds a conservative mask on the already padded field;
    it trades an additional one-byte-per-sample array and preprocessing for
    skipping homogeneous regions within the same global extraction. It does
    not reduce output geometry, field construction, or the later mesh audit.
    """
    from skimage.measure import marching_cubes
    from .reader import fraction_for_material

    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[-1] != 3:
        raise MeshingError("raw must be a uint8 array with shape (Nx, Ny, Nz, 3)")
    dims = tuple(int(v) for v in raw.shape[:3])
    if any(v <= 0 for v in dims):
        raise MeshingError("raw has an empty spatial axis")
    declared = tuple(int(v) for v in meta["dimensions_xyz"])
    if declared != dims:
        raise MeshingError("metadata dimensions_xyz do not match raw XYZ dimensions")
    if isinstance(material_id, bool) or not isinstance(material_id, (int, np.integer)) or not 0 <= material_id <= 255:
        raise MeshingError("material_id must be an integer between 0 and 255")
    if not math.isfinite(level) or not 0 < level < 1:
        raise MeshingError("level must be finite and strictly between zero and one")
    if surface not in {"fraction", "vof", "plane"}:
        raise MeshingError("surface must be fraction, vof or plane")
    if not isinstance(active_blocks, (bool, np.bool_)):
        raise MeshingError("active_blocks must be a boolean")
    if surface != "fraction" and level != 0.5:
        raise MeshingError("Distance-based modes use zero distance; --level only applies to fraction mode")
    if surface != "fraction" and step_size != 1:
        raise MeshingError("Distance-based modes require step_size=1")
    if isinstance(step_size, bool) or not isinstance(step_size, (int, np.integer)) or step_size < 1:
        raise MeshingError("step_size must be a positive integer")
    spacing = _xyz(meta["voxel_size_m"], "voxel_size_m", positive=True)
    origin = _xyz(meta["origin_m"], "origin_m")
    lo, hi = _crop_bounds(crop_indices, dims)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    fraction = np.asarray(fraction_for_material(np.ascontiguousarray(raw[slices]), int(material_id)), dtype=np.float32)
    if fraction.shape != tuple((hi - lo).tolist()):
        raise MeshingError("fraction reader returned an unexpected XYZ shape")
    if not np.all(np.isfinite(fraction)) or float(fraction.min()) < 0 or float(fraction.max()) > 1:
        raise MeshingError("fraction reader returned values outside finite [0, 1]")
    source_volume = float(np.sum(fraction, dtype=np.float64)) * float(np.prod(spacing))
    translation = origin + lo * spacing
    warnings = [
        "A fraction is not a unique surface: this output uses trilinear 0.5-style isosurface reconstruction.",
        "Zero padding closes boundaries and rounds/chamfers voxel-scale corners; this is not the exact union of voxel cubes.",
        "Independent per-material surfaces do not guarantee a conforming shared interface or watertight multi-material partition at triple junctions.",
        "Edge closure checks do not prove absence of self-intersections or physical accuracy.",
    ]
    if np.any(lo != 0) or np.any(hi != np.asarray(dims)):
        warnings.append("Cropping replaces the exterior with void and creates artificial closed cut surfaces.")
    if step_size != 1:
        warnings.append("step_size > 1 subsamples the field and can discard features; it is not an accuracy-preserving compression.")
    if meta.get("has_plane_equations") and surface != "plane":
        warnings.append("The source reports plane equations; this fraction-only reconstruction does not use them.")
    report: dict[str, Any] = {
        "algorithm": "skimage.measure.marching_cubes/lewiner/fraction_isosurface",
        "material_id": int(material_id), "level": float(level), "step_size": int(step_size),
        "axis_order": "XYZ", "sample_location": "voxel_center",
        "voxel_size_m": spacing.tolist(), "source_origin_m": origin.tolist(),
        "crop_indices_xyz": [[int(a), int(b)] for a, b in zip(lo, hi)],
        "crop_dimensions_xyz": (hi - lo).tolist(),
        "translation_m": translation.tolist(), "vertex_coordinates": "local_metres",
        "source_fraction_volume_m3": source_volume,
        "plane_equations_used": False, "warnings": warnings,
        "active_blocks": {"enabled": bool(active_blocks)},
    }
    extraction_level = float(level)
    exterior_value = 0.0
    gradient_direction = "descent"
    if surface == "vof":
        from .reconstruction import reconstruct_fraction_distance
        fraction, field_report = reconstruct_fraction_distance(fraction, spacing)
        report["algorithm"] = "PLIC-guided center distance / Lewiner marching cubes"
        report["field_reconstruction"] = field_report
        report["warnings"][0] = "A local PLIC plane is estimated from fractions; center-field MC is not an exact volume-conservative PLIC partition."
        extraction_level, exterior_value = 0.0, -0.5
    elif surface == "plane":
        from .planes import plane_center_field
        if planes is None or tuple(planes.shape) != dims + (4,):
            raise MeshingError("plane mode requires decoded source XYZx4 PlaneEquations")
        if not np.all(spacing == spacing[0]):
            raise MeshingError("The verified plane convention requires cubic voxels")
        negative_field, field_report = plane_center_field(np.ascontiguousarray(raw[slices]), planes[slices], int(material_id), float(spacing[0]))
        fraction = np.asarray(negative_field / spacing[0], dtype=np.float32)
        report["algorithm"] = "saved PlaneEquations center distance / Lewiner marching cubes"
        report["field_reconstruction"] = field_report
        report["plane_equations_used"] = True
        report["warnings"][0] = "Uses the explicitly selected local voxel-corner plane convention; center-field interpolation does not exactly clip the saved interface planes."
        extraction_level, exterior_value = 0.0, 0.5
        gradient_direction = "ascent"
        report["inside_sign"] = "negative"
    report["surface_mode"] = surface
    report["extraction_level"] = extraction_level
    if surface == "vof":
        warnings[1] = "Opposite-sign mirrored exterior ghost samples close the crop; interpolated corners are still artificial."
    elif surface == "plane":
        warnings[1] = "Exterior ghost samples at +0.5 voxel distance close the crop; closure walls need not coincide exactly with the crop planes."
    empty_field = float(fraction.min()) >= 0 if surface == "plane" else float(fraction.max()) <= extraction_level
    if empty_field:
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int32)
        report.update(audit_mesh(vertices, faces))
        report.update({"orientation_flipped": False, "empty_reason": "No fraction sample exceeds the selected isovalue.",
                       "volume_difference_m3": -source_volume,
                       "volume_relative_difference": -1.0 if source_volume else None})
        if active_blocks:
            report["active_blocks"]["skipped_reason"] = "empty material field"
        return vertices, faces, report
    # A larger upper padding for strided extraction ensures the last sampled
    # plane is zero.  With step_size=1 this is exactly one sample per side.
    high_pad = [1 + (-(int(n) + 1)) % int(step_size) for n in fraction.shape]
    if surface == "vof":
        from .reconstruction import pad_distance_field
        padded = pad_distance_field(fraction)
        report["crop_padding"] = "opposite-sign mirrored magnitude at exterior ghost samples"
    else:
        padded = np.pad(fraction, [(1, p) for p in high_pad], mode="constant", constant_values=exterior_value)
    mc_spacing = tuple(spacing) if surface == "fraction" else (1., 1., 1.)
    del fraction
    mask = None
    if active_blocks:
        mask, report["active_blocks"] = _active_block_mask(
            padded, extraction_level, step_size=int(step_size))
    extraction_started = time.perf_counter()
    try:
        vertices, faces, _normals, _values = marching_cubes(
            padded, level=extraction_level, spacing=mc_spacing,
            gradient_direction=gradient_direction, step_size=int(step_size),
            allow_degenerate=False, method="lewiner", mask=mask)
    except RuntimeError as exc:
        raise MeshingError("No surface could be extracted; try step_size=1 or a different isovalue") from exc
    report["marching_cubes_seconds"] = time.perf_counter() - extraction_started
    del padded, mask, _normals, _values
    vertices = np.asarray(vertices, dtype=np.float64)
    if surface != "fraction":
        vertices *= spacing
        report["mc_coordinate_scaling"] = "normalized grid first, float64 SI scaling afterwards"
    # Padded index 1 is the original sample at local +0.5 voxel, therefore
    # subtract 0.5 voxel from the padded-grid coordinate.
    vertices -= spacing * 0.5
    faces = np.asarray(faces, dtype=np.int32)
    audit = audit_mesh(vertices, faces)
    flipped = audit["signed_volume_m3"] < 0
    if flipped:
        faces = faces[:, [0, 2, 1]].copy()
        audit["signed_volume_m3"] *= -1
    report.update(audit)
    report["orientation_flipped"] = flipped
    report["padding_samples_xyz"] = [[1, int(p)] for p in high_pad]
    report["volume_difference_m3"] = audit["volume_m3"] - source_volume
    report["volume_relative_difference"] = ((audit["volume_m3"] - source_volume) / source_volume
                                             if source_volume else None)
    if audit["degenerate_faces"] or not audit["watertight_edges"] or not audit["consistent_winding"]:
        raise MeshingError("Extracted surface failed degeneracy, edge-closure or winding validation: "
                           + _audit_failure_counts(audit))
    return vertices, faces, report


def write_binary_stl(path: str | Path, vertices_local_m: Any, faces: Any, *,
                     coordinate_unit: str = "um", chunk_size: int = 100_000) -> dict[str, Any]:
    """Atomically write a new streamed STL, refusing every existing destination.

    STL has no standard length-unit, material or translation fields.  The
    consumer must use the returned units and the extraction report's separate
    translation.  The header is a human-readable hint, not a standard unit tag.
    Output vertices and normals are IEEE float32, as required by binary STL.
    """
    units = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
    if coordinate_unit not in units:
        raise MeshingError("coordinate_unit must be one of m, mm, um, nm")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise MeshingError("chunk_size must be a positive integer")
    vertices, faces = _mesh_arrays(vertices_local_m, faces)
    if len(faces) > np.iinfo(np.uint32).max:
        raise MeshingError("binary STL cannot contain more than 2**32-1 triangles")
    scale = units[coordinate_unit]
    maximum_quantization_error_m = 0.0
    for begin in range(0, len(vertices), chunk_size):
        original = vertices[begin:begin + chunk_size]
        with np.errstate(over="ignore", invalid="ignore"):
            quantized = (original / scale).astype(np.float32)
        if not np.all(np.isfinite(quantized)):
            raise MeshingError("STL float32 coordinates overflow in the selected unit")
        error = np.linalg.norm(quantized.astype(np.float64) * scale - original, axis=1)
        if len(error):
            maximum_quantization_error_m = max(maximum_quantization_error_m, float(error.max()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record_dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                             ("attribute", "<u2")])
    if record_dtype.itemsize != 50:
        raise MeshingError("internal binary STL record layout is invalid")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            header = (f"VDATA standalone fraction isosurface; local coords {coordinate_unit}; see metadata").encode("ascii")[:80]
            stream.write(header.ljust(80, b" "))
            stream.write(struct.pack("<I", len(faces)))
            for begin in range(0, len(faces), chunk_size):
                part = faces[begin:begin + chunk_size]
                records = np.zeros(len(part), dtype=record_dtype)
                records["vertices"] = vertices[part] / scale
                # Derive normals from the actual stored float32 triangles,
                # then normalize in float64 to avoid unit-scale underflow.
                stored = records["vertices"].astype(np.float64)
                normal = np.cross(stored[:, 1] - stored[:, 0], stored[:, 2] - stored[:, 0])
                norm = np.linalg.norm(normal, axis=1)
                if np.any(norm == 0) or not np.all(np.isfinite(norm)):
                    raise MeshingError("binary STL float32 quantization collapsed a triangle")
                records["normal"] = normal / norm[:, None]
                stream.write(records.tobytes())
        # Publishing an already complete file with a hard link is atomic and
        # fails if another writer has created the destination in the meantime.
        os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return {"format": "binary_stl", "path": str(path), "faces": len(faces),
            "bytes": path.stat().st_size, "coordinate_unit": coordinate_unit,
            "metres_per_coordinate_unit": scale, "coordinate_frame": "local",
            "translation_stored_in_stl": False, "materials_stored_in_stl": False,
            "max_vertex_float32_quantization_error_m": maximum_quantization_error_m,
            "normals_from_stored_vertices": True}


def _surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int,
                     seed: int) -> tuple[np.ndarray, dict[str, int]]:
    """Bounded samples: spread vertices, face centres and area-weighted points."""
    vcount = min(count // 4, len(vertices))
    fcount = min(count // 4, len(faces))
    random_count = count - vcount - fcount
    pieces = []
    if vcount:
        indices = np.linspace(0, len(vertices) - 1, vcount, dtype=np.int64)
        pieces.append(vertices[indices])
    if fcount:
        indices = np.linspace(0, len(faces) - 1, fcount, dtype=np.int64)
        pieces.append(vertices[faces[indices]].mean(axis=1))
    if random_count:
        # One float64 value per triangle; no dense sample cloud and no full
        # (F, 3, 3) temporary.  Accumulate the CDF in place.
        area_cdf = np.empty(len(faces), dtype=np.float64)
        for begin in range(0, len(faces), 100_000):
            tri = vertices[faces[begin:begin + 100_000]]
            area_cdf[begin:begin + len(tri)] = np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        np.cumsum(area_cdf, out=area_cdf)
        if not len(area_cdf) or not math.isfinite(float(area_cdf[-1])) or area_cdf[-1] <= 0:
            raise MeshingError("cannot sample a surface with zero/non-finite area")
        rng = np.random.default_rng(seed)
        selected = np.searchsorted(area_cdf, rng.uniform(0, area_cdf[-1], random_count), side="right")
        selected = np.minimum(selected, len(faces) - 1)
        tri = vertices[faces[selected]]
        r = np.sqrt(rng.random(random_count))
        s = rng.random(random_count)
        weights = np.column_stack((1 - r, r * (1 - s), r * s))
        pieces.append(np.einsum("ij,ijk->ik", weights, tri))
    return np.concatenate(pieces), {"vertices": vcount, "face_centres": fcount,
                                    "area_weighted_barycentric": random_count}


def _sampled_distance(source_vertices: np.ndarray, source_faces: np.ndarray,
                      target_vertices: np.ndarray, target_faces: np.ndarray,
                      sample_count: int, seed: int) -> dict[str, Any]:
    """Bidirectional point-to-triangle sampling in a unit-sized local frame."""
    try:
        import trimesh
        import rtree  # noqa: F401 -- closest-point spatial index dependency
    except ImportError as exc:
        raise MeshingError("QEM distance validation requires trimesh and rtree") from exc
    lower = np.minimum(source_vertices.min(axis=0), target_vertices.min(axis=0))
    upper = np.maximum(source_vertices.max(axis=0), target_vertices.max(axis=0))
    reference = (lower + upper) * 0.5
    scale = float(np.max(upper - lower))
    if not scale > 0:
        raise MeshingError("cannot validate a zero-size surface")
    original = (source_vertices - reference) / scale
    reduced = (target_vertices - reference) / scale
    sample_original, kinds_original = _surface_samples(original, source_faces, sample_count, seed)
    sample_reduced, kinds_reduced = _surface_samples(reduced, target_faces, sample_count, seed + 1)

    def query(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
        proximity = trimesh.proximity.ProximityQuery(mesh)
        distance = np.empty(len(points), dtype=np.float64)
        for begin in range(0, len(points), 512):
            _closest, values, _triangle = proximity.on_surface(points[begin:begin + 512])
            distance[begin:begin + len(values)] = values * scale
        if not np.all(np.isfinite(distance)):
            raise MeshingError("closest-point sampling returned non-finite distances")
        return distance

    # Sequential target trees reduce the peak memory compared with keeping
    # both meshes' spatial caches and expanded triangle arrays alive together.
    forward = query(reduced, target_faces, sample_original)
    reverse = query(original, source_faces, sample_reduced)
    distances_nm = np.r_[forward, reverse] * 1e9
    return {"metric": "bidirectional sampled point-to-triangle distance",
            "maximum_nm": float(distances_nm.max()),
            "rms_nm": float(np.sqrt(np.mean(distances_nm * distances_nm))),
            "p95_nm": float(np.percentile(distances_nm, 95)),
            "source_to_reduced_maximum_nm": float(forward.max() * 1e9),
            "reduced_to_source_maximum_nm": float(reverse.max() * 1e9),
            "samples_each_direction": sample_count,
            "source_sample_kinds": kinds_original, "reduced_sample_kinds": kinds_reduced,
            "seed": seed, "is_strict_hausdorff_bound": False,
            "is_optical_position_accuracy_validation": False,
            "normalization_extent_m": scale,
            "warning": "Finite deterministic sampling may miss the true maximum geometric deviation and says nothing by itself about optical position bias."}


def simplify_mesh(vertices: Any, faces: Any, reduction: float, *,
                  max_sampled_error_nm: float | None = None,
                  sample_count: int = 5000, seed: int = 53) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optional open-source QEM with topology checks and bounded error sampling.

    ``reduction=0.45`` requests removal of 45% of the triangles.  A threshold
    only limits the *sampled* geometric deviation; it is not a guaranteed
    Hausdorff tolerance and is not an electromagnetic accuracy guarantee.
    Every reduced result must remain closed, consistently wound and retain
    the Euler characteristic.  These tests alone do not prove full topology
    preservation, vertex manifoldness or absence of self-intersection.
    """
    vertices, faces = _mesh_arrays(vertices, faces)
    if not math.isfinite(reduction) or not 0 <= reduction < 1:
        raise MeshingError("reduction must be finite and in [0, 1)")
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)) or sample_count < 4:
        raise MeshingError("sample_count must be an integer at least four")
    if max_sampled_error_nm is not None and (not math.isfinite(max_sampled_error_nm) or max_sampled_error_nm < 0):
        raise MeshingError("max_sampled_error_nm must be finite and nonnegative")
    before = audit_mesh(vertices, faces)
    if before["empty"] or before["degenerate_faces"] or not before["watertight_edges"] or not before["consistent_winding"]:
        raise MeshingError("QEM requires a nonempty, nondegenerate, edge-closed and consistently wound source mesh: "
                           + _audit_failure_counts(before))
    if reduction == 0:
        return vertices, faces, {"algorithm": "identity", "requested_reduction": 0.0,
                                "actual_reduction": 0.0, "source_audit": before,
                                "reduced_audit": dict(before),
                                "sampled_bidirectional_distance": None,
                                "geometry_unchanged": True}
    try:
        import fast_simplification
    except ImportError as exc:
        raise MeshingError("QEM requires the optional fast-simplification package") from exc
    # This library's default quadric-error thresholds assume sensible numeric
    # magnitudes.  Normalize micron-scale geometry before calling it.
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    reference = (lower + upper) * 0.5
    scale = float(np.max(upper - lower))
    normalized = (vertices - reference) / scale
    # A conservative threshold schedule was selected after the real 4.98M-
    # triangle test: agg=7 created nonmanifold edges at 95% reduction; agg=3
    # passed the 90% case.  Any result is still audited, and higher reductions
    # can legitimately be rejected.  No hole filling or relaxed validation.
    qem_aggressiveness = 3.0
    reduced_vertices, reduced_faces = fast_simplification.simplify(
        normalized, faces, target_reduction=float(reduction), agg=qem_aggressiveness,
        preserve_border=True, verbose=False)
    reduced_vertices = np.asarray(reduced_vertices, dtype=np.float64) * scale + reference
    reduced_faces = np.asarray(reduced_faces, dtype=np.int32)
    after = audit_mesh(reduced_vertices, reduced_faces)
    if after["empty"] or after["degenerate_faces"] or not after["watertight_edges"] or not after["consistent_winding"]:
        raise MeshingError("QEM result failed degeneracy, edge-closure or winding validation: "
                           + _audit_failure_counts(after))
    if after["euler_characteristic"] != before["euler_characteristic"]:
        raise MeshingError("QEM changed the Euler characteristic; refusing this reduced result")
    if before["signed_volume_m3"] * after["signed_volume_m3"] <= 0:
        raise MeshingError("QEM changed or invalidated the global surface orientation")
    distances = _sampled_distance(vertices, faces, reduced_vertices, reduced_faces,
                                 int(sample_count), int(seed))
    if max_sampled_error_nm is not None and distances["maximum_nm"] > max_sampled_error_nm:
        raise MeshingError(f"QEM sampled deviation {distances['maximum_nm']:.6g} nm exceeds the requested {max_sampled_error_nm:.6g} nm")
    return reduced_vertices, reduced_faces, {
        "algorithm": "fast_simplification/quadric_error_metrics",
        "requested_reduction": float(reduction),
        "qem_aggressiveness": qem_aggressiveness,
        "actual_reduction": 1 - len(reduced_faces) / len(faces),
        "source_audit": before, "reduced_audit": after,
        "source_volume_m3": before["volume_m3"], "reduced_volume_m3": after["volume_m3"],
        "volume_relative_difference": (after["volume_m3"] - before["volume_m3"]) / before["volume_m3"],
        "sampled_bidirectional_distance": distances,
        "max_sampled_error_nm_requested": max_sampled_error_nm,
        "topology_preservation_proven": False,
        "normalization_extent_m": scale,
        "warnings": ["QEM modifies the reconstructed geometry; it cannot recover missing process detail.",
                     "Edge closure, winding and Euler checks are necessary checks, not proof of full topology preservation.",
                     "Distance sampling is bounded and may miss the true maximum; it does not establish optical position accuracy."]}
