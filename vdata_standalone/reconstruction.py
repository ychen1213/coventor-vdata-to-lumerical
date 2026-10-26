"""Optional volume-fraction-guided interface reconstruction without vendor code.

The default fraction isosurface treats voxel averages like point samples.  This
module instead estimates a local plane normal and inverts the volume of a plane
cut through the original rectangular voxel.  This is a PLIC-inspired *distance
field*, not a conforming, exactly volume-conserving PLIC surface mesh.  Curvature,
multiple interfaces in one voxel, and sharp edges remain underdetermined.
"""

from __future__ import annotations

from itertools import product
import math
from typing import Any

import numpy as np


class ReconstructionError(ValueError):
    """Invalid fraction data or unsupported reconstruction request."""


def pad_distance_field(field: Any) -> np.ndarray:
    """One negative ghost layer with reflected amplitudes at the crop faces.

    A constant negative padding would move the artificial cut boundary when
    extrapolated interior distances exceed half a voxel.  Opposite amplitudes
    instead place a linear zero crossing halfway between the inside sample and
    its ghost.  Corner chamfers from MC interpolation are still possible.
    """
    value = np.asarray(field, dtype=np.float32)
    if value.ndim != 3 or any(n < 1 for n in value.shape) or not np.all(np.isfinite(value)):
        raise ReconstructionError("field must be a nonempty finite XYZ array")
    padded = np.pad(value, 1, mode="edge")
    for axis in range(3):
        for end, adjacent in ((0, 1), (-1, -2)):
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            dst[axis], src[axis] = end, adjacent
            padded[tuple(dst)] = -np.maximum(np.abs(padded[tuple(src)]), 1e-6)
    return padded


def box_fraction(normal_xyz: Any, offset: Any, voxel_size_xyz: Any = (1., 1., 1.)) -> np.ndarray:
    """Volume fraction of ``normal dot x <= offset`` in a box centred at zero.

    ``normal_xyz[..., 3]`` and ``offset`` broadcast.  Box side lengths may be
    anisotropic.  Normals need not be normalized.  The formula is the CDF of a
    sum of independent uniform random variables, evaluated by inclusion and
    exclusion.  Negligible widths below 1e-7 of the largest width are treated
    as zero to avoid catastrophic cancellation close to axis-aligned planes;
    this perturbs the cut offset by at most 1.5e-7 of the largest width.
    """
    n = np.asarray(normal_xyz, dtype=np.float64)
    h = np.asarray(voxel_size_xyz, dtype=np.float64)
    if n.ndim < 1 or n.shape[-1] != 3 or h.shape != (3,):
        raise ReconstructionError("normal_xyz must end in XYZ and voxel_size_xyz must have three values")
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ReconstructionError("normals and positive voxel sizes must be finite")
    a = np.asarray(offset, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ReconstructionError("plane offsets must be finite")
    shape = np.broadcast_shapes(n.shape[:-1], a.shape)
    widths = np.broadcast_to(np.abs(n * h), shape + (3,)).reshape(-1, 3).copy()
    a = np.broadcast_to(a, shape).reshape(-1)
    largest = widths.max(axis=1)
    if np.any(largest == 0):
        raise ReconstructionError("zero normal does not define a plane")
    widths[widths < largest[:, None] * 1e-7] = 0
    widths.sort(axis=1)
    total = widths.sum(axis=1)
    t = np.clip(a + total * .5, 0., total)
    # Symmetry both halves computation and reduces cancellation in the tails.
    flip = t > total * .5
    t = np.minimum(t, total - t)
    result = np.zeros(len(t), dtype=np.float64)
    dimensions = np.count_nonzero(widths, axis=1)
    for d in (1, 2, 3):
        take = dimensions == d
        if not np.any(take):
            continue
        w = widths[take, 3 - d:]
        u = t[take]
        acc = np.zeros(len(w), dtype=np.float64)
        for subset in product((0, 1), repeat=d):
            shift = np.sum(w * np.asarray(subset), axis=1)
            acc += (-1 if sum(subset) % 2 else 1) * np.maximum(u - shift, 0.) ** d
        result[take] = acc / (math.factorial(d) * np.prod(w, axis=1))
        if d > 1:
            # A dominant axis has an exactly linear central region.  This
            # branch avoids subtracting almost equal large polynomials.
            central = u >= np.sum(w[:, :-1], axis=1)
            if np.any(central):
                r = result[take]
                r[central] = (u[central] - .5 * np.sum(w[central, :-1], axis=1)) / w[central, -1]
                result[take] = r
    result = np.clip(result, 0., 1.)
    result[flip] = 1. - result[flip]
    return result.reshape(shape)


def solve_plane_offsets(fraction: Any, normal_xyz: Any,
                        voxel_size_xyz: Any = (1., 1., 1.), *, iterations: int = 40) -> np.ndarray:
    """Find the centred plane offset whose half-space has the given fraction."""
    c = np.asarray(fraction, dtype=np.float64)
    n = np.asarray(normal_xyz, dtype=np.float64)
    h = np.asarray(voxel_size_xyz, dtype=np.float64)
    if np.any(~np.isfinite(c)) or np.any((c < 0) | (c > 1)):
        raise ReconstructionError("fraction must be finite and between zero and one")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 16:
        raise ReconstructionError("iterations must be an integer >= 16")
    # Validate and establish broadcast shape through the public geometric API.
    box_fraction(n, np.zeros_like(c), h)
    half = .5 * np.sum(np.abs(n * h), axis=-1)
    lo, hi = np.broadcast_arrays(-half, half, c)[:2]
    lo, hi = lo.copy(), hi.copy()
    for _ in range(iterations):
        mid = .5 * (lo + hi)
        below = box_fraction(n, mid, h) < c
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    # Central symmetry makes c=1/2 exactly a=0.  Preserve that identity rather
    # than introducing a bisection-scale negative epsilon: MC can otherwise
    # turn an exact grid-vertex crossing into extremely small sliver faces.
    return np.where(c == .5, 0., .5 * (lo + hi))


def _sample(field: np.ndarray, coords: np.ndarray, delta: np.ndarray) -> np.ndarray:
    ijk = np.clip(coords + delta, 0, np.asarray(field.shape) - 1)
    return field[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float64)


def estimate_normals(fraction: np.ndarray, coordinates: np.ndarray,
                     voxel_size_xyz: Any = (1., 1., 1.)) -> tuple[np.ndarray, dict[str, int]]:
    """Outward normals: weighted central gradients, then valid height columns.

    Five-cell columns are used only if their ends are pure opposite phases and
    all five samples are monotone.  Under a single planar-interface assumption,
    integrated column heights reproduce exact slopes; otherwise the weighted
    gradient is the explicit fallback.  No fraction values are filtered.
    """
    c = np.asarray(fraction)
    coords = np.asarray(coordinates, dtype=np.int64)
    h = np.asarray(voxel_size_xyz, dtype=np.float64)
    gradient = np.zeros((len(coords), 3), dtype=np.float64)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for j, k in product((-1, 0, 1), repeat=2):
            d = np.zeros(3, dtype=int)
            d[others] = (j, k)
            d[axis] = 1
            right = _sample(c, coords, d)
            d[axis] = -1
            left = _sample(c, coords, d)
            weight = (2 if j == 0 else 1) * (2 if k == 0 else 1)
            gradient[:, axis] += weight * (right - left) / (32 * h[axis])
    normals = -gradient
    lengths = np.linalg.norm(normals, axis=1)
    zero = lengths < 1e-12 / float(h.max())
    normals[zero] = (1., 0., 0.)
    dominant = np.argmax(np.abs(normals * h), axis=1)
    heights_used = 0
    for axis in range(3):
        selected = np.flatnonzero((dominant == axis) & ~zero)
        if not len(selected):
            continue
        xyz = coords[selected]
        others = [i for i in range(3) if i != axis]
        columns = []
        valid = np.all((xyz >= 2) & (xyz < np.asarray(c.shape) - 2), axis=1)
        outward = np.sign(normals[selected, axis])
        for other, side in ((axis, 0), (others[0], -1), (others[0], 1), (others[1], -1), (others[1], 1)):
            column = np.empty((len(xyz), 5), dtype=np.float64)
            for slot, step in enumerate(range(-2, 3)):
                d = np.zeros(3, dtype=int)
                d[axis] = step
                if side:
                    d[other] = side
                column[:, slot] = _sample(c, xyz, d)
            valid &= np.where(outward > 0,
                              (column[:, 0] == 1) & (column[:, -1] == 0),
                              (column[:, 0] == 0) & (column[:, -1] == 1))
            valid &= np.all(np.diff(column, axis=1) * outward[:, None] <= 1e-7, axis=1)
            columns.append(column.sum(axis=1))
        height_normal = np.zeros_like(normals[selected])
        height_normal[:, axis] = outward
        for j, other in enumerate(others):
            height_normal[:, other] = -(columns[2 + j * 2] - columns[1 + j * 2]) * h[axis] / (2 * h[other])
        normals[selected[valid]] = height_normal[valid]
        heights_used += int(np.count_nonzero(valid))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals[zero] = 0.
    return normals, {"height_function_cells": heights_used,
                     "gradient_fallback_cells": int(len(coords) - heights_used - np.count_nonzero(zero)),
                     "undefined_normal_cells": int(np.count_nonzero(zero))}


def reconstruct_fraction_distance(fraction: Any, voxel_size_m: Any, *,
                                  normals_xyz: Any = None,
                                  chunk_size: int = 50_000) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a positive-inside distance field and a reconstruction audit.

    Distances are divided by min(voxel_size_m), which avoids tiny SI values in
    marching cubes.  Extract its zero isosurface, not the 0.5 isosurface.
    ``normals_xyz`` optionally supplies outward normals for all cells (XYZ,3)
    or a constant XYZ vector.  Raw fractions are never modified.  Only mixed
    cells and their six face-neighbour pure cells receive plane distances.
    Near ambiguous corners neighbouring extrapolations are averaged, with the
    sign of pure cells preserved.  This is not a signed-distance PDE solution.
    """
    c = np.asarray(fraction)
    h_m = np.asarray(voxel_size_m, dtype=np.float64)
    if c.ndim != 3 or any(n < 1 for n in c.shape) or not np.all(np.isfinite(c)) or np.any((c < 0) | (c > 1)):
        raise ReconstructionError("fraction must be a nonempty finite XYZ array in [0,1]")
    if h_m.shape != (3,) or np.any(~np.isfinite(h_m)) or np.any(h_m <= 0):
        raise ReconstructionError("voxel_size_m must be three positive finite values")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ReconstructionError("chunk_size must be a positive integer")
    scale_m = float(h_m.min())
    h = h_m / scale_m
    coords = np.argwhere((c > 0) & (c < 1))
    field = (c.astype(np.float32) - .5)
    report: dict[str, Any] = {
        "method": "volume_fraction_plane_distance/height_function_or_weighted_gradient",
        "mixed_cells": int(len(coords)), "distance_unit_m": scale_m,
        "height_function_cells": 0, "gradient_fallback_cells": 0,
        "undefined_normal_cells": 0, "max_local_fraction_residual": 0.,
        "calibrated_mixed_cells": 0,
        "pure_neighbour_sign_clamps": 0, "plane_equations_used": False,
        "per_cell_plic_volume_constraint": True,
        "final_mc_volume_conservation_guaranteed": False,
        "supplied_normals": normals_xyz is not None,
        "limitations": [
            "A single local plane is assumed; fractions do not uniquely determine normals or topology.",
            "The final interpolated MC surface does not exactly conserve each voxel volume.",
            "Curvature, corners, thin layers, and multiple interfaces per cell may bias reconstruction.",
            "Pure neighbour extrapolations are averaged and sign-limited; this is not exact PLIC stitching.",
            "Voxel fraction quantization and absent physical detail cannot be undone.",
            "Crop and exterior closure are imposed by the caller and create artificial surfaces.",
        ],
    }
    if not len(coords):
        report["per_cell_plic_volume_constraint"] = False
        report["empty_mixed_band"] = True
        return field, report
    n_input = None if normals_xyz is None else np.asarray(normals_xyz, dtype=np.float64)
    if n_input is not None and n_input.shape not in ((3,), c.shape + (3,)):
        raise ReconstructionError("normals_xyz must be one XYZ vector or fraction.shape + (3,)")
    # Sparse pure-neighbour accumulation avoids three more full 3-D arrays.
    neighbour_indices: list[np.ndarray] = []
    neighbour_values: list[np.ndarray] = []
    for start in range(0, len(coords), chunk_size):
        ijk = coords[start:start + chunk_size]
        if n_input is None:
            n, normal_report = estimate_normals(c, ijk, h)
            for key, value in normal_report.items():
                report[key] += value
        else:
            n = np.broadcast_to(n_input, (len(ijk), 3)).copy() if n_input.shape == (3,) else n_input[tuple(ijk.T)].copy()
            norm = np.linalg.norm(n, axis=1)
            if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
                raise ReconstructionError("supplied mixed-cell normals must be finite and nonzero")
            n /= norm[:, None]
        defined = np.linalg.norm(n, axis=1) > 0
        ijk, n = ijk[defined], n[defined]
        if not len(ijk):
            continue
        report["calibrated_mixed_cells"] += len(ijk)
        value = c[tuple(ijk.T)].astype(np.float64)
        offset = solve_plane_offsets(value, n, h, iterations=32)
        residual = np.abs(box_fraction(n, offset, h) - value)
        report["max_local_fraction_residual"] = max(report["max_local_fraction_residual"], float(residual.max()))
        field[tuple(ijk.T)] = offset.astype(np.float32)
        for axis, side in product(range(3), (-1, 1)):
            neighbour = ijk.copy()
            neighbour[:, axis] += side
            valid = (neighbour[:, axis] >= 0) & (neighbour[:, axis] < c.shape[axis])
            q = neighbour[valid]
            pure = (c[tuple(q.T)] == 0) | (c[tuple(q.T)] == 1)
            if not np.any(pure):
                continue
            q = q[pure]
            dist = (offset[valid] - n[valid, axis] * h[axis] * side)[pure]
            inside = c[tuple(q.T)] > .5
            wrong = np.where(inside, dist <= 0, dist >= 0)
            report["pure_neighbour_sign_clamps"] += int(np.count_nonzero(wrong))
            dist = np.where(wrong, np.where(inside, 1e-6, -1e-6), dist)
            neighbour_indices.append(np.ravel_multi_index(q.T, c.shape))
            neighbour_values.append(dist.astype(np.float32))
    if neighbour_indices:
        indices = np.concatenate(neighbour_indices)
        values = np.concatenate(neighbour_values)
        order = np.argsort(indices, kind="stable")
        indices, values = indices[order], values[order]
        begins = np.r_[0, np.flatnonzero(indices[1:] != indices[:-1]) + 1]
        counts = np.diff(np.r_[begins, len(indices)])
        means = np.add.reduceat(values.astype(np.float64), begins) / counts
        field.ravel()[indices[begins]] = means.astype(np.float32)
        report["extrapolated_pure_cells"] = len(begins)
    else:
        report["extrapolated_pure_cells"] = 0
    if report["undefined_normal_cells"]:
        report["per_cell_plic_volume_constraint"] = False
        report["limitations"].append("Cells with undefined normals retain the original fraction field values.")
    return field, report
