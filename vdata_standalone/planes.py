"""Independent decoding and explicitly experimental use of VoxelXML planes.

The observed payload is little-endian float32 (nx, ny, nz, d) in XYZ order.
The observed geometric convention is n.dot(u) = d, where u is in [0,1]^3
relative to the voxel's lower corner.  This convention was checked against
saved native measurements for one AH53 crop; it is not a published universal
Coventor interchange specification.  No vendor code is loaded here.
"""
from __future__ import annotations

import hashlib
import math
import operator
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from .reader import (_ByteReader, _CHUNK, _inflate_block, _NoDoctypeBuilder,
                     DEFAULT_MAX_RAW_BYTES, MAX_XML_BYTES, inspect_vdata,
                     fraction_for_material)


def decode_plane_equations(path, output_path=None, *,
                           max_raw_bytes=DEFAULT_MAX_RAW_BYTES):
    """Decode exact plane bytes; return (float32 XYZ4 array, metadata).

    Accept only the observed !zlib! / float32-LE schema. Zero normals are
    allowed (no valid interface plane). Finite unit normals are required for
    nonzero records, preventing common wrong-byte-order interpretations.
    Optional output is a newly created .npy mmap. No source file is modified.
    The returned metadata explicitly separates byte decoding from the
    empirical geometric convention. Doping and other fields are not decoded.
    """
    try:
        cap = operator.index(max_raw_bytes)
    except TypeError as exc:
        raise ValueError("max_raw_bytes must be an integer") from exc
    if cap <= 0:
        raise ValueError("max_raw_bytes must be positive")
    meta = inspect_vdata(path)
    if not meta["has_plane_equations"]:
        raise ValueError("VDATA contains no PlaneEquations payload")
    expected = math.prod(meta["dimensions_xyz"]) * 16
    if expected > cap or expected > np.iinfo(np.intp).max:
        raise ValueError(f"Plane array size {expected} exceeds the decoded byte limit")
    parser = ET.XMLParser(target=_NoDoctypeBuilder())
    source_hash = hashlib.sha256()
    consumed = 0
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(_CHUNK), b""):
            consumed += len(part)
            if consumed > MAX_XML_BYTES:
                raise ValueError("XML input exceeds parser size limit")
            source_hash.update(part)
            parser.feed(part)
    root = parser.close()
    if source_hash.hexdigest() != meta["source_sha256"]:
        raise ValueError("Input changed between metadata and plane decoding")
    nodes = root.findall("PlaneEquations")
    if not nodes or len(nodes) != sum(1 for _ in root.iter("PlaneEquations")):
        raise ValueError("Only top-level PlaneEquations nodes are supported")
    for node in nodes:
        if len(node) or not (node.text or "").strip():
            raise ValueError("PlaneEquations must contain nonempty base64 text only")
        if node.attrib and node.attrib != {"encoding": "base64"}:
            raise ValueError("Unverified PlaneEquations encoding declaration")
    shape = tuple(meta["dimensions_xyz"]) + (4,)
    target = None if output_path is None else Path(output_path).resolve()
    planes, flat = None, None
    created = False
    try:
        if target is None:
            planes = np.empty(shape, dtype="<f4", order="C")
        else:
            with target.open("xb"):
                pass
            created = True
            planes = np.lib.format.open_memmap(target, mode="w+", dtype="<f4", shape=shape)
        flat = planes.view(np.uint8).reshape(-1)
        count = blocks = 0
        raw_hash = hashlib.sha256()
        for node in nodes:
            reader = _ByteReader(node.text or "")
            while True:
                header = reader.take(14, eof_ok=True)
                if not header:
                    break
                if header[:6] != b"!zlib!":
                    raise ValueError("Unsupported plane block signature")
                raw_size, packed_size = struct.unpack("!II", header[6:])
                if raw_size <= 0 or packed_size <= 0 or count + raw_size > expected:
                    raise ValueError("Invalid plane block sizes")
                for chunk in _inflate_block(reader, packed_size, raw_size):
                    flat[count:count + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
                    count += len(chunk)
                    raw_hash.update(chunk)
                blocks += 1
        if count != expected:
            raise ValueError(f"Decoded plane length {count} differs from expected {expected}")
        nonzero_count = 0
        minimum, maximum = float("inf"), 0.0
        for start in range(0, planes.size // 4, _CHUNK):
            values = planes.reshape(-1, 4)[start:start + _CHUNK].astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite plane values; unsupported data or byte order")
            lengths = np.linalg.norm(values[:, :3], axis=1)
            active = lengths > 0
            if np.any(abs(lengths[active] - 1) > 1e-3):
                raise ValueError("Unverified nonzero plane normals: expected unit lengths")
            if np.any(active):
                nonzero_count += int(active.sum())
                minimum = min(minimum, float(lengths[active].min()))
                maximum = max(maximum, float(lengths[active].max()))
        if target is not None:
            planes.flush()
        report = {
            "source_sha256": meta["source_sha256"],
            "raw_plane_sha256": raw_hash.hexdigest(),
            "dimensions_xyz": meta["dimensions_xyz"], "plane_shape": list(shape),
            "origin_m": meta["origin_m"], "voxel_size_m": meta["voxel_size_m"],
            "dtype": "<f4", "axis_order": "X,Y,Z,(nx,ny,nz,d), C-order",
            "decoded_bytes": count, "compressed_blocks": blocks,
            "decoded_bytes_preserved": True, "nonzero_normal_count": nonzero_count,
            "normal_length_range": [minimum, maximum] if nonzero_count else None,
            "plane_equation": "dot(n,u)=d; u=(world-origin)/h-voxel_index",
            "coordinate_convention_status": "Empirically verified on saved AH53 crop measurements, not a universal vendor specification",
            "nonzero_normal_length_tolerance": 1e-3,
            "vendor_sdk_called": False, "external_references_followed": False,
        }
        return planes, report
    except Exception:
        if created:
            flat = None
            if planes is not None and getattr(planes, "_mmap", None) is not None:
                planes._mmap.close()
            planes = None
            target.unlink(missing_ok=True)
        raise


def plane_halfspace_fraction(normals, offsets):
    """Volume fraction of [0,1]^3 satisfying n.u <= d, for auditing.

    Inclusion-exclusion integrates a sum of independent uniform variables.
    Reflection about the cube center reduces cancellation. Components below
    1e-7 of the largest normal component are omitted, a deliberately disclosed
    numerical tolerance for this audit, not a physical geometry correction.
    """
    normals = np.asarray(normals, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if normals.shape != offsets.shape + (3,) or not np.isfinite(normals).all() or not np.isfinite(offsets).all():
        raise ValueError("Expected finite normals [...,3] and offsets [...]")
    scale = np.max(abs(normals), axis=-1)
    if np.any(scale == 0):
        raise ValueError("A zero normal has no defined halfspace fraction")
    n = normals / scale[..., None]
    d = offsets / scale
    weights = abs(n)
    # A dropped component contributes its midpoint to preserve centering.
    dropped = weights < 1e-7
    threshold = d - np.minimum(n, 0).sum(axis=-1) - np.where(dropped, weights * .5, 0).sum(axis=-1)
    weights = np.where(dropped, 0, weights)
    total = weights.sum(axis=-1)
    reflected = threshold > total * .5
    threshold = np.where(reflected, total - threshold, threshold)
    result = np.zeros(offsets.shape, dtype=np.float64)
    active_dimensions = (weights > 0).sum(axis=-1)
    for dimension in (1, 2, 3):
        mask = active_dimensions == dimension
        w, t = weights[mask], threshold[mask]
        value = np.zeros(len(t), dtype=np.float64)
        denominator = np.prod(np.where(w > 0, w, 1), axis=-1) * math.factorial(dimension)
        for bits in range(8):
            selected = np.array([(bits >> axis) & 1 for axis in range(3)])
            legal = np.all((selected == 0) | (w > 0), axis=-1)
            value += (-1) ** int(selected.sum()) * np.maximum(t - (w * selected).sum(axis=-1), 0) ** dimension * legal
        result[mask] = value / denominator
    result = np.where(reflected, 1 - result, result)
    return np.clip(result, 0, 1)


def _statistics(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "rms": float(np.sqrt(np.mean(values ** 2))),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values))}


def plane_center_field(raw, planes, material_id, voxel_size_m):
    """Experimental negative-inside center field from stored plane offsets.

    This is deliberately restricted to one target material plus implicit Air
    (255). Mixed records use material1 on the n.u <= d side; normals are not
    individually flipped to force agreement with quantized fill fractions.
    Pure cells obtain their inside/outside sign from the material label. A
    zero-normal mixed cell falls back to (0.5-fraction)*h. The field is later
    interpolated by marching cubes; this is not exact clipped-plane meshing.
    """
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    if not isinstance(planes, np.ndarray) or planes.shape != raw.shape[:-1] + (4,):
        raise ValueError("Plane array dimensions do not match voxel records")
    fraction = fraction_for_material(np.ascontiguousarray(raw), material_id)
    ids = np.unique(raw[..., 1:])
    if material_id == 255 or any(int(v) not in {material_id, 255} for v in ids):
        raise ValueError("Plane surface currently requires one target solid and Air (255); multi-material orientation is unverified")
    if not np.isfinite(planes).all():
        raise ValueError("Plane array contains nonfinite values")
    field = np.empty(fraction.shape, dtype=np.float32)
    fflat, pflat = fraction.reshape(-1), planes.reshape(-1, 4)
    rflat, oflat = raw.reshape(-1, 3), field.reshape(-1)
    errors, best_errors = [], []
    valid_count = pure_flips = mixed_valid = mixed_missing = 0
    for start in range(0, len(fflat), _CHUNK):
        f = fflat[start:start + _CHUNK]
        p = pflat[start:start + _CHUNK].astype(np.float64)
        r = rflat[start:start + _CHUNK]
        norm = np.linalg.norm(p[:, :3], axis=1)
        valid = norm > 0
        if np.any(abs(norm[valid] - 1) > 1e-3):
            raise ValueError("Unverified non-unit plane normals")
        valid_count += int(valid.sum())
        mixed = (f > 0) & (f < 1)
        distance = (.5 - f) * voxel_size_m
        signed = (.5 * p[:, :3].sum(axis=1) - p[:, 3]) * voxel_size_m / np.where(valid, norm, 1)
        # The observed convention defines material1 as the negative side.
        signed = np.where(r[:, 1] == material_id, signed, -signed)
        pure = ~mixed
        pure_flips += int(np.count_nonzero(valid & pure & ((signed < 0) != (f == 1)) & (signed != 0)))
        signed = np.where(pure, np.where(f == 1, -abs(signed), abs(signed)), signed)
        distance[valid] = signed[valid]
        oflat[start:start + len(f)] = distance
        selected = mixed & valid
        mixed_valid += int(selected.sum())
        mixed_missing += int(np.count_nonzero(mixed & ~valid))
        if np.any(selected):
            volumes = plane_halfspace_fraction(p[selected, :3], p[selected, 3])
            volumes = np.where(r[selected, 1] == material_id, volumes, 1 - volumes)
            errors.append(abs(volumes - f[selected]))
            best_errors.append(np.minimum(abs(volumes - f[selected]), abs(1 - volumes - f[selected])))
    error = np.concatenate(errors) if errors else np.empty(0)
    best_error = np.concatenate(best_errors) if best_errors else np.empty(0)
    report = {
        "method": "Stored-plane center signed field; negative inside; interpolated zero contour",
        "experimental": True, "plane_equations_used": True,
        "coordinate_convention": "dot(n,local_corner_normalized_xyz)=d",
        "material_side_convention": "material1 negative halfspace; material2 positive; pure signs from labels",
        "plane_offsets_modified": False, "normalization": "center residual divided by normal length and multiplied by h",
        "nonzero_plane_count": valid_count, "mixed_voxels_with_planes": mixed_valid,
        "mixed_voxels_without_planes": mixed_missing, "pure_sign_corrections": pure_flips,
        "mixed_fraction_error": _statistics(error),
        "mixed_fraction_error_best_of_either_orientation": _statistics(best_error),
        "mixed_within_one_fraction_quantum": int(np.count_nonzero(error <= 1 / 128)),
        "fraction_quantum": 1 / 128,
        "halfspace_audit_relative_component_tolerance": 1e-7,
        "warnings": [
            "The plane-coordinate and material-side conventions are empirical for the tested VoxelXML files.",
            "Exported plane halfspace volumes can disagree with stored material fractions; offsets are preserved rather than forced to match them.",
            "Center sampling and marching-cubes interpolation are approximate, not an exact union of clipped voxel planes.",
            "Pure-cell sign correction and zero-normal fallback can change the interpolated interface; the field is not a global signed-distance function.",
            "Crop padding creates closure surfaces. No general native-equivalence, physical-accuracy, or FDTD position-error guarantee is made.",
        ],
    }
    return field, report
