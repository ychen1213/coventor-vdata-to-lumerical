"""Strict, independent reader for the verified three-byte Coventor VoxelXML.

Only Python's standard library is used for inspection.  Array decoding imports
NumPy lazily.  No vendor module, executable, licence setting or external file
reference is consulted.  This is deliberately NOT a decoder for every format
that may use the .vdata extension.

Supported payload: base64-encoded sequences of ``!zlib!`` + network-order
``uint32 raw_size, uint32 packed_size`` + one complete zlib stream.  The decoded
bytes are C-order [X,Y,Z,(fraction_code, material_1, material_2)] records.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import math
import operator
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib


MAX_XML_BYTES = 512 * 1024**2
DEFAULT_MAX_RAW_BYTES = 8 * 1024**3
_CHUNK = 64 * 1024
_UNIT_M = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "metre": 1.0, "metres": 1.0,
    "cm": 1e-2, "mm": 1e-3,
    "um": 1e-6, "µm": 1e-6, "μm": 1e-6,
    "micron": 1e-6, "microns": 1e-6,
    "micrometer": 1e-6, "micrometers": 1e-6,
    "micrometre": 1e-6, "micrometres": 1e-6,
    "nm": 1e-9, "nanometer": 1e-9, "nanometers": 1e-9,
    "nanometre": 1e-9, "nanometres": 1e-9,
    "a": 1e-10, "å": 1e-10, "angstrom": 1e-10, "angstroms": 1e-10,
}


class _NoDoctypeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError("DOCTYPE and entity declarations are not supported")


def _unit_factor(unit):
    if unit is None or str(unit).strip().lower() not in _UNIT_M:
        raise ValueError(f"Unsupported or missing length unit: {unit!r}")
    return _UNIT_M[str(unit).strip().lower()]


def _one(parent, tag):
    matches = parent.findall(tag)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {parent.tag}/{tag}")
    return matches[0]


def _parse_document(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    initial_stat = path.stat()
    if initial_stat.st_size <= 0 or initial_stat.st_size > MAX_XML_BYTES:
        raise ValueError(f"XML input size must be 1..{MAX_XML_BYTES} bytes")
    digest = hashlib.sha256()
    parser = ET.XMLParser(target=_NoDoctypeBuilder())
    consumed = 0
    try:
        with path.open("rb") as source:
            for part in iter(lambda: source.read(_CHUNK), b""):
                consumed += len(part)
                if consumed > MAX_XML_BYTES:
                    raise ValueError("XML input exceeds the parser size limit")
                digest.update(part)
                parser.feed(part)
        root = parser.close()
    except ET.ParseError as exc:
        raise ValueError(f"Not a readable VoxelXML document: {exc}") from exc
    final_stat = path.stat()
    if (consumed != initial_stat.st_size or
            final_stat.st_size != initial_stat.st_size or
            final_stat.st_mtime_ns != initial_stat.st_mtime_ns):
        raise ValueError("Input changed while being read")
    if root.tag != "VoxelXML":
        raise ValueError(f"Unsupported XML root: {root.tag!r}; expected VoxelXML")
    if any(k.lower() in {"encoding", "format", "compression"} for k in root.attrib):
        raise ValueError("Unverified VoxelXML encoding/format declaration")
    props = _one(root, "Properties")
    size, counts, origin = (_one(props, name) for name in
                            ("VoxelSize", "NumVoxels", "ModelOrigin"))
    try:
        step = float(size.attrib["Value"]) * _unit_factor(size.get("Units"))
        dimensions = [int(counts.attrib[k]) for k in ("NX", "NY", "NZ")]
        origin_m = [float(origin.attrib[k]) * _unit_factor(origin.get("Units"))
                    for k in "XYZ"]
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Incomplete or invalid grid metadata: {exc}") from exc
    if not math.isfinite(step) or step <= 0 or any(n <= 0 for n in dimensions):
        raise ValueError("Voxel size and dimensions must be finite and positive")
    if not all(math.isfinite(v) for v in origin_m):
        raise ValueError("Model origin must contain finite coordinates")
    upper = [x + n * step for x, n in zip(origin_m, dimensions)]
    if not all(math.isfinite(v) for v in upper):
        raise ValueError("Nonfinite model bounds")

    materials = []
    seen = set()
    for item in _one(root, "Materials"):
        if item.get("MaterialID") is None:
            # The known palette contains a visual default with no material ID.
            continue
        try:
            mid = int(item.attrib["MaterialID"])
            rgb = [int(item.get(k, "128")) for k in ("red", "green", "blue")]
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid material ID or RGB colour") from exc
        if not 0 <= mid <= 255 or any(not 0 <= c <= 255 for c in rgb):
            raise ValueError("Material IDs and RGB colours must lie in 0..255")
        if mid in seen:
            raise ValueError(f"Duplicate material ID: {mid}")
        seen.add(mid)
        name = item.get("name")
        if not name:
            raise ValueError(f"Material {mid} has no name")
        materials.append({"id": mid, "name": name, "color_rgb": rgb,
                          "type": item.get("type", "unspecified"), "implicit": False})
    declared_ids = sorted(seen)
    if 255 not in seen:
        materials.append({"id": 255, "name": "Air", "color_rgb": [255, 255, 255],
                          "type": "implicit air", "implicit": True})
    nodes = root.findall("VoxelData")
    if len(nodes) != sum(1 for _ in root.iter("VoxelData")):
        raise ValueError("Nested VoxelData is not a verified schema")
    if not nodes or any(not (node.text or "").strip() for node in nodes):
        raise ValueError("Missing or empty VoxelData payload")
    for node in nodes:
        if len(node):
            raise ValueError("VoxelData must contain only base64 text")
        if node.attrib and node.attrib != {"encoding": "base64"}:
            raise ValueError(f"Unverified VoxelData encoding: {node.attrib!r}")
    optional = [child.tag for child in root
                if child.tag not in {"Properties", "Materials", "VoxelData"}]
    voxel_count = math.prod(dimensions)
    plane_present = any(True for _ in root.iter("PlaneEquations"))
    meta = {
        "schema": "Coventor VoxelXML: !zlib! blocks, 3-byte ff/m1/m2 records",
        "schema_version": 1,
        "input_vdata": str(path), "source_sha256": digest.hexdigest(),
        "file_bytes": consumed, "xml_root": root.tag,
        "xml_attributes": dict(root.attrib), "dimensions_xyz": dimensions,
        "axis_order": "X,Y,Z,record-byte; C-order (Z changes fastest)",
        "voxel_size_m": step, "origin_m": origin_m, "origin_kind": "lower_corner",
        "bounds_m": [origin_m, upper],
        "original_voxel_unit": size.get("Units"),
        "original_origin_unit": origin.get("Units"),
        "voxel_count": voxel_count, "bytes_per_voxel": 3,
        "expected_raw_bytes": voxel_count * 3,
        "materials": materials, "declared_material_ids": declared_ids,
        "has_plane_equations": plane_present, "optional_fields": optional,
        "optional_fields_decoded": [],
        "optional_fields_note": "Only voxel material records are decoded. Plane equations, "
                                "doping and other optional fields are not included in the array.",
        "external_references": {
            tag: dict(props.find(tag).attrib)
            for tag in ("ModelFile", "LayoutFile", "ProcessFile", "MaterialFile")
            if props.find(tag) is not None
        },
        "external_references_followed": False,
        "fraction_rule": "ff=129: single material and m1=m2; otherwise fraction(m1)=ff/128",
        "record_validation_performed": False,
    }
    return nodes, meta


def inspect_vdata(path):
    """Inspect metadata without importing NumPy or decompressing voxel records.

    The hash covers the source file bytes.  This function does not prove that
    the compressed body is valid; use :func:`decode_vdata` to validate it.
    XML input is capped at 512 MiB, independently of the decoded-data cap.
    """
    return _parse_document(path)[1]


def _base64_chunks(text):
    carry = ""
    padded = False
    for start in range(0, len(text), _CHUNK):
        part = "".join(text[start:start + _CHUNK].split())
        if not part:
            continue
        if padded:
            raise ValueError("Base64 data follows end padding")
        carry += part
        end = len(carry) // 4 * 4
        if not end:
            continue
        encoded, carry = carry[:end], carry[end:]
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 VoxelData: {exc}") from exc
        if "=" in encoded:
            padded = True
            if carry:
                raise ValueError("Base64 data follows end padding")
        yield decoded
    if carry:
        raise ValueError("Incomplete base64 quartet")


class _ByteReader:
    def __init__(self, text):
        self.parts = iter(_base64_chunks(text))
        self.pending = bytearray()

    def take(self, count, *, eof_ok=False):
        while len(self.pending) < count:
            try:
                self.pending.extend(next(self.parts))
            except StopIteration:
                if eof_ok and not self.pending:
                    return b""
                raise ValueError("Truncated !zlib! block header or payload") from None
        result = bytes(self.pending[:count])
        del self.pending[:count]
        return result


def _inflate_block(reader, packed_size, expected_size):
    """Yield bounded pieces and never call unbounded zlib.flush()."""
    decoder = zlib.decompressobj()
    emitted = 0
    remaining = packed_size
    try:
        while remaining:
            part = reader.take(min(_CHUNK, remaining))
            remaining -= len(part)
            while part:
                chunk = decoder.decompress(part, min(_CHUNK, expected_size - emitted + 1))
                emitted += len(chunk)
                if emitted > expected_size:
                    raise ValueError("Compressed block exceeds its declared raw size")
                if decoder.unused_data:
                    raise ValueError("Trailing bytes or concatenated streams inside a zlib block")
                part = decoder.unconsumed_tail
                if chunk:
                    yield chunk
            if decoder.eof and remaining:
                raise ValueError("Zlib stream ends before its declared packed size")
    except zlib.error as exc:
        raise ValueError(f"Invalid zlib stream: {exc}") from exc
    if not decoder.eof or decoder.unconsumed_tail or emitted != expected_size:
        raise ValueError("Incomplete zlib stream or mismatched raw size")


def _validate_records(raw, meta):
    import numpy as np
    records = raw.reshape(-1, 3)
    allowed = np.zeros(256, dtype=bool)
    allowed[[m["id"] for m in meta["materials"]]] = True
    present = np.zeros(256, dtype=bool)
    for start in range(0, len(records), 1024 * 1024):
        part = records[start:start + 1024 * 1024]
        ff, m1, m2 = part[:, 0], part[:, 1], part[:, 2]
        if np.any(ff > 129):
            raise ValueError("Invalid fraction code: expected 0..128 or single-material code 129")
        if np.any((ff == 129) & (m1 != m2)):
            raise ValueError("Single-material ff=129 record must have m1=m2")
        ids = np.unique(part[:, 1:])
        if np.any(~allowed[ids]):
            raise ValueError(f"Unknown material IDs in voxel data: {ids[~allowed[ids]].tolist()}")
        present[ids] = True
    meta["material_ids_in_records"] = np.flatnonzero(present).tolist()
    meta["record_validation_performed"] = True


def decode_vdata(path, output_path=None, *, max_raw_bytes=DEFAULT_MAX_RAW_BYTES):
    """Decode exact source record bytes to a uint8 ``(NX,NY,NZ,3)`` array.

    ``output_path`` creates a new .npy memory map and refuses to overwrite an
    existing file.  Its parent directory must exist.  If validation fails the
    newly created incomplete output is removed.  There is no binarization,
    resampling, smoothing or surface reconstruction in this operation.
    """
    import numpy as np

    try:
        cap = operator.index(max_raw_bytes)
    except TypeError as exc:
        raise ValueError("max_raw_bytes must be an integer byte limit") from exc
    if cap <= 0:
        raise ValueError("max_raw_bytes must be positive")
    nodes, meta = _parse_document(path)
    expected = meta["expected_raw_bytes"]
    if expected > cap:
        raise ValueError(f"Decoded size {expected} exceeds max_raw_bytes={cap}")
    if expected > np.iinfo(np.intp).max:
        raise ValueError("Decoded array is too large for this Python platform")
    shape = tuple(meta["dimensions_xyz"]) + (3,)
    target = None if output_path is None else Path(output_path).resolve()
    raw = None
    flat = None
    created = False
    try:
        if target is None:
            raw = np.empty(shape, dtype=np.uint8, order="C")
        else:
            # Exclusive reservation protects both the original input and other
            # outputs against accidental overwrite, including an identical path.
            with target.open("xb"):
                pass
            created = True
            raw = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint8, shape=shape)
        flat = raw.reshape(-1)
        count = 0
        blocks = 0
        for node in nodes:
            reader = _ByteReader(node.text or "")
            while True:
                header = reader.take(14, eof_ok=True)
                if not header:
                    break
                if header[:6] != b"!zlib!":
                    raise ValueError("Unsupported block signature; expected !zlib!")
                unpacked_size, packed_size = struct.unpack("!II", header[6:])
                if unpacked_size <= 0 or packed_size <= 0:
                    raise ValueError("Compressed block lengths must be positive")
                if count + unpacked_size > expected:
                    raise ValueError("Declared block output exceeds the expected array size")
                for chunk in _inflate_block(reader, packed_size, unpacked_size):
                    end = count + len(chunk)
                    flat[count:end] = np.frombuffer(chunk, dtype=np.uint8)
                    count = end
                blocks += 1
        if count != expected:
            raise ValueError(f"Decoded length {count} differs from expected {expected}")
        _validate_records(raw, meta)
        meta.update({"compressed_blocks": blocks, "decoded_raw_bytes": count,
                     "raw_dtype": "uint8", "raw_shape": list(shape),
                     "decoded_bytes_preserved": True})
        if target is not None:
            raw.flush()
            meta["raw_npy"] = str(target)
        return raw, meta
    except Exception:
        if created:
            flat = None
            if raw is not None and getattr(raw, "_mmap", None) is not None:
                raw._mmap.close()
            raw = None
            target.unlink(missing_ok=True)
        raise


def fraction_for_material(raw, material_id):
    """Return float32 material occupancy, preserving the source ff/128 rule.

    This scalar fraction does not reconstruct an interface normal or recover
    unresolved geometry.  PlaneEquations is intentionally not interpreted.
    """
    import numpy as np
    try:
        mid = operator.index(material_id)
    except TypeError as exc:
        raise ValueError("material_id must be an integer in 0..255") from exc
    if not 0 <= mid <= 255:
        raise ValueError("material_id must be an integer in 0..255")
    if (not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or
            raw.ndim != 4 or raw.shape[-1] != 3 or not raw.flags.c_contiguous):
        raise ValueError("Expected a C-contiguous uint8 array with shape (NX,NY,NZ,3)")
    out = np.empty(raw.shape[:-1], dtype=np.float32)
    records, values = raw.reshape(-1, 3), out.reshape(-1)
    for start in range(0, len(records), _CHUNK):
        part = records[start:start + _CHUNK]
        ff, m1, m2 = part[:, 0], part[:, 1], part[:, 2]
        if np.any(ff > 129) or np.any((ff == 129) & (m1 != m2)):
            raise ValueError("Invalid source fraction record")
        fraction1 = ff.astype(np.float32) / np.float32(128)
        fraction1[ff == 129] = 1
        values[start:start + len(part)] = ((m1 == mid) * fraction1 +
                                          (m2 == mid) * (1 - fraction1))
    return out
