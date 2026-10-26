"""Bit-exact, independently decodable storage for an indexed triangle mesh.

VMESH version 1 is an ordinary ZIP_STORED container. ``manifest.json`` describes
three arrays and their independently compressed blocks. No vertex coordinates,
triangle indices, ordering, or IEEE 754 bits are changed. In particular the
integer difference below operates on the *bits* of each float64, never on its
floating-point numerical value. The transformations are reversible modulo 2**w.

Each block: unsigned words -> row differences per column -> modular zigzag ->
little-endian byte shuffle -> XZ/LZMA (or zlib). The first row is differenced
against zero. Each block and complete array has a SHA256 of the original bytes.
Geometry remains Cartesian local metres; ``translation_m`` is stored separately.
Only NumPy and the Python standard library are required. No vendor runtime.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator
import zipfile
import zlib

import numpy as np


class StorageError(ValueError):
    """An unsupported, corrupted, or invalid mesh archive."""


FORMAT = "vdata-standalone-indexed-mesh"
VERSION = 1
ARRAY_NAMES = ("vertices_local_m", "faces", "translation_m")
TRANSFORM = "word-delta-zigzag-byte-shuffle-v1"
MAX_BLOCK_BYTES = 64 * 1024**2
MAX_MANIFEST_BYTES = 16 * 1024**2


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _word_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype(dtype.str[0] + "u" + str(dtype.itemsize))


def _encode(block: np.ndarray, packed_dtype: np.dtype) -> bytes:
    # A float array is reinterpreted bit-for-bit, not converted to integers.
    if block.dtype.kind == "f":
        words = block.view(_word_dtype(block.dtype)).astype(packed_dtype, copy=True)
    else:
        words = block.astype(packed_dtype, copy=True)
    if words.ndim == 1:
        words = words.reshape(-1, 1)
    words[1:] -= words[:-1].copy()
    width = packed_dtype.itemsize * 8
    words = (words << 1) ^ (np.zeros_like(words) - (words >> (width - 1)))
    return words.view(np.uint8).reshape(-1, packed_dtype.itemsize).T.copy().tobytes()


def _decode(data: bytes, shape: tuple[int, ...], dtype: np.dtype,
            packed_dtype: np.dtype) -> np.ndarray:
    words = np.frombuffer(data, dtype=np.uint8).reshape(packed_dtype.itemsize, -1)
    words = words.T.copy().reshape(-1).view(packed_dtype).reshape(shape)
    words = (words >> 1) ^ (np.zeros_like(words) - (words & 1))
    np.cumsum(words, axis=0, dtype=packed_dtype, out=words)
    if dtype.kind == "f":
        return words.astype(_word_dtype(dtype), copy=False).view(dtype)
    return words.astype(dtype, copy=False)


def _bounded_decompress(data: bytes, codec: str, expected: int) -> bytes:
    try:
        if codec == "lzma":
            stream = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=256 * 1024**2)
            result = stream.decompress(data, max_length=expected + 1)
            if len(result) != expected or not stream.eof or stream.unused_data:
                raise StorageError("Invalid LZMA block length or trailing compressed data")
        else:
            stream = zlib.decompressobj()
            result = stream.decompress(data, expected + 1)
            if (len(result) != expected or not stream.eof or stream.unused_data
                    or stream.unconsumed_tail):
                raise StorageError("Invalid zlib block length or trailing compressed data")
        return result
    except (lzma.LZMAError, zlib.error) as exc:
        raise StorageError("Corrupted compressed mesh block") from exc


def _validate_inputs(vertices: Any, faces: Any, translation: Any):
    vertices, faces, translation = map(np.asarray, (vertices, faces, translation))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind != "f" or vertices.dtype.itemsize != 8:
        raise StorageError("vertices_local_m must be an (N, 3) float64 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or faces.dtype.itemsize not in (1, 2, 4, 8):
        raise StorageError("faces must be an (M, 3) integer array")
    if translation.shape != (3,) or translation.dtype.kind != "f" or translation.dtype.itemsize != 8:
        raise StorageError("translation_m must contain three float64 values")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(translation)):
        raise StorageError("mesh coordinates must be finite")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise StorageError("face vertex index is outside the vertex array")
    return vertices, faces, translation


def write_mesh_archive(path: str | Path, vertices: Any, faces: Any,
                       translation_m: Any, metadata: dict[str, Any] | None = None,
                       *, codec: str = "lzma", level: int = 3,
                       chunk_rows: int = 262144) -> dict[str, Any]:
    """Write an atomic new archive, preserving array dtype/shape/order/bytes.

    ``codec='lzma', level=3`` balances compression and working memory. ``zlib``
    provides a faster, usually larger alternative. ``chunk_rows`` bounds coding
    temporaries; input arrays remain owned by the caller. Existing files are never
    overwritten. A reader does not need this module: the format uses JSON, ZIP,
    standard compression, and the reversible transform described above.
    """
    started = time.perf_counter()
    arrays = dict(zip(ARRAY_NAMES, _validate_inputs(vertices, faces, translation_m)))
    if codec not in ("lzma", "zlib") or type(level) is not int or not 0 <= level <= 9:
        raise StorageError("codec must be lzma/zlib and level must be 0..9")
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= MAX_BLOCK_BYTES // 24:
        raise StorageError("chunk_rows must be positive and fit the 64 MiB block limit")
    if metadata is not None and not isinstance(metadata, dict):
        raise StorageError("metadata must be a JSON object")
    # Check before creating the output file; do not silently serialize NaN.
    metadata = json.loads(json.dumps(metadata or {}, ensure_ascii=False, allow_nan=False))
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT, "version": VERSION, "coordinate_unit": "m",
                "coordinate_frame": "local_plus_translation_m", "lossless": True,
                "geometry_changed": False, "arrays": {}, "metadata": metadata}
    handle, temporary = tempfile.mkstemp(prefix=".vmesh-", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, array in arrays.items():
                packed_dtype = np.dtype("<u8" if array.dtype.kind == "f" or (array.size and int(array.max()) > 0xFFFFFFFF) else "<u4")
                descriptor = {"dtype": array.dtype.str, "shape": list(array.shape),
                              "packed_dtype": packed_dtype.str, "transform": TRANSFORM,
                              "codec": codec, "codec_level": level, "blocks": []}
                aggregate = hashlib.sha256()
                for index, start in enumerate(range(0, len(array), chunk_rows)):
                    block = np.ascontiguousarray(array[start:start + chunk_rows])
                    original = memoryview(block).cast("B")
                    aggregate.update(original)
                    transformed = _encode(block, packed_dtype)
                    compressed = lzma.compress(transformed, preset=level) if codec == "lzma" else zlib.compress(transformed, level)
                    member = f"arrays/{name}/{index:08d}.{'xz' if codec == 'lzma' else 'zlib'}"
                    archive.writestr(member, compressed)
                    descriptor["blocks"].append({"member": member, "row_start": start,
                        "row_count": len(block), "encoded_bytes": len(transformed),
                        "compressed_bytes": len(compressed), "original_bytes": len(original),
                        "sha256": hashlib.sha256(original).hexdigest()})
                descriptor["sha256"] = aggregate.hexdigest()
                descriptor["original_bytes"] = int(array.nbytes)
                manifest["arrays"][name] = descriptor
            document = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            if len(document) > MAX_MANIFEST_BYTES:
                raise StorageError("Manifest exceeds size limit; use larger chunks")
            archive.writestr("manifest.json", document)
        # link fails if a concurrent writer has already created destination.
        os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    size = destination.stat().st_size
    original_bytes = sum(a.nbytes for a in arrays.values())
    return {"path": str(destination.resolve()), "format": FORMAT, "version": VERSION,
            "archive_bytes": size, "original_array_bytes": int(original_bytes),
            "compression_ratio": float(original_bytes / size),
            "vertex_count": len(arrays["vertices_local_m"]), "face_count": len(arrays["faces"]),
            "codec": codec, "codec_level": level, "chunk_rows": chunk_rows,
            "compression_seconds": time.perf_counter() - started,
            "lossless": True, "geometry_changed": False, "sdk_used": False,
            "array_sha256": {name: desc["sha256"] for name, desc in manifest["arrays"].items()}}


def _manifest(archive: zipfile.ZipFile, max_raw_bytes: int) -> dict[str, Any]:
    try:
        members = archive.infolist()
        by_name = {m.filename: m for m in members}
        if len(members) != len(by_name):
            raise StorageError("Duplicate archive members")
        info = by_name["manifest.json"]
        if info.compress_type != zipfile.ZIP_STORED or info.file_size > MAX_MANIFEST_BYTES:
            raise StorageError("Invalid manifest compression or size")
        manifest = json.loads(archive.read(info))
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise StorageError("Unsupported VMESH format/version")
        if manifest.get("coordinate_unit") != "m" or manifest.get("coordinate_frame") != "local_plus_translation_m":
            raise StorageError("Unsupported coordinate convention")
        if manifest.get("lossless") is not True or manifest.get("geometry_changed") is not False:
            raise StorageError("Unexpected lossless-geometry declaration")
        if not isinstance(manifest.get("metadata"), dict) or set(manifest["arrays"]) != set(ARRAY_NAMES):
            raise StorageError("Invalid metadata or array set")
        total_raw, expected_members = 0, {"manifest.json"}
        for name in ARRAY_NAMES:
            descriptor = manifest["arrays"][name]
            dtype, packed = np.dtype(descriptor["dtype"]), np.dtype(descriptor["packed_dtype"])
            shape = descriptor["shape"]
            if (not isinstance(shape, list) or not all(type(v) is int and v >= 0 for v in shape)
                    or (name == "translation_m" and shape != [3])
                    or (name != "translation_m" and (len(shape) != 2 or shape[1] != 3))):
                raise StorageError("Invalid array shape")
            if ((name != "faces" and (dtype.kind != "f" or dtype.itemsize != 8))
                    or (name == "faces" and (dtype.kind not in "iu" or dtype.itemsize not in (1, 2, 4, 8)))
                    or packed.str not in ("<u4", "<u8")
                    or (name != "faces" and packed.itemsize != 8)):
                raise StorageError("Invalid original/packed dtype")
            if descriptor["transform"] != TRANSFORM or descriptor["codec"] not in ("lzma", "zlib"):
                raise StorageError("Unsupported transform or compression")
            width = 1 if len(shape) == 1 else 3
            expected_size = shape[0] * width * dtype.itemsize
            total_raw += expected_size
            if total_raw > max_raw_bytes or descriptor["original_bytes"] != expected_size:
                raise StorageError("Array size limit exceeded or byte count inconsistent")
            next_row = 0
            for block in descriptor["blocks"]:
                count = block["row_count"]
                if type(count) is not int or count < 1 or block["row_start"] != next_row:
                    raise StorageError("Non-contiguous block rows")
                next_row += count
                if next_row > shape[0] or block["encoded_bytes"] != count * width * packed.itemsize:
                    raise StorageError("Invalid block shape/length")
                if block["original_bytes"] != count * width * dtype.itemsize or block["encoded_bytes"] > MAX_BLOCK_BYTES:
                    raise StorageError("Invalid original block size or excessive encoded block")
                member = block["member"]
                member_info = by_name[member]
                if (member in expected_members or member_info.compress_type != zipfile.ZIP_STORED
                        or member_info.file_size != block["compressed_bytes"]
                        or member_info.file_size > MAX_BLOCK_BYTES + 1024**2):
                    raise StorageError("Invalid or excessive compressed member")
                expected_members.add(member)
            if next_row != shape[0]:
                raise StorageError("Incomplete block coverage")
        if set(by_name) != expected_members:
            raise StorageError("Unexpected archive members")
        return manifest
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError, OverflowError) as exc:
        raise StorageError("Malformed VMESH manifest") from exc


def inspect_mesh_archive(path: str | Path, *, max_raw_bytes: int = 8 * 1024**3) -> dict[str, Any]:
    """Read and validate metadata/dimensions without decompressing arrays."""
    try:
        with zipfile.ZipFile(path) as archive:
            return _manifest(archive, max_raw_bytes)
    except zipfile.BadZipFile as exc:
        raise StorageError("Invalid ZIP mesh archive") from exc


def _blocks(archive: zipfile.ZipFile, descriptor: dict[str, Any]) -> Iterator[np.ndarray]:
    aggregate = hashlib.sha256()
    dtype, packed = np.dtype(descriptor["dtype"]), np.dtype(descriptor["packed_dtype"])
    for block in descriptor["blocks"]:
        transformed = _bounded_decompress(archive.read(block["member"]), descriptor["codec"], block["encoded_bytes"])
        shape = (block["row_count"], *descriptor["shape"][1:])
        result = _decode(transformed, shape, dtype, packed)
        raw = memoryview(np.ascontiguousarray(result)).cast("B")
        if hashlib.sha256(raw).hexdigest() != block["sha256"]:
            raise StorageError("Decoded block SHA256 mismatch")
        aggregate.update(raw)
        yield result
    if aggregate.hexdigest() != descriptor["sha256"]:
        raise StorageError("Decoded array SHA256 mismatch")


def iter_mesh_array_blocks(path: str | Path, array_name: str,
                           *, max_raw_bytes: int = 8 * 1024**3) -> Iterator[np.ndarray]:
    """Decode just one array in bounded blocks. Exhaust to check its whole SHA.

    Individual block hashes are checked before yielding. A caller that stops early
    verifies only the consumed prefix, not the complete array checksum.
    """
    if array_name not in ARRAY_NAMES:
        raise StorageError(f"Unknown mesh array: {array_name}")
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = _manifest(archive, max_raw_bytes)
            yield from _blocks(archive, manifest["arrays"][array_name])
    except zipfile.BadZipFile as exc:
        raise StorageError("Invalid/corrupted ZIP mesh archive") from exc


def read_mesh_archive(path: str | Path, *, max_raw_bytes: int = 8 * 1024**3
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Restore complete original arrays plus user metadata, checking every SHA.

    This convenience API allocates the full decoded arrays. Use block iteration
    when only a stream is needed; compressed file size is not decoded RAM usage.
    """
    arrays = {}
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = _manifest(archive, max_raw_bytes)
            for name in ARRAY_NAMES:
                descriptor = manifest["arrays"][name]
                array = np.empty(descriptor["shape"], dtype=descriptor["dtype"])
                start = 0
                for block in _blocks(archive, descriptor):
                    array[start:start + len(block)] = block
                    start += len(block)
                arrays[name] = array
    except zipfile.BadZipFile as exc:
        raise StorageError("Invalid/corrupted ZIP mesh archive") from exc
    _validate_inputs(arrays["vertices_local_m"], arrays["faces"], arrays["translation_m"])
    return arrays["vertices_local_m"], arrays["faces"], arrays["translation_m"], manifest["metadata"]
