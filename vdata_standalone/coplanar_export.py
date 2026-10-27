"""Offline, transactional export of exact coplanar polygon facets to MAT/LSF.

This module never imports a vendor API or starts CAD.  Lumerical executes the
generated script separately.  The input mesh remains the reconstruction reference.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import time

import numpy as np


def _csr_arrays(vertices, offsets, indices, translation):
    vertices = np.asarray(vertices, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    offsets, indices = np.asarray(offsets), np.asarray(indices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Polygon vertices must be a nonempty finite N by 3 array")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Polygon translation must be a finite XYZ vector")
    if (offsets.ndim != 1 or len(offsets) < 2 or indices.ndim != 1
            or offsets.dtype.kind not in "iu" or indices.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != len(indices)
            or np.any(offsets[1:] < offsets[:-1]) or np.any(np.diff(offsets) < 3)):
        raise ValueError("Invalid single-loop polygon CSR offsets/indices")
    if not len(indices) or indices.min() < 0 or indices.max() >= len(vertices):
        raise ValueError("Polygon index is outside the vertex table")
    if len(vertices) >= 2 ** 53 or len(indices) >= 2 ** 53:
        raise ValueError("Polygon indices exceed exact double integer representation")
    return vertices, offsets, indices, translation


def write_coplanar_mat(path, vertices, offsets, indices, translation):
    """Write numeric CSR, avoiding max-loop-length by facet-count dense padding."""
    from hdf5storage import savemat

    vertices, offsets, indices, translation = _csr_arrays(vertices, offsets, indices, translation)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to replace existing polygon MAT: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        savemat(str(temporary), {
            "vdata_poly_vertices": vertices,
            "vdata_poly_flat": (indices.astype(np.float64) + 1).reshape(-1, 1),
            "vdata_poly_offsets": (offsets.astype(np.float64) + 1).reshape(-1, 1),
            "vdata_poly_translation_m": translation.reshape(3, 1),
        }, appendmat=False, format="7.3", store_python_metadata=False,
            truncate_existing=True, oned_as="column", matlab_compatible=True,
            compress=True, compression_algorithm="gzip", gzip_compression_level=7)
        os.link(temporary, path)
        return {
            "path": str(path), "bytes": path.stat().st_size,
            "vertices": len(vertices), "facets": len(offsets) - 1,
            "boundary_indices": len(indices), "largest_loop": int(np.diff(offsets).max()),
            "vertices_unit": "m", "vertex_frame": "local", "indices_base": 1,
            "offsets": "one-based start positions plus final sentinel; single loop per facet",
            "format": "MATLAB v7.3 HDF5 numeric CSR; variable-length cells constructed in LSF",
            "cad_import_executed": False, "optical_simulation_run": False,
        }
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def compact_vertex_table(vertices, indices):
    """Remove only unreferenced rows; preserve every referenced coordinate bit."""
    vertices = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(indices)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or indices.min() < 0 or indices.max() >= len(vertices)):
        raise ValueError("Invalid vertex table or polygon indices for compaction")
    used = np.unique(indices)
    lookup = np.full(len(vertices), -1, dtype=np.int64)
    lookup[used] = np.arange(len(used), dtype=np.int64)
    remapped = lookup[indices]
    compacted = np.ascontiguousarray(vertices[used])
    if not np.array_equal(used[remapped], indices):
        raise RuntimeError("Vertex compaction failed its inverse-index audit")
    report = {
        "enabled": True, "input_vertices": len(vertices), "output_vertices": len(compacted),
        "removed_unreferenced_vertices": len(vertices) - len(compacted),
        "raw_coordinate_bytes_removed": (len(vertices) - len(compacted)) * 3 * 8,
        "coordinate_movement": 0, "referenced_coordinate_bits_preserved": True,
        "inverse_index_audit_passed": True,
        "scope": "Unused vertex rows only; all facet loops and their coordinate sequences are unchanged.",
    }
    return compacted, remapped, report


def generate_coplanar_lsf(objects, folder, *, compact_vertices=False):
    from .cli import _lsf_string

    key = "coplanar_compact_mat" if compact_vertices else "coplanar_mat"
    suffix = "_compact" if compact_vertices else ""
    group = "VDATA_standalone"
    lines = [
        "# Exact coplanar facets exported offline without any vendor SDK.",
        "# Open an empty layout. On FDTD 2025 R2, use the Classic viewport.",
        "# Units: local metres; apply the stored translation exactly once.",
        "# Numeric CSR avoids a dense max-loop-length by facet-count table.",
        "# Do not read the full facets property back: CAD may densely pad it.",
        "# Geometry only: no FDTD region, source, monitors or EM solve.",
    ]
    missing = [o["source_name"] for o in objects if not o.get("fdtd_material")]
    if missing:
        lines += ["?" + _lsf_string("Import blocked: resolve material_map.json for " + ", ".join(missing)) + ";"]
    else:
        lines += ["switchtolayout;", "vdata_poly_ready=1;",
                  "if (getnamednumber(" + _lsf_string(group) + ")>0) {vdata_poly_ready=0;",
                  "?" + _lsf_string("Destination group exists; use an empty layout.") + ";}"]
        for material in sorted({o["fdtd_material"] for o in objects}):
            lines += ["if (materialexists(" + _lsf_string(material) + ")==0) {vdata_poly_ready=0;",
                      "?" + _lsf_string("Material missing from this FDTD library: " + material) + ";}"]
        lines += ["if (vdata_poly_ready) {", "vdata_poly_redraw=redrawmode;", "redrawoff;",
                  "addstructuregroup;", 'set("name",' + _lsf_string(group) + ");"]
        for obj in objects:
            mat = (Path(folder) / obj[key]).resolve()
            lines += ["?" + _lsf_string("Loading coplanar material " + str(obj["material_id"]) + "...") + ";",
                      "matlabload(" + _lsf_string(mat.as_posix()) + ");",
                      "vdata_poly_n=length(vdata_poly_offsets)-1;",
                      "vdata_poly_facets=cell(vdata_poly_n);",
                      "for(vdata_poly_i=1:vdata_poly_n){",
                      "vdata_poly_facets{vdata_poly_i}=cell(1);",
                      "vdata_poly_first=vdata_poly_offsets(vdata_poly_i);",
                      "vdata_poly_last=vdata_poly_offsets(vdata_poly_i+1)-1;",
                      "vdata_poly_facets{vdata_poly_i}{1}=transpose(vdata_poly_flat(vdata_poly_first:vdata_poly_last));",
                      "}", "addplanarsolid(vdata_poly_vertices,vdata_poly_facets);",
                      'set("name",' + _lsf_string("material_" + str(obj["material_id"])) + ");",
                      'set("material",' + _lsf_string(obj["fdtd_material"]) + ");",
                      'set("x",vdata_poly_translation_m(1));',
                      'set("y",vdata_poly_translation_m(2));',
                      'set("z",vdata_poly_translation_m(3));',
                      "addtogroup(" + _lsf_string(group) + ");",
                      'clear(vdata_poly_facets,vdata_poly_vertices,vdata_poly_flat,vdata_poly_offsets,vdata_poly_translation_m);']
        lines += ["unselectall;", "redrawmode(vdata_poly_redraw);",
                  '?"Geometry imported. Configure the optical simulation before running FDTD.";', "}"]
    path = Path(folder) / ("import_lumerical_coplanar" + suffix + ".lsf")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _inside(folder, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Expected a nonempty relative artifact path")
    target = (folder / relative).resolve()
    if folder not in target.parents:
        raise ValueError("Artifact path is outside the conversion folder")
    return target


def export_coplanar(args):
    """Extend an existing completed conversion, with full batch rollback."""
    import json
    from .cli import _load_mapping, _lsf_string, file_hash, write_json
    from .coplanar import merge_coplanar

    started = time.perf_counter()
    minimum = args.min_faces
    if type(minimum) is not int or minimum < 2:
        raise ValueError("Coplanar minimum face count must be at least 2")
    compact = bool(getattr(args, "compact_vertices", False))
    folder = Path(args.folder).expanduser().resolve()
    manifest_path = folder / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "vdata-standalone-conversion-v1" or manifest.get("status") != "complete":
        raise ValueError("Expected a completed standalone conversion manifest")
    objects = copy.deepcopy(manifest.get("objects", []))
    if not objects or not all(isinstance(o, dict) for o in objects):
        raise ValueError("This conversion has no surface meshes")
    mapping = _load_mapping(args.material_map)
    hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
    # Validate both variants, even when this call reuses the other variant.
    # Otherwise rebuilding manifest hashes could silently adopt a damaged file.
    for obj in objects:
        for artifact_key in ("coplanar_mat", "coplanar_compact_mat"):
            if obj.get(artifact_key):
                artifact = _inside(folder, obj[artifact_key])
                if not artifact.is_file() or file_hash(artifact) != hashes.get(obj[artifact_key]):
                    raise ValueError("Existing polygon MAT is missing or differs from its manifest")
    for report_name in ("coplanar_report.json", "coplanar_compact_report.json"):
        if report_name in hashes:
            artifact = _inside(folder, report_name)
            if not artifact.is_file() or file_hash(artifact) != hashes[report_name]:
                raise ValueError("Existing polygon report is missing or differs from its manifest")
    mat_key = "coplanar_compact_mat" if compact else "coplanar_mat"
    export_key = "coplanar_compact_export" if compact else "coplanar_export"
    suffix = "_compact" if compact else ""
    report_path = folder / ("coplanar" + suffix + "_report.json")
    if report_path.exists() and (not report_path.is_file() or file_hash(report_path) != hashes.get(report_path.name)):
        raise ValueError("Existing coplanar report is unrecorded or differs from its manifest")
    planned, targets, material_ids = [], set(), set()
    for obj in objects:
        mid = obj.get("material_id")
        if type(mid) is not int or not 0 <= mid <= 255 or mid in material_ids:
            raise ValueError("Invalid or duplicate material ID in conversion manifest")
        material_ids.add(mid)
        source_key = "compact_mesh" if obj.get("compact_mesh") else "indexed_mesh"
        source = _inside(folder, obj.get(source_key))
        if not source.is_file() or file_hash(source) != hashes.get(obj[source_key]):
            raise ValueError("Source indexed mesh is missing or differs from its manifest")
        relative = obj.get(mat_key) or "meshes/material_{:04d}_coplanar{}.mat".format(mid, suffix)
        target = _inside(folder, relative)
        if target in targets:
            raise ValueError("Duplicate coplanar MAT target")
        targets.add(target)
        cached = bool(obj.get(mat_key))
        if cached:
            if not target.is_file() or file_hash(target) != hashes.get(relative):
                raise ValueError("Existing coplanar MAT is missing or differs from its manifest")
            previous = obj.get(export_key, {})
            if (previous.get("min_faces") != minimum
                    or previous.get("source_mesh_sha256") != hashes[obj[source_key]]
                    or previous.get("compact_vertices") is not compact):
                raise ValueError("Existing coplanar export uses different options or source; use a fresh conversion folder")
            if not previous.get("mat", {}).get("format", "").startswith("MATLAB v7.3"):
                raise ValueError("Existing polygon MAT uses the superseded legacy format; use a fresh conversion folder for HDF5 MAT")
        elif target.exists() or target.is_symlink():
            raise FileExistsError("Refusing to replace unrecorded polygon MAT: " + str(target))
        name = obj.get("source_name")
        if not isinstance(name, str):
            raise ValueError("Missing source material name")
        obj["fdtd_material"] = mapping.get("id:" + str(mid), mapping.get(str(mid), mapping.get(name, obj.get("fdtd_material"))))
        _lsf_string(name)
        if obj["fdtd_material"]:
            _lsf_string(obj["fdtd_material"])
        _lsf_string(target.as_posix())
        if compact and not cached and obj.get("coplanar_mat"):
            baseline = _inside(folder, obj["coplanar_mat"])
            if not baseline.is_file() or file_hash(baseline) != hashes.get(obj["coplanar_mat"]):
                raise ValueError("Baseline coplanar MAT is missing or differs from its manifest")
        planned.append((obj, source_key, source, relative, cached))

    script_path = folder / ("import_lumerical_coplanar" + suffix + ".lsf")
    text_paths = [script_path, report_path, folder / "material_map.json", manifest_path]
    originals = {}
    for target in text_paths:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError("Generated text destination must be a regular local file")
        originals[target] = target.read_bytes() if target.exists() else None
    created = []
    text_started = False
    try:
        for obj, source_key, source, relative, cached in planned:
            if cached:
                continue
            print("Merging strictly coplanar patches for material {}...".format(obj["material_id"]), flush=True)
            if source_key == "compact_mesh":
                from .storage import read_mesh_archive
                vertices, faces, translation, _ = read_mesh_archive(source)
            else:
                with np.load(source, allow_pickle=False) as archive:
                    vertices, faces, translation = (archive[k] for k in ("vertices_local_m", "faces", "translation_m"))
            if file_hash(source) != hashes[obj[source_key]]:
                raise ValueError("Source mesh changed during coplanar export")
            if not np.array_equal(translation, np.asarray(obj["translation_m"], dtype=np.float64)):
                raise ValueError("Source mesh translation differs from its manifest")
            previous = obj.get("coplanar_export", {})
            reuse = (compact and obj.get("coplanar_mat")
                     and previous.get("min_faces") == minimum
                     and previous.get("source_mesh_sha256") == hashes[obj[source_key]])
            if reuse:
                from hdf5storage import loadmat
                baseline = loadmat(_inside(folder, obj["coplanar_mat"]))
                if baseline["vdata_poly_vertices"].tobytes() != vertices.tobytes():
                    raise ValueError("Baseline polygon vertex table differs from the source mesh")
                if not np.array_equal(baseline["vdata_poly_translation_m"].ravel(), translation):
                    raise ValueError("Baseline polygon translation differs from the source mesh")
                offsets_float = baseline["vdata_poly_offsets"].ravel() - 1
                indices_float = baseline["vdata_poly_flat"].ravel() - 1
                for array in (offsets_float, indices_float):
                    if (not np.isfinite(array).all() or np.any(array < 0)
                            or np.any(array >= 2 ** 53) or not np.array_equal(array, np.rint(array))):
                        raise ValueError("Baseline polygon CSR contains invalid integer values")
                offsets, indices = offsets_float.astype(np.int64), indices_float.astype(np.int64)
                _csr_arrays(vertices, offsets, indices, translation)
                merge = copy.deepcopy(previous["merge"])
                del baseline, offsets_float, indices_float
            else:
                offsets, indices, merge = merge_coplanar(vertices, faces, min_faces=minimum)
            vertex_report = {"enabled": False}
            if compact:
                vertices, indices, vertex_report = compact_vertex_table(vertices, indices)
            target = _inside(folder, relative)
            written = write_coplanar_mat(target, vertices, offsets, indices, translation)
            created.append((target, target.stat().st_ino))
            obj[mat_key] = relative
            obj[export_key] = {
                "min_faces": minimum, "compact_vertices": compact,
                "source_mesh_sha256": hashes[obj[source_key]], "merge": merge, "mat": written,
                "vertex_compaction": vertex_report, "reused_audited_baseline_polygons": bool(reuse),
            }
            del vertices, faces, offsets, indices
        text_started = True
        for obj, source_key, source, _relative, _cached in planned:
            if file_hash(source) != hashes[obj[source_key]]:
                raise ValueError("Source mesh changed before coplanar export commit")
        generated = generate_coplanar_lsf(objects, folder, compact_vertices=compact)
        report = {
            "schema": "vdata-coplanar-export-v1", "status": "complete",
            "min_faces": minimum, "compact_vertices": compact,
            "objects": [{"material_id": o["material_id"], **o[export_key]} for o in objects],
            "elapsed_seconds": time.perf_counter() - started,
            "coventor_sdk_called": False, "cad_import_executed": False, "optical_simulation_run": False,
            "scope": "Existing reconstructed mesh geometry only; no new physical accuracy claim. Independent material meshes are not a conforming multi-material partition.",
        }
        write_json(report_path, report)
        write_json(folder / "material_map.json", {"materials": {"id:" + str(o["material_id"]): o["fdtd_material"] for o in objects}})
        manifest["objects"] = objects
        manifest["coplanar" + suffix + "_export_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["files"] = [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
                             for p in sorted(folder.rglob("*")) if p.is_file() and p != manifest_path]
        write_json(manifest_path, manifest)
    except BaseException:
        for target, inode in reversed(created):
            if target.is_file() and not target.is_symlink() and target.stat().st_ino == inode:
                target.unlink()
        if text_started:
            for target, content in originals.items():
                if content is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(content)
        raise
    print("Generated coplanar MAT/LSF (CAD import not executed): " + str(generated), flush=True)
    return 0
