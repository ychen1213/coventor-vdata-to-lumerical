"""SDK-free command-line conversion. This module never launches another process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

from . import __version__
from .reader import decode_vdata, inspect_vdata, fraction_for_material


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _crop(meta, bounds_um):
    import numpy as np
    shape = np.asarray(meta["dimensions_xyz"], dtype=np.int64)
    if bounds_um is None:
        return np.zeros(3, dtype=np.int64), shape
    b = np.asarray(bounds_um, dtype=np.float64).reshape(3, 2) * 1e-6
    if not np.isfinite(b).all() or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("Crop bounds must be finite increasing pairs: xmin xmax ymin ymax zmin zmax")
    q = (b - np.asarray(meta["origin_m"])[:, None]) / meta["voxel_size_m"]
    grid = np.rint(q).astype(np.int64)
    if not np.allclose(q, grid, rtol=0, atol=1e-6):
        raise ValueError("Crop boundaries must coincide with source voxel faces; no silent snapping")
    lo, hi = grid[:, 0], grid[:, 1]
    if np.any(lo < 0) or np.any(hi > shape):
        raise ValueError("Crop lies outside the VDATA bounds")
    return lo, hi


def _default_optical(name):
    leaf = str(name).replace("\\", "/").rsplit("/", 1)[-1].lower()
    if leaf in {"si_xtal", "silicon", "crystalline silicon", "crystalline_si"}:
        return "Si (Silicon) - Palik"
    if leaf in {"sio2", "silicon dioxide"}:
        return "SiO2 (Glass) - Palik"
    return None


def _background(item):
    leaf = str(item.get("name", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return int(item["id"]) == 255 or leaf in {"air", "void", "vacuum"}


def _load_mapping(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Material map must be a JSON object")
    data = data.get("materials", data)
    if not isinstance(data, dict):
        raise ValueError("Material map materials must be an object")
    for key, value in data.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Material mapping values must be nonempty strings or null")
    return data


def _lsf_string(value):
    # LSF strings can contain neither raw control characters nor unescaped quotes.
    s = str(value).replace("\\", "/")
    if any(ord(c) < 32 for c in s):
        raise ValueError("Control characters are unsupported in LSF names and paths")
    return '"' + s.replace('"', '\\"') + '"'


def generate_lsf(objects, folder):
    """Write plain script text only. A proprietary FSP is created by FDTD itself."""
    group = "VDATA_standalone"
    lines = [
        "# Generated without Coventor or Lumerical. Run in an empty FDTD layout.",
        "# STL numbers are micrometres; object positions below are metres.",
        "# No source, optical solve or FSP file is produced by this converter.",
    ]
    missing = [o["source_name"] for o in objects if not o["fdtd_material"]]
    unexpanded = [o for o in objects if not o.get("stl")]
    if unexpanded:
        lines += ["# Lossless compact meshes must first be expanded to STL.",
                  "# Run: python run.py export-stl OUTPUT_FOLDER",
                  '?' + _lsf_string("Import blocked: run export-stl to expand the compact mesh archive.") + ';']
    elif missing:
        lines += ["# Resolve material_map.json and regenerate the conversion script.",
                  "?" + _lsf_string("Import blocked: material mapping required for " + ", ".join(missing)) + ";"]
    else:
        lines += ["switchtolayout;", "vdata_standalone_ready=1;",
                  "if (getnamednumber(" + _lsf_string(group) + ")>0) { vdata_standalone_ready=0;",
                  "?" + _lsf_string("Destination group already exists; use an empty layout.") + "; }"]
        for name in sorted({o["fdtd_material"] for o in objects}):
            lines += ["if (materialexists(" + _lsf_string(name) + ")==0) { vdata_standalone_ready=0;",
                      "?" + _lsf_string("Material missing from this FDTD library: " + name) + "; }"]
        lines += ["if (vdata_standalone_ready) {", "addstructuregroup;", 'set("name",' + _lsf_string(group) + ");"]
        for obj in objects:
            path = (Path(folder) / obj["stl"]).resolve()
            lines += ["stlimport(" + _lsf_string(path.as_posix()) + ",1e-6,1e-12,0);",
                      'set("name",' + _lsf_string("material_" + str(obj["material_id"])) + ");",
                      'set("material",' + _lsf_string(obj["fdtd_material"]) + ");"]
            lines += ['set("' + axis + '",' + format(v, ".17g") + ");" for axis, v in zip("xyz", obj["translation_m"])]
            lines += ["addtogroup(" + _lsf_string(group) + ");"]
        lines += ["unselectall;", "}", '# Inspect and then save from FDTD, for example save("imported.fsp");']
    (Path(folder) / "import_lumerical.lsf").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _versions():
    result = {"python": sys.version.split()[0], "converter": __version__}
    for name in ("numpy", "scikit-image", "fast-simplification", "trimesh", "scipy", "rtree"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def export_planar(args):
    import numpy as np
    from .planar import write_planar_mat, generate_planar_lsf
    folder = Path(args.folder).expanduser().resolve()
    path = folder / "conversion_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "vdata-standalone-conversion-v1" or manifest.get("status") != "complete":
        raise ValueError("Expected a completed standalone conversion manifest")
    objects = manifest.get("objects", [])
    if not objects:
        raise ValueError("This output has no surface meshes")
    mapping = _load_mapping(args.material_map)
    known_hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
    planned = []
    output_paths = set()
    # Preflight the entire batch before creating any geometry. A corrupt second
    # material or an unrelated target file must not leave a half-exported batch.
    for obj in objects:
        if obj.get("planar_mat"):
            mat = (folder / obj["planar_mat"]).resolve()
            if folder not in mat.parents or not mat.is_file() or file_hash(mat) != known_hashes.get(obj["planar_mat"]):
                raise ValueError("Existing MAT is missing, moved outside the output, or differs from its manifest")
            planned.append((obj, None, mat, obj["planar_mat"]))
        else:
            key = "compact_mesh" if obj.get("compact_mesh") else "indexed_mesh"
            src = (folder / obj[key]).resolve()
            if folder not in src.parents or not src.is_file() or file_hash(src) != known_hashes.get(obj[key]):
                raise ValueError("Indexed mesh is missing, moved outside the output, or differs from its manifest")
            relative = "meshes/material_{:04d}_planar.mat".format(obj["material_id"])
            mat = (folder / relative).resolve()
            if folder not in mat.parents:
                raise ValueError("MAT target is outside the output folder")
            if mat.exists() or mat.is_symlink():
                raise FileExistsError("Refusing to replace an unrecorded MAT: " + str(mat))
            planned.append((obj, key, src, relative))
        target = (folder / planned[-1][3]).resolve()
        if target in output_paths:
            raise ValueError("Duplicate MAT output target in conversion manifest")
        output_paths.add(target)
        mid, name = obj["material_id"], obj["source_name"]
        obj["fdtd_material"] = mapping.get("id:" + str(mid), mapping.get(str(mid), mapping.get(name, obj.get("fdtd_material"))))
        _lsf_string(name)
        if obj["fdtd_material"]:
            _lsf_string(obj["fdtd_material"])
        _lsf_string(target.as_posix())
    text_paths = [folder / "import_lumerical_planar.lsf", folder / "material_map.json", path]
    originals = {}
    for target in text_paths:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError("Generated text destination must be a regular local file")
        originals[target] = target.read_bytes() if target.exists() else None
    created = []
    text_started = False
    try:
        for obj, key, src, relative in planned:
            if key is None:
                continue
            if key == "compact_mesh":
                from .storage import read_mesh_archive
                vertices, faces, translation, _ = read_mesh_archive(src)
            else:
                with np.load(src, allow_pickle=False) as archive:
                    vertices, faces, translation = (archive[k] for k in ("vertices_local_m", "faces", "translation_m"))
            if file_hash(src) != known_hashes[obj[key]]:
                raise ValueError("Indexed mesh changed during export")
            if not np.array_equal(translation, np.asarray(obj["translation_m"], dtype=np.float64)):
                raise ValueError("Indexed mesh translation differs from the conversion manifest")
            target = (folder / relative).resolve()
            obj["planar_export"] = write_planar_mat(target, vertices, faces, translation)
            created.append((target, target.stat().st_ino))
            obj["planar_mat"] = relative
            del vertices, faces
        text_started = True
        generated = generate_planar_lsf(objects, folder)
        write_json(folder / "material_map.json", {"materials": {"id:" + str(o["material_id"]): o["fdtd_material"] for o in objects}})
        manifest["planar_export_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["files"] = [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
                             for p in sorted(folder.rglob("*")) if p.is_file() and p != path]
        write_json(path, manifest)
    except Exception:
        # Only files successfully created by this call are eligible for cleanup.
        # A concurrently replaced path has a different inode and is preserved.
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
    print("Generated offline MAT/LSF (CAD import not executed): " + str(generated))
    return 0


def convert(args):
    import numpy as np
    started = time.perf_counter()
    source = Path(args.input).expanduser().resolve()
    meta = inspect_vdata(source)
    lo, hi = _crop(meta, args.crop_um)
    if args.surface == "plane" and not meta["has_plane_equations"]:
        raise ValueError("This VDATA has no PlaneEquations; cannot reconstruct missing interface information")
    if args.surface != "fraction" and args.level != 0.5:
        raise ValueError("--level only applies to fraction mode; plane and vof use zero distance")
    if not math.isfinite(args.level) or not 0 < args.level < 1:
        raise ValueError("Isosurface level must be strictly between zero and one")
    if not math.isfinite(args.reduction) or not 0 <= args.reduction < 1:
        raise ValueError("Reduction is the fraction of faces to remove and must lie in [0,1)")
    if args.planar_min_faces < 4:
        raise ValueError("Planar minimum face count must be at least 4")
    if args.max_sampled_error_nm is not None and (not math.isfinite(args.max_sampled_error_nm) or args.max_sampled_error_nm < 0):
        raise ValueError("Sampled error limit must be finite and nonnegative")
    if args.max_sampled_error_nm is not None and args.reduction == 0:
        raise ValueError("A sampled QEM error limit requires a nonzero --reduction")
    mapping = _load_mapping(args.material_map)
    out = Path(args.output).expanduser().resolve() if args.output else source.parent / (source.stem + "_standalone_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Output directory must be new or empty: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "vdata-standalone-conversion-v1", "status": "running", "source": str(source),
        "source_sha256": meta["source_sha256"], "started_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _versions(), "coventor_required": False, "lumerical_required_for_conversion": False,
        "vendor_api_calls": [], "optical_simulation_run": False,
        "surface_mode": args.surface, "storage_mode": args.storage,
        "planar_optimization_requested": args.planar_optimize,
        "active_blocks_requested": args.active_blocks,
        "source_metadata": meta, "objects": [], "files": [], "stage_seconds": {},
        "limitations": [
            "Supported VoxelXML byte-record layout only; file extension alone does not identify a supported format.",
            "The selected reconstruction interpolates a cell-centred scalar field; the surface is an approximation.",
            "PlaneEquations are used for geometry only when surface_mode=plane; doping does not become an optical material model.",
            "Padding closes the crop with background; crop walls are artificial and corners can be rounded by interpolation.",
            "Geometric checks do not certify physical shape accuracy, self-intersection freedom or FDTD position accuracy.",
        ],
    }
    manifest_path = out / "conversion_manifest.json"
    write_json(manifest_path, manifest)
    cache = out / ".decode-cache.npy"
    raw = None
    planes = None
    region = None
    try:
        print("Decoding VoxelXML directly (no vendor SDK)...", flush=True)
        stage_started = time.perf_counter()
        source_crop = list(zip(lo.tolist(), hi.tolist()))
        raw, decoded = decode_vdata(source, cache, crop_indices=source_crop,
                                   max_raw_bytes=int(args.max_raw_gib * 1024 ** 3))
        if decoded["source_sha256"] != meta["source_sha256"]:
            raise RuntimeError("Source changed between inspection and decoding")
        manifest["stage_seconds"]["voxel_decode"] = time.perf_counter() - stage_started
        manifest["source_metadata"] = decoded
        if meta["has_plane_equations"] and (args.surface == "plane" or not args.skip_voxels):
            stage_started = time.perf_counter()
            from .planes import decode_plane_equations
            planes, plane_meta = decode_plane_equations(source, crop_indices=source_crop,
                max_raw_bytes=int(args.max_raw_gib * 1024 ** 3))
            if plane_meta["source_sha256"] != meta["source_sha256"]:
                raise RuntimeError("Source changed between material and PlaneEquations decoding")
            manifest["plane_metadata"] = plane_meta
            manifest["stage_seconds"]["plane_decode"] = time.perf_counter() - stage_started
        # Decoding already retained exactly this region. Keep source metadata
        # intact and give the mesher a separate local grid to avoid a second
        # crop or applying the origin offset twice.
        region = raw
        crop_origin = np.asarray(meta["origin_m"]) + lo * meta["voxel_size_m"]
        mesh_meta = {**decoded, **decoded["region_metadata"]}
        voxel_meta = {
            "schema": "voxel-material-records-v1", "axis_order": "XYZ", "storage_order": "C",
            "dimensions_xyz": [int(v) for v in hi - lo], "origin_m": crop_origin.tolist(),
            "origin_kind": "lower_corner", "voxel_size_m": meta["voxel_size_m"],
            "materials": meta["materials"], "record_channels": ["fraction_code", "material_1", "material_2"],
            "record_dtype": "uint8", "fraction_rule": "code 129: pure material_1=material_2; otherwise fraction_1=code/128, fraction_2=1-fraction_1",
            "source_sha256": meta["source_sha256"], "crop_source_indices": [lo.tolist(), hi.tolist()],
            "plane_equations_present_in_source": meta["has_plane_equations"],
            "plane_equations_in_this_array": False,
            "physical_accuracy": "Preserves the selected source byte records; not a certification of the physical process model.",
        }
        if not args.skip_voxels:
            stage_started = time.perf_counter()
            print("Saving lossless material/fraction records...", flush=True)
            np.savez_compressed(out / "voxels.npz", raw=region)
            write_json(out / "voxel_metadata.json", voxel_meta)
            if planes is not None:
                np.savez_compressed(out / "planes.npz", plane_equations=planes)
                write_json(out / "plane_metadata.json", {**plane_meta, "crop_origin_m": crop_origin.tolist(),
                    "crop_source_indices": [lo.tolist(), hi.tolist()], "crop_dimensions_xyz": (hi - lo).tolist(),
                    "archive": "planes.npz", "units_note": "n and d are kept as source bytes; geometric interpretation is an explicit surface-mode assumption"})
                voxel_meta["plane_equations_archive"] = "planes.npz"
                write_json(out / "voxel_metadata.json", voxel_meta)
            manifest["stage_seconds"]["voxel_archive"] = time.perf_counter() - stage_started
        materials = meta["materials"]
        selected = [int(v) for v in args.materials.split(",")] if args.materials else [int(m["id"]) for m in materials if not _background(m)]
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate material IDs in --materials")
        palette = {int(m["id"]): m for m in materials}
        if any(mid not in palette for mid in selected):
            raise ValueError("Requested material ID is absent from the material table")
        manifest["crop"] = voxel_meta
        volumes = {}
        active = []
        # Chunk across X so the volume audit never allocates another full float volume.
        stage_started = time.perf_counter()
        for mid in palette:
            total = 0.0
            for begin in range(0, region.shape[0], 16):
                block = np.ascontiguousarray(region[begin:begin + 16])
                total += float(fraction_for_material(block, mid).sum(dtype=np.float64))
            volumes[str(mid)] = total * meta["voxel_size_m"] ** 3
            if mid in selected and total > 0:
                active.append(mid)
        manifest["material_volumes_m3"] = volumes
        manifest["stage_seconds"]["material_volume_audit"] = time.perf_counter() - stage_started
        if args.reduction and len(active) != 1:
            raise ValueError("QEM currently requires exactly one exported solid; shared multi-material interfaces are not reduced independently")
        if args.planar_optimize and len(active) != 1:
            raise ValueError("Planar optimization requires exactly one exported solid; shared multi-material interfaces must be optimized together")
        if args.command == "convert":
            from .meshing import mesh_material, write_binary_stl
            (out / "meshes").mkdir()
            for mid in active:
                print("Reconstructing material {} using {} field...".format(mid, args.surface), flush=True)
                mesh_started = time.perf_counter()
                vertices, faces, report = mesh_material(raw, mesh_meta, mid, level=args.level,
                    step_size=1, surface=args.surface, planes=planes, active_blocks=args.active_blocks)
                report["source_crop_indices_xyz"] = [list(pair) for pair in source_crop]
                report["grid_origin_m"] = report["source_origin_m"]
                report["grid_crop_indices_xyz"] = report["crop_indices_xyz"]
                report["source_origin_m"] = list(meta["origin_m"])
                report["crop_indices_xyz"] = [list(pair) for pair in source_crop]
                if decoded["crop_applied"]:
                    report["warnings"].append("The streamed source crop replaces exterior material with background and creates artificial closed cut surfaces.")
                report["extraction_seconds"] = time.perf_counter() - mesh_started
                if len(faces) == 0:
                    retained = "source VDATA" if args.skip_voxels else "voxels.npz"
                    manifest.setdefault("warnings", []).append("Material {} has positive volume but no level-{} surface; records remain in {}".format(mid, args.level, retained))
                    continue
                if args.planar_optimize:
                    from .planar_optimize import optimize_planar_mesh
                    print("Retriangulating eligible exact planar patches with locked boundaries...", flush=True)
                    planar_started = time.perf_counter()
                    vertices, faces, planar_report = optimize_planar_mesh(vertices, faces,
                        min_faces=args.planar_min_faces)
                    planar_report["elapsed_seconds"] = time.perf_counter() - planar_started
                    report["planar_optimization"] = planar_report
                if args.reduction:
                    from .meshing import simplify_mesh
                    print("Reducing faces and measuring additional sampled distance...", flush=True)
                    vertices, faces, reduction_report = simplify_mesh(vertices, faces, args.reduction, max_sampled_error_nm=args.max_sampled_error_nm)
                    report["qem"] = reduction_report
                relative_stl = npz_path = compact_path = None
                stl_report = compact_report = None
                export_started = time.perf_counter()
                if args.storage in {"standard", "both"}:
                    relative_stl = "meshes/material_{:04d}.stl".format(mid)
                    stl_report = write_binary_stl(out / relative_stl, vertices, faces, coordinate_unit="um")
                    npz_path = "meshes/material_{:04d}.npz".format(mid)
                    np.savez_compressed(out / npz_path, vertices_local_m=vertices, faces=faces,
                                        translation_m=np.asarray(report["translation_m"], dtype=np.float64))
                if args.storage in {"compact", "both"}:
                    from .storage import write_mesh_archive
                    compact_path = "meshes/material_{:04d}.vmesh".format(mid)
                    compact_report = write_mesh_archive(out / compact_path, vertices, faces,
                        np.asarray(report["translation_m"], dtype=np.float64),
                        metadata={"material_id": mid, "source_sha256": meta["source_sha256"],
                                  "units": "metres", "vertex_frame": "local", "reconstruction": report["algorithm"]})
                name = palette[mid]["name"]
                optical = mapping.get("id:" + str(mid), mapping.get(str(mid), mapping.get(name, _default_optical(name))))
                obj = {"material_id": mid, "source_name": name, "fdtd_material": optical,
                       "stl": relative_stl, "indexed_mesh": npz_path, "compact_mesh": compact_path,
                       "compact_storage": compact_report, "stl_coordinate_unit": "um",
                       "mesh_export_seconds": time.perf_counter() - export_started,
                       "exported_vertex_count": len(vertices), "exported_face_count": len(faces),
                       "translation_m": report["translation_m"], "reconstruction": report, "stl_export": stl_report}
                manifest["objects"].append(obj)
            if len(active) > 1:
                manifest["limitations"].append("Independent material isosurfaces are not a conforming multi-material partition; gaps/overlaps near junctions require verification.")
            if manifest["objects"]:
                write_json(out / "material_map.json", {"materials": {"id:" + str(o["material_id"]): o["fdtd_material"] for o in manifest["objects"]},
                           "note": "Exact local FDTD library names. Defaults are assumptions. Null blocks execution of the generated script."})
                generate_lsf(manifest["objects"], out)
            else:
                manifest.setdefault("warnings", []).append("No solid surface was exported")
        manifest["source_unchanged"] = file_hash(source) == meta["source_sha256"]
        if not manifest["source_unchanged"]:
            raise RuntimeError("Source file changed while converting")
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = time.perf_counter() - started
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        region = None
        planes = None
        if raw is not None:
            if getattr(raw, "_mmap", None) is not None:
                raw._mmap.close()
            raw = None
        gc.collect()
        if cache.exists():
            cache.unlink()
        manifest["files"] = [{"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size,
                              "sha256": file_hash(p)} for p in sorted(out.rglob("*")) if p.is_file() and p != manifest_path]
        write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "output": str(out), "materials_exported": len(manifest["objects"]),
                      "elapsed_seconds": manifest["elapsed_seconds"], "source_unchanged": manifest["source_unchanged"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Independent VoxelXML conversion with ordinary Python; no Coventor or Lumerical API")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Read metadata using only the Python standard library")
    inspect.add_argument("input")
    script = sub.add_parser("lsf", help="Regenerate a text import script from an existing standalone output")
    script.add_argument("folder")
    script.add_argument("--material-map")
    expand = sub.add_parser("export-stl", help="Expand lossless compact meshes to STL and regenerate the import script")
    expand.add_argument("folder")
    expand.add_argument("--material-map")
    planar = sub.add_parser("export-planar", help="Export indexed MAT and addplanarsolid script, preserving all triangles")
    planar.add_argument("folder")
    planar.add_argument("--material-map")
    optimize = sub.add_parser("optimize-mesh", help="Retriangulate eligible exact planar patches in one existing closed mesh; no VDATA decoding")
    optimize.add_argument("input", help="Existing .vmesh or indexed .npz mesh")
    optimize.add_argument("--output", required=True, help="New .vmesh path; never overwrite")
    optimize.add_argument("--min-faces", type=int, default=128)
    pack = sub.add_parser("pack-mesh", help="Losslessly repack an existing indexed NPZ mesh, preserving every vertex and face")
    pack.add_argument("input")
    pack.add_argument("--output", required=True, help="New .vmesh archive path")
    for name in ("decode", "convert"):
        p = sub.add_parser(name, help="Export material/fraction records" if name == "decode" else "Export records and reconstructed STL")
        p.add_argument("input")
        p.add_argument("--output")
        p.add_argument("--crop-um", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
        p.add_argument("--max-raw-gib", type=float, default=8.0, help="Decoded byte limit; default 8 GiB")
        p.add_argument("--materials", help="Comma-separated source material IDs; default excludes Air/Void")
        p.add_argument("--level", type=float, default=0.5, help="Fraction isosurface threshold; default 0.5")
        p.add_argument("--reduction", type=float, default=0.0, help="QEM fraction of faces to remove, e.g. 0.45")
        p.add_argument("--max-sampled-error-nm", type=float, help="Reject QEM when sampled additional distance exceeds this value")
        p.add_argument("--material-map", help="JSON mapping from id:NUMBER or full source name to optical library name")
        p.add_argument("--skip-voxels", action="store_true", help="Skip the lossless voxel archive, export geometry only")
        p.add_argument("--storage", choices=("standard", "compact", "both"), default="standard",
                       help="standard: STL+NPZ; compact: lossless .vmesh only; both: all formats. No face reduction implied")
        p.add_argument("--surface", choices=("fraction", "vof", "plane"), default="fraction",
                       help="fraction: historical 0.5 isosurface; vof: experimental PLIC-guided field; plane: experimental saved planes in local voxel-corner convention")
        if name == "convert":
            p.add_argument("--planar-optimize", action="store_true",
                           help="Retriangulate eligible exact axis-aligned convex planar patches; lock all original patch boundary vertices")
            p.add_argument("--planar-min-faces", type=int, default=128,
                           help="Minimum patch face count for planar optimization; default 128")
            p.add_argument("--active-blocks", action="store_true",
                           help="Use experimental conservative block min/max mask for the same Lewiner extraction")
        else:
            p.set_defaults(planar_optimize=False, planar_min_faces=128, active_blocks=False)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_vdata(args.input), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "optimize-mesh":
            import numpy as np
            from .planar_optimize import optimize_planar_mesh
            from .storage import read_mesh_archive, write_mesh_archive
            src = Path(args.input).expanduser().resolve()
            dst = Path(args.output).expanduser().resolve()
            if dst.exists():
                raise ValueError("Optimization destination already exists")
            source_hash = file_hash(src)
            if src.suffix.lower() == ".vmesh":
                vertices, faces, translation, _ = read_mesh_archive(src)
            else:
                with np.load(src, allow_pickle=False) as archive:
                    vertices, faces, translation = (archive[k] for k in
                        ("vertices_local_m", "faces", "translation_m"))
            vertices, faces, optimization = optimize_planar_mesh(vertices, faces, min_faces=args.min_faces)
            if file_hash(src) != source_hash:
                raise ValueError("Source mesh changed during optimization")
            written = write_mesh_archive(dst, vertices, faces, translation,
                metadata={"source_mesh_sha256": source_hash, "planar_optimization": optimization,
                    "scope": "One mesh only; not a coordinated multi-material interface optimization"})
            print(json.dumps({"optimization": optimization, "archive": written}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "pack-mesh":
            import numpy as np
            from .storage import write_mesh_archive
            with np.load(args.input, allow_pickle=False) as src:
                report = write_mesh_archive(args.output, src["vertices_local_m"], src["faces"],
                                            src["translation_m"], metadata={"source_npz_sha256": file_hash(args.input)})
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "export-planar":
            return export_planar(args)
        if args.command in {"lsf", "export-stl"}:
            folder = Path(args.folder).expanduser().resolve()
            path = folder / "conversion_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            if manifest.get("schema") != "vdata-standalone-conversion-v1" or manifest.get("status") != "complete":
                raise ValueError("Expected a completed standalone conversion manifest")
            objects = manifest.get("objects", [])
            if not objects:
                raise ValueError("This output has no surface meshes")
            mapping = _load_mapping(args.material_map)
            known_hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
            planned = []
            output_paths = set()
            # Validate all sources and destinations before starting the batch.
            for obj in objects:
                if args.command == "export-stl" and not obj.get("stl"):
                    archive = (folder / obj["compact_mesh"]).resolve()
                    if folder not in archive.parents or not archive.is_file():
                        raise ValueError("Compact mesh is missing or outside the output folder")
                    expected = known_hashes.get(obj["compact_mesh"])
                    if expected is None or file_hash(archive) != expected:
                        raise ValueError("Compact archive hash differs from the conversion manifest")
                    relative = "meshes/material_{:04d}.stl".format(obj["material_id"])
                    target = (folder / relative).resolve()
                    if folder not in target.parents:
                        raise ValueError("STL target is outside the output folder")
                    if target.exists() or target.is_symlink():
                        raise FileExistsError("Refusing to replace an unrecorded STL: " + str(target))
                    planned.append((obj, archive, relative))
                if obj.get("stl"):
                    stl = (folder / obj["stl"]).resolve()
                    if folder not in stl.parents or not stl.is_file():
                        raise ValueError("Mesh is missing or outside the output folder")
                    if file_hash(stl) != known_hashes.get(obj["stl"]):
                        raise ValueError("Existing STL hash differs from the conversion manifest")
                    target = stl
                elif args.command != "export-stl":
                    target = None
                if target is not None:
                    if target in output_paths:
                        raise ValueError("Duplicate STL output target in conversion manifest")
                    output_paths.add(target)
                    _lsf_string(target.as_posix())
                mid, name = obj["material_id"], obj["source_name"]
                obj["fdtd_material"] = mapping.get("id:" + str(mid), mapping.get(str(mid), mapping.get(name, obj.get("fdtd_material"))))
                _lsf_string(name)
                if obj["fdtd_material"]:
                    _lsf_string(obj["fdtd_material"])
            text_paths = [folder / "import_lumerical.lsf", folder / "material_map.json", path]
            originals = {}
            for target in text_paths:
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError("Generated text destination must be a regular local file")
                originals[target] = target.read_bytes() if target.exists() else None
            created = []
            text_started = False
            try:
                for obj, archive, relative in planned:
                    from .storage import read_mesh_archive
                    from .meshing import write_binary_stl
                    import numpy as np
                    vertices, faces, translation, _ = read_mesh_archive(archive)
                    if file_hash(archive) != known_hashes[obj["compact_mesh"]]:
                        raise ValueError("Compact archive changed during export")
                    if not np.array_equal(translation, np.asarray(obj["translation_m"], dtype=np.float64)):
                        raise ValueError("Compact archive translation differs from the conversion manifest")
                    target = (folder / relative).resolve()
                    obj["stl_export"] = write_binary_stl(target, vertices, faces, coordinate_unit="um")
                    created.append((target, target.stat().st_ino))
                    obj["stl"] = relative
                    del vertices, faces
                text_started = True
                generate_lsf(objects, folder)
                write_json(folder / "material_map.json", {"materials": {"id:" + str(o["material_id"]): o["fdtd_material"] for o in objects}})
                manifest["script_regenerated_utc"] = datetime.now(timezone.utc).isoformat()
                manifest["files"] = [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size,
                                      "sha256": file_hash(p)} for p in sorted(folder.rglob("*")) if p.is_file() and p != path]
                write_json(path, manifest)
            except Exception:
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
            print("Generated text script: " + str(folder / "import_lumerical.lsf"))
            return 0
        if not math.isfinite(args.max_raw_gib) or args.max_raw_gib <= 0:
            raise ValueError("--max-raw-gib must be finite and positive")
        if args.command == "decode" and (args.skip_voxels or args.reduction):
            raise ValueError("decode requires voxel output and does not perform QEM")
        return convert(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print("Conversion failed: " + str(exc), file=sys.stderr)
        return 2
