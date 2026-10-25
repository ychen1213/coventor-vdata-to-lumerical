"""Create and optionally verify a material-aware Lumerical STL import.

The source is a conversion manifest, never a saved process project. STL
coordinates must be in micrometres.  This module deliberately performs no
optical solve and does not connect to a user's existing CAD session.

The public entry point uses only Python's standard library (Python >= 3.6).
Verification is delegated to Lumerical's own Python and a private hidden CAD.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import traceback


GROUP_NAME = "Coventor_VDATA_import"
SI_MATERIAL = "Si (Silicon) - Palik"
SIO2_MATERIAL = "SiO2 (Glass) - Palik"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def lsf_string(value):
    """One LSF string literal; never interpolate unchecked script source."""
    value = str(value)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("LSF strings cannot contain control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _resolved_path(value, relative_to):
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def read_manifest(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    materials = data.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("Manifest must contain a nonempty materials list")
    bounds = data.get("bounds_um")
    if (not isinstance(bounds, list) or len(bounds) != 2 or
            any(not isinstance(row, list) or len(row) != 3 for row in bounds)):
        raise ValueError("bounds_um must be [[xmin,ymin,zmin],[xmax,ymax,zmax]]")
    bounds = [[float(value) for value in row] for row in bounds]
    if (not all(math.isfinite(value) for row in bounds for value in row) or
            any(bounds[1][axis] <= bounds[0][axis] for axis in range(3))):
        raise ValueError("bounds_um must contain finite, increasing bounds")
    normalized = []
    for index, material in enumerate(materials):
        name = material.get("source_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each material needs a nonempty source_name")
        lsf_string(name)
        stl = _resolved_path(material["stl_path"], path.parent)
        if not stl.is_file():
            raise ValueError("STL is missing: " + str(stl))
        lsf_string(stl.as_posix())
        item = dict(material)
        translation = material.get("translation_um", data.get("translation_um", [0, 0, 0]))
        if not isinstance(translation, (tuple, list)) or len(translation) != 3:
            raise ValueError("translation_um must contain three coordinates")
        translation = [float(value) for value in translation]
        if not all(math.isfinite(value) for value in translation):
            raise ValueError("translation_um must contain finite coordinates")
        item.update(source_name=name, stl_path=str(stl),
                    source_ids=material.get("source_ids", []),
                    translation_um=translation,
                    object_name="VDATA_material_%04d" % (index + 1))
        normalized.append(item)
    data = dict(data)
    data.update(materials=normalized, bounds_um=bounds, manifest_path=str(path))
    return data


def read_material_map(path):
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Material mapping must be a JSON object")
    mapping = data.get("materials", data)
    if not isinstance(mapping, dict):
        raise ValueError("The materials mapping must be a JSON object")
    for source, target in mapping.items():
        if not isinstance(source, str):
            raise ValueError("Material mapping keys must be source names")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ValueError("Material mapping values must be names or null")
        if target is not None:
            lsf_string(target)
    return mapping


def map_materials(materials, overrides):
    """Conservative chemical-name aliases; every automatic choice is reported."""
    silicon_aliases = {"sixtal", "sicrystalline", "crystallinesilicon", "siliconcrystalline"}
    oxide_aliases = {"sio2", "sio2glass", "sio2thermal", "sio2deposited",
                     "sio2pecvd", "sio2hdp", "sio2teos", "silicondioxide"}
    result = []
    for material in materials:
        name = material["source_name"]
        item = dict(material)
        id_keys = _source_id_keys(item)
        selected_id_keys = [key for key in id_keys if key in overrides]
        if selected_id_keys:
            targets = [overrides[key] for key in selected_id_keys]
            if any(target != targets[0] for target in targets) or len(selected_id_keys) != len(id_keys):
                raise ValueError("A grouped STL with multiple source IDs needs identical explicit mappings for all of its IDs: " + ", ".join(id_keys))
            target = targets[0]
            item.update(fdtd_material=target, mapping_source="user_id",
                        mapping_key=selected_id_keys[0] if len(selected_id_keys) == 1 else selected_id_keys,
                        mapping_assumption=None,
                        mapping_status="mapped" if target else "needs_material_mapping")
        elif name in overrides:
            target = overrides[name]
            item.update(fdtd_material=target, mapping_source="user",
                        mapping_key=name,
                        mapping_assumption=None,
                        mapping_status="mapped" if target else "needs_material_mapping")
        else:
            leaf_name = name.replace("\\", "/").rsplit("/", 1)[-1]
            key = re.sub(r"[\s_()\-]", "", leaf_name).lower()
            if key in silicon_aliases:
                item.update(fdtd_material=SI_MATERIAL, mapping_source="built_in_alias",
                            mapping_status="mapped",
                            mapping_assumption="Crystalline Si is mapped to the installed Palik Si optical model; process-specific optical constants are not inferred.")
            elif key in oxide_aliases:
                item.update(fdtd_material=SIO2_MATERIAL, mapping_source="built_in_alias",
                            mapping_status="mapped",
                            mapping_assumption="SiO2 is mapped to Palik glass; deposition, density and process-specific optical differences are not inferred.")
            else:
                item.update(fdtd_material=None, mapping_source=None,
                            mapping_assumption=None, mapping_status="needs_material_mapping")
        result.append(item)
    return result


def _source_id_keys(material):
    keys = []
    for identifier in material.get("source_ids", []):
        key = "id:" + str(identifier)
        if key not in keys:
            keys.append(key)
    return keys


def make_mapping_template(materials):
    """A duplicate source name is a label, not the material's identity."""
    name_counts = {}
    for item in materials:
        name_counts[item["source_name"]] = name_counts.get(item["source_name"], 0) + 1
    mappings = {}
    source_names = {}
    for item in materials:
        name = item["source_name"]
        keys = _source_id_keys(item) if name_counts[name] > 1 else [name]
        if not keys:
            raise ValueError("Repeated source names require source_ids for an unambiguous mapping template: " + name)
        for key in keys:
            if key in mappings and mappings[key] != item["fdtd_material"]:
                raise ValueError("Material mapping key has conflicting targets: " + key)
            mappings[key] = item["fdtd_material"]
            source_names[key] = name
    return dict(schema_version=1,
        instructions="Map each key to an exact FDTD material-library name. id:<MaterialID> overrides a full source name. Repeated source names use ID keys; source_names_by_key preserves the original labels. Null blocks import; automatic Palik aliases are optical-model assumptions.",
        materials=mappings, source_names_by_key=source_names)


def generate_lsf(materials):
    lines = [
        "# Coventor VDATA surface import. STL numeric coordinates are micrometres.",
        "# Run in an empty FDTD layout. No optical source or time-domain solve is added.",
        "# One planar solid per source material; optical dispersion remains in the material library.",
    ]
    missing = [item["source_name"] for item in materials if not item["fdtd_material"]]
    if missing:
        # An inert script is safer than importing unknown materials with CAD defaults.
        lines += ["# IMPORT BLOCKED: edit material_map.template.json and regenerate this script.",
                  "?" + lsf_string("needs_material_mapping: " + ", ".join(missing)) + ";"]
        return "\n".join(lines) + "\n"
    lines += ["switchtolayout;", "vdata_bridge_import_ready=1;",
              "if (getnamednumber(" + lsf_string(GROUP_NAME) + ")>0) {",
              "?" + lsf_string("Import blocked: the destination group already exists. Use an empty project.") + ";",
              "vdata_bridge_import_ready=0;", "}"]
    for optical_name in sorted(set(item["fdtd_material"] for item in materials)):
        lines += ["if (materialexists(" + lsf_string(optical_name) + ")==0) {",
                  "?" + lsf_string("needs_material_mapping: this FDTD library has no " + optical_name) + ";",
                  "vdata_bridge_import_ready=0;", "}"]
    lines += ["if (vdata_bridge_import_ready) {", "addstructuregroup;", 'set("name",' + lsf_string(GROUP_NAME) + ");"]
    for item in materials:
        lines += ["stlimport(" + lsf_string(Path(item["stl_path"]).as_posix()) + ",1e-6,1e-12,0);",
                  'set("name",' + lsf_string(item["object_name"]) + ");",
                  'set("material",' + lsf_string(item["fdtd_material"]) + ");",
                  'set("x",%.17g);' % (item.get("translation_um", [0, 0, 0])[0] * 1e-6),
                  'set("y",%.17g);' % (item.get("translation_um", [0, 0, 0])[1] * 1e-6),
                  'set("z",%.17g);' % (item.get("translation_um", [0, 0, 0])[2] * 1e-6),
                  "addtogroup(" + lsf_string(GROUP_NAME) + ");"]
    lines += ["unselectall;", "}", '# Save after inspecting geometry, e.g. save("my_import.fsp");']
    return "\n".join(lines) + "\n"


def discover_lumerical(explicit=None):
    from common import (installation_path, installation_search_roots,
                        installation_version_key, safe_installation_glob,
                        unique_installation_paths)
    if explicit:
        candidates = [installation_path(explicit)]
    else:
        candidates = []
        for variable in ("LUMERICAL_ROOT", "LUMERICAL_INSTALL_ROOT"):
            if os.environ.get(variable):
                candidates.append(installation_path(os.environ[variable]))
        # Ansys installers commonly publish AWP_ROOT### for their version.
        for variable in sorted(os.environ,key=installation_version_key,reverse=True):
            if re.match(r"^AWP_ROOT\d+$",variable,re.I) and os.environ[variable]:
                candidates.append(installation_path(os.environ[variable]) / "Lumerical")
        found = []
        for base in installation_search_roots():
            for pattern in ("Ansys Inc/v*/Lumerical", "Ansys/v*/Lumerical",
                            "Lumerical/v*", "Lumerical", "v*/Lumerical"):
                found.extend(safe_installation_glob(base,pattern))
        candidates.extend(sorted(found,key=installation_version_key,reverse=True))
    for candidate in unique_installation_paths(candidates):
        candidate = candidate.resolve()
        if not (candidate / "api/python/lumapi.py").is_file():
            continue
        python_candidates = sorted(safe_installation_glob(candidate,"python-*/python.exe"),
                                   key=installation_version_key,reverse=True)
        python_candidates += [candidate / "python/python.exe", candidate / "python/bin/python3"]
        for executable in python_candidates:
            if executable.is_file():
                return candidate, executable.resolve()
    suffix = " at " + str(explicit) if explicit else ""
    raise RuntimeError("No Lumerical installation with a bundled Python was found"+suffix+
                       "; use --lumerical-root or set LUMERICAL_ROOT")


def _read_stl_numpy(path, np):
    """Read binary or ASCII STL without losing ASCII precision before metrology."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(84)
    count = struct.unpack("<I", header[80:84])[0] if len(header) == 84 else 0
    if count and size == 84 + 50 * count:
        dtype = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attribute", "<u2")])
        records = np.memmap(str(path), mode="r", dtype=dtype, offset=84, shape=(count,))
        triangles = records["v"].astype(float)
    else:
        points = []
        with path.open("r", encoding="ascii") as handle:
            for line in handle:
                fields = line.split()
                if fields and fields[0].lower() == "vertex":
                    if len(fields) != 4:
                        raise ValueError("Malformed ASCII STL vertex")
                    points.append([float(value) for value in fields[1:]])
        if not points or len(points) % 3:
            raise ValueError("STL contains no complete triangles")
        triangles = np.asarray(points, dtype=float).reshape(-1, 3, 3)
    if not np.isfinite(triangles).all():
        raise ValueError("STL has nonfinite coordinates")
    return triangles


def _jsonable(value, np):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item, np) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item, np) for item in value]
    return value


def _facet_digest(value, np):
    blob = json.dumps(_jsonable(value, np), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_worker(request_path):
    """Only called by the bundled Python; writes a result even on CAD failure."""
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = Path(request["output"])
    result_path = output / "_fdtd_verify_result.json"
    result = dict(fdtd_status="verification_failed", time_domain_solver_run=False,
                  private_hidden_session=True, save_reopen_verified=False, materials=[],
                  request_id=request["request_id"])
    started = time.monotonic()
    try:
        import numpy as np
        from scipy.spatial import cKDTree
        sys.path.insert(0, str(Path(request["lumerical_root"]) / "api/python"))
        import lumapi
        target = output / "imported.fsp"
        geometries = {}
        with lumapi.FDTD(hide=True) as session:
            missing = [item["fdtd_material"] for item in request["materials"]
                       if not bool(session.materialexists(item["fdtd_material"]))]
            if missing:
                result.update(fdtd_status="needs_material_mapping", unavailable_optical_materials=missing)
                raise RuntimeError("Optical materials absent from this FDTD installation: " + ", ".join(missing))
            session.eval(Path(request["script"]).read_text(encoding="utf-8"))
            for item in request["materials"]:
                name = GROUP_NAME + "::" + item["object_name"]
                vertices = np.asarray(session.getnamed(name, "vertices"), dtype=float)
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    raise RuntimeError("Invalid imported vertices for " + item["source_name"])
                actual_material = str(session.getnamed(name, "material"))
                if actual_material != item["fdtd_material"]:
                    raise RuntimeError("Imported material differs from requested material")
                facets_digest = _facet_digest(session.getnamed(name, "facets"), np)
                translation_m = np.asarray([float(session.getnamed(name, axis)) for axis in ("x", "y", "z")])
                requested_translation_m = np.asarray(item.get("translation_um", [0, 0, 0])) * 1e-6
                if not np.array_equal(translation_m, requested_translation_m):
                    raise RuntimeError("Imported object translation differs from the manifest")
                triangles_um = _read_stl_numpy(item["stl_path"], np)
                source_um = triangles_um.reshape(-1, 3)
                source_m = source_um * 1e-6
                # STL import can quantize ASCII coordinates to float32.  Report
                # that explicitly instead of silently counting it as exact input.
                quantized_m = source_um.astype(np.float32).astype(float) * 1e-6
                tree = cKDTree(vertices)
                raw_error = float(tree.query(source_m)[0].max())
                quantized_error = float(tree.query(quantized_m)[0].max())
                reverse_error = float(cKDTree(quantized_m).query(vertices)[0].max())
                tolerance_m = 1.01e-12 + float(np.abs(quantized_m).max()) * 8 * np.finfo(float).eps
                if max(quantized_error, reverse_error) > tolerance_m:
                    raise RuntimeError("STL import changed coordinates beyond float32 representation and 1 pm merge tolerance")
                source_bounds = np.stack((source_um.min(axis=0), source_um.max(axis=0)))
                imported_bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
                entry = dict(source_name=item["source_name"], source_ids=item.get("source_ids", []),
                             object_name=name, fdtd_material=actual_material,
                             triangles=int(triangles_um.shape[0]), imported_vertices=int(vertices.shape[0]),
                             source_stl_bounds_um=source_bounds.tolist(), imported_bounds_m=imported_bounds.tolist(),
                             translation_um=item.get("translation_um", [0, 0, 0]),
                             imported_translation_m=translation_m.tolist(),
                             source_world_bounds_um=(source_bounds + np.asarray(item.get("translation_um", [0, 0, 0]))).tolist(),
                             imported_world_bounds_m=(imported_bounds + translation_m).tolist(),
                             max_source_to_import_error_pm=raw_error * 1e12,
                             max_error_after_stl_float32_pm=quantized_error * 1e12,
                             max_import_to_source_after_stl_float32_pm=reverse_error * 1e12,
                             import_merge_tolerance_pm=1.0, vertex_coordinates_verified=True,
                             facets_sha256=facets_digest, save_reopen_verified=False)
                result["materials"].append(entry)
                geometries[name] = (vertices, actual_material, facets_digest, translation_m)
            world_min = np.min(np.array([item[0].min(axis=0) + item[3] for item in geometries.values()]), axis=0)
            world_max = np.max(np.array([item[0].max(axis=0) + item[3] for item in geometries.values()]), axis=0)
            margin = max(float((world_max - world_min).max()) * .05, 1e-9)
            view_extent = np.column_stack((world_min - margin, world_max + margin)).reshape(6, 1)
            session.setview("extent", view_extent)
            # No FDTD region is needed for a geometry-only FSP.  This avoids
            # imposing a domain, boundary condition or mesh on arbitrary input.
            session.save(str(target))
            session.load(str(target))
            for entry in result["materials"]:
                vertices, material, facets_digest, translation_m = geometries[entry["object_name"]]
                restored = np.asarray(session.getnamed(entry["object_name"], "vertices"), dtype=float)
                reopen_tolerance_m = max(1e-18, float(max(np.abs(vertices).max(), np.abs(translation_m).max())) * 8 * np.finfo(float).eps)
                if restored.shape != vertices.shape:
                    raise RuntimeError("Saved/reopened vertex array shape changed")
                reopen_displacement_m = float(np.abs(vertices - restored).max())
                entry.update(save_reopen_max_vertex_coordinate_change_pm=reopen_displacement_m * 1e12,
                             save_reopen_vertex_tolerance_pm=reopen_tolerance_m * 1e12,
                             save_reopen_vertices_bitwise_equal=bool(np.array_equal(vertices, restored)))
                if reopen_displacement_m > reopen_tolerance_m:
                    raise RuntimeError("Saved/reopened vertex coordinates changed beyond floating-point tolerance")
                if str(session.getnamed(entry["object_name"], "material")) != material:
                    raise RuntimeError("Saved/reopened optical material changed")
                if _facet_digest(session.getnamed(entry["object_name"], "facets"), np) != facets_digest:
                    raise RuntimeError("Saved/reopened facets changed")
                restored_translation = np.asarray([float(session.getnamed(entry["object_name"], axis)) for axis in ("x", "y", "z")])
                if not np.array_equal(translation_m, restored_translation):
                    raise RuntimeError("Saved/reopened object translation changed")
                entry["save_reopen_verified"] = True
            result.update(fdtd_status="verified", save_reopen_verified=True,
                          imported_fsp=str(target), fsp_bytes=target.stat().st_size,
                          vertices_bounds_m=[world_min.tolist(), world_max.tolist()])
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - started
    atomic_json(result_path, result)
    print(json.dumps({"fdtd_status": result["fdtd_status"], "report": str(result_path)}))
    return 0 if result["fdtd_status"] == "verified" else 1


def build(manifest_path, output, material_map=None, lumerical_root=None, verify=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(manifest_path)
    mappings = map_materials(manifest["materials"], read_material_map(material_map))
    unknown = [item["source_name"] for item in mappings if not item["fdtd_material"]]
    template_path = output / "material_map.template.json"
    atomic_json(template_path, make_mapping_template(mappings))
    script = output / "import_geometry.lsf"
    script.write_text(generate_lsf(mappings), encoding="utf-8")
    report = dict(schema_version=1, input_manifest=manifest["manifest_path"],
                  input_vdata=manifest.get("input_vdata"), source_sha256=manifest.get("source_sha256"),
                  bounds_um=manifest["bounds_um"], stl_coordinate_unit="um", fdtd_coordinate_unit="m",
                  fdtd_status="needs_material_mapping" if unknown else "ready_to_import",
                  unknown_materials=unknown, material_mappings=mappings,
                  import_script=str(script), material_map_template=str(template_path),
                  verification_requested=bool(verify), time_domain_solver_run=False,
                  save_reopen_verified=False)
    if verify and not unknown:
        try:
            install, python = discover_lumerical(lumerical_root)
            report.update(lumerical_root=str(install), verification_python=str(python))
            request_path = output / "_fdtd_verify_request.json"
            request_id = "%d-%d" % (os.getpid(), time.time() * 1000000)
            atomic_json(request_path, dict(output=str(output), script=str(script), materials=mappings,
                                          lumerical_root=str(install), request_id=request_id))
            completed = subprocess.run([str(python), str(Path(__file__).resolve()), "--verify-worker", str(request_path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                       encoding="utf-8", errors="replace")
            log_path = output / "fdtd_verify.log"
            log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
            result_path = output / "_fdtd_verify_result.json"
            if not result_path.is_file():
                raise RuntimeError("Lumerical verifier did not return a result; inspect " + str(log_path))
            verification = json.loads(result_path.read_text(encoding="utf-8"))
            if verification.get("request_id") != request_id:
                raise RuntimeError("The verifier result belongs to an older run; inspect " + str(log_path))
            report.update(fdtd_status=verification["fdtd_status"],
                          save_reopen_verified=verification["save_reopen_verified"],
                          verification=verification, verification_log=str(log_path),
                          verification_exit_code=completed.returncode)
        except Exception as exc:
            report.update(fdtd_status="verification_failed", error=str(exc), traceback=traceback.format_exc())
    atomic_json(output / "fdtd_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="conversion_manifest.json")
    parser.add_argument("--output", help="Destination directory")
    parser.add_argument("--material-map", help="Optional JSON source-name to optical-material mapping")
    parser.add_argument("--lumerical-root", help="Lumerical installation directory")
    parser.add_argument("--verify", action="store_true", help="Verify in a private hidden FDTD session")
    parser.add_argument("--verify-worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify_worker:
        return verify_worker(args.verify_worker)
    if not args.manifest or not args.output:
        parser.error("--manifest and --output are required")
    try:
        report = build(args.manifest, args.output, args.material_map, args.lumerical_root, args.verify)
    except Exception as exc:
        print(json.dumps({"error": "FDTD bridge error: " + str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"fdtd_status": report["fdtd_status"],
                      "report": str(Path(args.output).resolve() / "fdtd_report.json"),
                      "unknown_materials": report["unknown_materials"]}))
    if args.verify and report["fdtd_status"] != "verified":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
