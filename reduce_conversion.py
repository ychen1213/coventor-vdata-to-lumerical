"""Optional reduction of an existing VDATA conversion, never a VDATA decoder.

Python 3.10+. Reduction 0 performs a byte-exact STL copy without QEM dependencies.
Nonzero reduction currently accepts exactly one solid-material entry: independent
multi-material reduction could create gaps/overlaps at shared interfaces.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import sha256
import fdtd_bridge


ALGORITHM_VERSION = "qem-dense-surface-v1"


def _resolve(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def input_state(manifest_path, manifest):
    materials = manifest["materials"]
    state = {"manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
             "materials": [{"source_name": item["source_name"], "source_ids": item.get("source_ids", []),
                            "stl_path": item["stl_path"], "stl_sha256": sha256(item["stl_path"])}
                           for item in materials]}
    original = manifest.get("input_vdata")
    original_path = _resolve(original, manifest_path.parent) if isinstance(original, str) and original else None
    state["vdata_path"] = str(original_path) if original_path else None
    state["vdata_available"] = bool(original_path and original_path.is_file())
    state["vdata_sha256"] = sha256(original_path) if state["vdata_available"] else None
    if state["vdata_available"] and manifest.get("source_sha256") and state["vdata_sha256"] != manifest["source_sha256"]:
        raise ValueError("Original VDATA no longer matches manifest source_sha256")
    for material, snapshot in zip(materials, state["materials"]):
        if material.get("stl_sha256") and material["stl_sha256"] != snapshot["stl_sha256"]:
            raise ValueError("Input STL no longer matches manifest stl_sha256: " + snapshot["stl_path"])
    return state


def unchanged_state(before):
    manifest_ok = sha256(before["manifest_path"]) == before["manifest_sha256"]
    material_checks = [{"stl_path": item["stl_path"], "unchanged": sha256(item["stl_path"]) == item["stl_sha256"]}
                       for item in before["materials"]]
    vdata_ok = sha256(before["vdata_path"]) == before["vdata_sha256"] if before["vdata_available"] else None
    return {"input_manifest_unchanged": manifest_ok, "input_stls": material_checks,
            "source_vdata_unchanged": vdata_ok,
            "all_available_inputs_unchanged": manifest_ok and all(item["unchanged"] for item in material_checks) and vdata_ok is not False,
            "source_vdata_note": "checked" if before["vdata_available"] else "not present; original source_sha256 preserved but not reverified"}


def load_manifest(path):
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not isinstance(raw.get("materials"), list):
        raise ValueError("Expected a conversion manifest with a materials list")
    if raw.get("stl_units", "um") not in ("um", "µm", "μm"):
        raise ValueError("Only micrometre STL coordinates are supported")
    if raw["materials"]:
        return fdtd_bridge.read_manifest(path)
    return raw


def validate_resume(output, signature):
    report_path = output / "reduction_report.json"
    manifest_path = output / "conversion_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise ValueError("--resume requires completed reduced geometry and its report")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("geometry_status") != "complete" or report.get("geometry_signature") != signature:
        raise ValueError("Resume source hashes or reduction/metrology parameters differ; use a new output directory")
    if sha256(manifest_path) != report.get("output_manifest_sha256"):
        raise ValueError("Reduced output manifest was modified; refusing resume")
    outputs = report.get("output_stls")
    if not isinstance(outputs, list):
        raise ValueError("Resume report has no output STL hash list")
    output_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    listed_outputs = {str(Path(item["stl_path"]).resolve()): item["stl_sha256"] for item in outputs}
    manifested_outputs = {str(_resolve(item["stl_path"], output)): item.get("stl_sha256")
                          for item in output_manifest["materials"]}
    if listed_outputs != manifested_outputs or len(outputs) != len(output_manifest["materials"]):
        raise ValueError("Resume output STL hash list is inconsistent with its manifest")
    for item in outputs:
        path = Path(item["stl_path"])
        if not path.is_file() or sha256(path) != item["stl_sha256"]:
            raise ValueError("Reduced output STL was modified/missing; refusing resume: " + str(path))
    report["resumed_geometry"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, help="Existing conversion_manifest.json")
    parser.add_argument("--manifest", dest="manifest_option", type=Path, help="Alias for positional manifest")
    parser.add_argument("--output", required=True, type=Path, help="New or empty reduced-conversion directory")
    parser.add_argument("--reduction", type=float, default=0.45, help="Fraction of triangles to remove; 0 is exact copy; default 0.45")
    parser.add_argument("--sampled-max-error-nm", type=float, help="Optional sampled additional-distance tolerance; NOT a strict Hausdorff/CD bound")
    parser.add_argument("--material-map", type=Path, help="Same optical-material JSON mapping accepted by fdtd_bridge")
    parser.add_argument("--lumerical-root", type=Path, help="Lumerical installation used only for requested verification")
    parser.add_argument("--verify-fdtd", action="store_true", help="Import/save/reopen in a private CAD session; no electromagnetic solve")
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-matched completed geometry; redo mapping/FDTD verification as requested")
    args = parser.parse_args(argv)
    if (args.manifest is None) == (args.manifest_option is None):
        parser.error("Supply exactly one positional manifest or --manifest")
    if not math.isfinite(args.reduction) or not 0 <= args.reduction < 1:
        parser.error("--reduction must be finite and in [0,1)")
    if args.sampled_max_error_nm is not None and (not math.isfinite(args.sampled_max_error_nm) or args.sampled_max_error_nm < 0):
        parser.error("--sampled-max-error-nm must be finite and nonnegative")
    manifest_path = (args.manifest or args.manifest_option).resolve()
    output = args.output.resolve()
    report, state = None, None
    created_output = False
    begin = time.perf_counter()
    try:
        manifest = load_manifest(manifest_path)
        if args.reduction > 0 and len(manifest["materials"]) != 1:
            raise ValueError("Nonzero QEM currently requires exactly one solid-material entry; empty/multi-material conversions must use --reduction 0. Shared interfaces are not jointly constrained.")
        state = input_state(manifest_path, manifest)
        signature = {"input_manifest_sha256": state["manifest_sha256"],
                     "input_stls": state["materials"], "reduction": args.reduction,
                     "sampled_max_error_nm": args.sampled_max_error_nm,
                     "algorithm_version": ALGORITHM_VERSION,
                     "aggressiveness": 3.0,
                     "sampling": "all vertices + all face centroids + all distinct edge midpoints, both directions"}
        if args.resume:
            report = validate_resume(output, signature)
            report["current_input_state"] = state
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("Output is not empty; use a new directory or --resume")
            if args.reduction > 0:
                import qem_tools
                qem_tools.require_dependencies()
            output.mkdir(parents=True, exist_ok=True)
            created_output = True
            mesh_dir = output / "meshes"
            mesh_dir.mkdir(exist_ok=False)
            report = {"schema_version": 1, "status": "reducing", "geometry_status": "pending",
                      "input_manifest": str(manifest_path), "input_state": state,
                      "source_sha256": manifest.get("source_sha256"), "output_dir": str(output),
                      "geometry_signature": signature, "resumed_geometry": False,
                      "requested_reduction": args.reduction, "strict_hausdorff_bound": False,
                      "source_process_simulation_run": False, "materials": [], "output_stls": [],
                      "limitations": ["QEM is approximate; errors are additional to the input STL approximation.",
                                      "Finite sampled distances do not certify continuous Hausdorff distance, CD, LER or electromagnetic accuracy.",
                                      "Topology checks do not certify freedom from self-intersections.",
                                      "Nonzero reduction of multiple materials is blocked because shared interfaces are not jointly constrained.",
                                      "Surface reduction alone does not promise lower FDTD solver RAM for an unchanged Yee grid."]}
            reduced_manifest = deepcopy(manifest)
            reduced_manifest.pop("manifest_path", None)
            if state["vdata_path"]:
                reduced_manifest["input_vdata"] = state["vdata_path"]
            reduced_materials = []
            for index, original in enumerate(manifest["materials"], 1):
                source = Path(original["stl_path"])
                destination = mesh_dir / ("material_%04d.stl" % index)
                entry = deepcopy(original)
                entry.pop("object_name", None)
                source_audit = deepcopy(original.get("audit", {}))
                if args.reduction == 0:
                    shutil.copy2(source, destination)
                    if sha256(source) != sha256(destination):
                        raise RuntimeError("Exact-copy STL hash mismatch")
                    measurement = {"mode": "byte_exact_copy", "accepted": True, "requested_reduction": 0.0,
                                   "sampled_symmetric_maximum_nm": None, "additional_surface_error_nm": 0.0,
                                   "additional_volume_change_percent": 0.0,
                                   "topology": "byte-identical to input; topology/metrology not recomputed",
                                   "qem_dependencies_used": False}
                else:
                    print("Reducing material %d/%d: %s" % (index, len(manifest["materials"]), original["source_name"]), flush=True)
                    measurement = qem_tools.reduce_stl(source, destination, args.reduction, args.sampled_max_error_nm)
                    entry["baseline_audit"] = source_audit
                    entry["audit"] = deepcopy(measurement["output_audit"])
                    native_volume = original.get("volume_um3")
                    entry["audit"].update(stl_bytes=destination.stat().st_size, native_volume_um3=native_volume,
                        volume_difference_percent=(100 * (entry["audit"]["mesh_volume_um3"] / native_volume - 1)
                                                   if isinstance(native_volume, (int, float)) and native_volume > 0 else None))
                material_report = {"source_name": original["source_name"], "source_ids": original.get("source_ids", []),
                                   "translation_um": original.get("translation_um", manifest.get("translation_um", [0, 0, 0])),
                                   "input_stl_path": str(source), "input_stl_sha256": sha256(source),
                                   "output_stl_path": str(destination), "measurement": measurement}
                report["materials"].append(material_report)
                fdtd_bridge.atomic_json(output / "reduction_report.json", report)
                if not measurement["accepted"]:
                    raise ValueError("Sampled maximum %.9g nm exceeds requested tolerance %.9g nm; candidate retained for inspection, no import manifest published" % (
                        measurement["sampled_symmetric_maximum_nm"], args.sampled_max_error_nm))
                entry["stl_path"] = str(destination)
                entry["stl_sha256"] = sha256(destination)
                entry["surface_reduction"] = measurement
                report["output_stls"].append({"stl_path": str(destination), "stl_sha256": entry["stl_sha256"]})
                reduced_materials.append(entry)
            checks = unchanged_state(state)
            if not checks["all_available_inputs_unchanged"]:
                raise RuntimeError("An available input changed during reduction")
            reduced_manifest["materials"] = reduced_materials
            reduced_manifest["status"] = "geometry_converted"
            reduced_manifest["surface_reduction"] = {
                "algorithm_version": ALGORITHM_VERSION, "requested_reduction": args.reduction,
                "baseline_manifest": str(manifest_path), "baseline_manifest_sha256": state["manifest_sha256"],
                "sampled_max_error_nm": args.sampled_max_error_nm, "strict_hausdorff_bound": False,
                "mode": "byte_exact_copy" if args.reduction == 0 else "optional_approximate_QEM"}
            reduced_manifest["limitations"] = list(reduced_manifest.get("limitations", [])) + report["limitations"]
            target_manifest = output / "conversion_manifest.json"
            fdtd_bridge.atomic_json(target_manifest, reduced_manifest)
            report.update(geometry_status="complete", geometry_manifest=str(target_manifest),
                          output_manifest_sha256=sha256(target_manifest), input_unchanged_checks=checks)
            fdtd_bridge.atomic_json(output / "reduction_report.json", report)
        target_manifest = output / "conversion_manifest.json"
        if manifest["materials"]:
            fdtd = fdtd_bridge.build(target_manifest, output / "fdtd", material_map=args.material_map,
                                     lumerical_root=args.lumerical_root, verify=args.verify_fdtd)
            report["fdtd_report"] = fdtd
            status = fdtd["fdtd_status"]
            report["status"] = ("complete" if status in ("ready_to_import", "verified") else status)
        else:
            report["status"] = "empty_geometry"
            report["fdtd_report"] = {"fdtd_status": "empty_geometry", "time_domain_solver_run": False}
        checks = unchanged_state(state)
        report.update(input_unchanged_checks=checks, source_file_unchanged=checks["source_vdata_unchanged"],
                      input_files_unchanged=checks["all_available_inputs_unchanged"],
                      material_count=len(manifest["materials"]), elapsed_seconds=time.perf_counter() - begin)
        if not checks["all_available_inputs_unchanged"]:
            raise RuntimeError("An available input changed during reduction/import")
        fdtd_bridge.atomic_json(output / "reduction_report.json", report)
        print(json.dumps({"status": report["status"], "geometry_manifest": str(target_manifest),
                          "report": str(output / "reduction_report.json"), "resumed_geometry": report["resumed_geometry"]}), flush=True)
        if args.verify_fdtd and report["fdtd_report"]["fdtd_status"] not in ("verified", "empty_geometry"):
            return 3
        # Geometry plus a mapping template is a successful non-verified stage,
        # matching convert_vdata.py. Requested verification has a separate code.
        return 0
    except Exception as exc:
        # Never overwrite a pre-existing output/report when initial validation
        # or a resume fingerprint check fails.
        if report is not None and (created_output or args.resume):
            report.update(status="failed", error=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - begin)
            if state is not None:
                try:
                    report["input_unchanged_checks"] = unchanged_state(state)
                except Exception as check_error:
                    report["input_check_error"] = str(check_error)
            fdtd_bridge.atomic_json(output / "reduction_report.json", report)
        print("Reduction failed: " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
