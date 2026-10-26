"""Offline MAT/LSF export of indexed triangles; no CAD or vendor API calls."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np


def write_planar_mat(path, vertices_local_m, faces, translation_m):
    """Preserve float64 geometry, using Ansys' documented (3,1,F) facet table.

    The output is MATLAB v5, readable with matlabload. Vertices use local metres;
    facets are one-based doubles. No vertex welding, triangulation or smoothing.
    This exports data only and makes no assertion that a CAD import was executed.
    """
    from scipy.io import savemat
    from .meshing import _mesh_arrays

    vertices, faces = _mesh_arrays(vertices_local_m, faces)
    translation = np.asarray(translation_m, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation_m must be a finite XYZ vector")
    if len(vertices) >= 2 ** 53:
        raise ValueError("Vertex indices exceed exact double integer representation")
    # MAT v5 array elements use a uint32 byte-count. Refuse large individual arrays.
    if max(vertices.size, faces.size) * 8 >= 2 ** 31:
        raise ValueError("This MAT v5 exporter limits each array to less than 2 GiB")
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to replace existing MAT: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temp_path = Path(f.name)
        facets = (faces.T.astype(np.float64) + 1.0)[:, None, :]
        savemat(str(temp_path), {"vdata_vertices": vertices, "vdata_facets": facets,
                                "vdata_translation_m": translation.reshape(3, 1)},
                appendmat=False, format="5", do_compression=True, oned_as="column")
        # Only publish a complete file. Existing/concurrently created destinations
        # are preserved, and a failed save never leaves a partial final MAT.
        os.link(temp_path, path)
        return {"path": str(path), "bytes": path.stat().st_size, "vertices": len(vertices),
                "faces": len(faces), "facet_shape": [3, 1, len(faces)],
                "vertices_unit": "m", "vertex_frame": "local", "indices_base": 1,
                "geometry_changed": False, "cad_import_executed": False,
                "doc": "https://optics.ansys.com/hc/en-us/articles/360034924113-addplanarsolid-Script-command"}
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def generate_planar_lsf(objects, folder):
    from .cli import _lsf_string
    group = "VDATA_standalone"
    lines = ["# Indexed triangles -> planar solid, generated offline without any CAD SDK.",
             "# Same vertices/faces as archive; local metres, one-based facet indices.",
             "# Open a blank FDTD layout. This script was not executed by the converter."]
    missing = [o["source_name"] for o in objects if not o.get("fdtd_material")]
    if missing:
        lines.append("?" + _lsf_string("Import blocked: resolve material_map.json for " + ", ".join(missing)) + ";")
    else:
        lines += ["switchtolayout;", "vdata_standalone_ready=1;",
                  "if (getnamednumber(" + _lsf_string(group) + ")>0) { vdata_standalone_ready=0;",
                  "?" + _lsf_string("Destination group already exists; use an empty layout.") + "; }"]
        for name in sorted({o["fdtd_material"] for o in objects}):
            lines += ["if (materialexists(" + _lsf_string(name) + ")==0) { vdata_standalone_ready=0;",
                      "?" + _lsf_string("Material missing from this FDTD library: " + name) + "; }"]
        lines += ["if (vdata_standalone_ready) {", "addstructuregroup;", 'set("name",' + _lsf_string(group) + ");"]
        for obj in objects:
            path = (Path(folder) / obj["planar_mat"]).resolve()
            lines += ["matlabload(" + _lsf_string(path.as_posix()) + ");", "addplanarsolid;",
                      'set("vertices",vdata_vertices);', 'set("facets",vdata_facets);',
                      'set("name",' + _lsf_string("material_" + str(obj["material_id"])) + ");",
                      'set("material",' + _lsf_string(obj["fdtd_material"]) + ");"]
            lines += ['set("' + axis + '",' + format(v, ".17g") + ");" for axis, v in zip("xyz", obj["translation_m"])]
            lines += ["addtogroup(" + _lsf_string(group) + ");"]
        lines += ["unselectall;", "}"]
    path = Path(folder) / "import_lumerical_planar.lsf"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
