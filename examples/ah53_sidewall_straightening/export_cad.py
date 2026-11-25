"""Export an AH53 example VMESH to a Lumerical planar-solid MAT and LSF."""
import argparse
import json
from pathlib import Path
import tempfile

from vdata_standalone.coplanar import merge_coplanar
from vdata_standalone.coplanar_export import (
    compact_vertex_table,
    generate_coplanar_lsf,
    write_coplanar_mat,
)
from vdata_standalone.storage import read_mesh_archive


def export(source, output, material):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    vertices, faces, translation, _ = read_mesh_archive(source)
    offsets, indices, audit = merge_coplanar(vertices, faces)
    vertices, indices, compaction = compact_vertex_table(vertices, indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".ah53-cad-") as staging:
        work = Path(staging) / "export"
        work.mkdir()
        report = write_coplanar_mat(work / "geometry.mat", vertices, offsets, indices, translation)
        generate_coplanar_lsf([{
            "source_name": "/Silicon/Si_Xtal",
            "material_id": 2,
            "fdtd_material": material,
            "coplanar_compact_mat": str(output / "geometry.mat"),
        }], work, compact_vertices=True)
        report.update(path=str(output / "geometry.mat"), mesh=str(source),
                      material=material, merge=audit, vertex_compaction=compaction)
        (work / "export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        work.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One indexed silicon VMESH")
    parser.add_argument("--output", type=Path, required=True, help="New CAD export directory")
    parser.add_argument("--material", default="Si (Silicon) - Palik", help="Lumerical library material name")
    options = parser.parse_args()
    export(options.input, options.output, options.material)
