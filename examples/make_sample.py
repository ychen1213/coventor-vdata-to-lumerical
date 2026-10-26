"""Create a tiny, wholly synthetic VoxelXML example using the standard library.

This is a format/installation example, not a Coventor process simulation or AH53.
The raised silicon cross includes a 25%-silicon top layer. No plane equations.
"""
import argparse
import base64
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New .vdata file; never overwrite")
    args = parser.parse_args()
    raw = bytearray()
    for x in range(24):
        for y in range(24):
            cross = ((10 <= x < 14 and 5 <= y < 19) or
                     (5 <= x < 19 and 10 <= y < 14))
            for z in range(20):
                base = 2 <= x < 22 and 2 <= y < 22 and 2 <= z < 8
                raised = cross and 8 <= z < 16
                if cross and z == 16:
                    raw.extend((32, 2, 255))
                else:
                    material = 2 if base or raised else 255
                    raw.extend((129, material, material))

    root = ET.Element("VoxelXML")
    root.append(ET.Comment("Synthetic demonstration only; not a process result"))
    props = ET.SubElement(root, "Properties")
    ET.SubElement(props, "VoxelSize", Units="um", Value="0.02")
    ET.SubElement(props, "NumVoxels", NX="24", NY="24", NZ="20")
    ET.SubElement(props, "ModelOrigin", Units="um", X="-0.24", Y="-0.24", Z="-0.2")
    materials = ET.SubElement(root, "Materials")
    ET.SubElement(materials, "Material", MaterialID="2", name="Si_Xtal")
    packed = zlib.compress(raw)
    payload = b"!zlib!" + struct.pack("!II", len(raw), len(packed)) + packed
    ET.SubElement(root, "VoxelData").text = base64.b64encode(payload).decode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    print(f"Created synthetic example: {args.output.resolve()}")
    print("24 x 24 x 20 cells; 20 nm spacing; Si=2, Air=255; no PlaneEquations.")


if __name__ == "__main__":
    main()
