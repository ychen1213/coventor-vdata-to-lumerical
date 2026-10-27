"""Convenience entry: accepts a subcommand or a VDATA file dropped onto run.cmd."""
import sys
from vdata_standalone.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["--help"]
    elif args[0] not in {"inspect", "decode", "convert", "lsf", "pack-mesh", "optimize-mesh", "export-stl", "export-planar"} and not args[0].startswith("-"):
        args.insert(0, "convert")
    raise SystemExit(main(args))
