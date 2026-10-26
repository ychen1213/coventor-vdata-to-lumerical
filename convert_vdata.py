"""Compatibility entry: python convert_vdata.py INPUT --output NEW_FOLDER."""
import sys
from vdata_standalone.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["convert"] + sys.argv[1:]))
