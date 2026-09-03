"""Fetch the base weapon dataset.

Recommended source
------------------
"Person-Gun Detection-1" on Roboflow Universe:
    https://universe.roboflow.com/gun-detection-nv4pf/person-gun-detection-1

Why this one:
  * 4,111 images, already annotated with BOTH person and gun boxes. Having the
    person boxes is what makes the person-crop training set in
    ``build_crop_dataset.py`` possible without pseudo-labelling anything.
  * CC BY 4.0 - permissive, attribution only. (Contrast with the academic
    Sohas / OD-WeaponDetection set, which is CC BY-SA 4.0, i.e. share-alike.)
  * Exports directly in YOLOv11 layout.

[Unverified] Image counts, licence and class names above are taken from the
dataset's Universe page as of August 2026. Re-check them before publishing any
result - Universe datasets can be re-versioned or taken down by their owners.

Usage
-----
    export ROBOFLOW_API_KEY=xxxxxxxx
    python -m src.datasets.download --out data/raw

Manual fallback (no API key needed): open the Universe link, click Download,
choose "YOLOv11", "download zip to computer", and unzip into data/raw/ so you
get data/raw/{train,valid,test}/{images,labels} and data/raw/data.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_WORKSPACE = "gun-detection-nv4pf"
DEFAULT_PROJECT = "person-gun-detection-1"
DEFAULT_VERSION = 1
DEFAULT_FORMAT = "yolov11"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the base weapon dataset")
    ap.add_argument("--out", default="data/raw", help="destination directory")
    ap.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--version", type=int, default=DEFAULT_VERSION)
    ap.add_argument("--format", default=DEFAULT_FORMAT,
                    help="roboflow export format (yolov11 / yolov8 / ...)")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    args = ap.parse_args(argv)

    if not args.api_key:
        print(
            "No API key. Set ROBOFLOW_API_KEY (free account -> Settings -> API "
            "keys), or download the zip manually - see the module docstring.",
            file=sys.stderr,
        )
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print("pip install roboflow", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    for fmt in (args.format, "yolov8"):  # yolov8 layout is identical, use as fallback
        try:
            dataset = version.download(fmt, location=str(out))
            print(f"downloaded ({fmt}) -> {dataset.location}")
            return 0
        except Exception as exc:  # noqa: BLE001 - report and try the fallback
            print(f"export format {fmt!r} failed: {exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
