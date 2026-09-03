"""Fine-tune YOLO11 into the stage-2 weapon detector.

This is the step that actually creates the new class. Starting from COCO
weights is still worth doing even though COCO has no gun: the backbone's
low-level features (edges, texture, material) transfer, and the detection head
is re-initialised for the new class count automatically by Ultralytics.

Augmentation notes
------------------
``fliplr`` is left on - a gun carried on the left shoulder is the same object
mirrored. ``mosaic`` is reduced from the default: mosaic is great for scale
diversity on full scenes, but our inputs are already tight crops at a
consistent scale, and heavy mosaic pushes the training distribution away from
what stage 2 sees at inference. ``close_mosaic`` disables it entirely for the
final epochs so the model finishes on clean crops.

    python -m src.train_weapon --data data/weapon_crops/data.yaml --epochs 80
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import PROJECT_ROOT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the stage-2 weapon detector")
    ap.add_argument("--data", default="data/weapon_crops/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="yolo11n.pt (fast/edge) .. yolo11m.pt (accurate)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=448,
                    help="must match models.weapon_imgsz in config.yaml")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu, mps")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="weapon_yolo11s")
    ap.add_argument("--project", default=None,
                    help="run output dir. On Colab point this at a Google Drive "
                         "path so checkpoints survive a disconnect.")
    ap.add_argument("--resume", action="store_true",
                    help="resume from last.pt in the existing run directory")
    ap.add_argument("--export-to", default="models/weapon_yolo11s.pt",
                    help="copy the best checkpoint here when training finishes")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        print(f"dataset yaml not found: {data_path}\n"
              "Run: python -m src.datasets.download && "
              "python -m src.datasets.build_crop_dataset", file=sys.stderr)
        return 2

    project_dir = Path(args.project) if args.project else (PROJECT_ROOT / "runs")

    # Ultralytics resumes from a CHECKPOINT, not from the base weights. Loading
    # args.model with resume=True silently restarts from epoch 0 - which on a
    # free-tier Colab that disconnects every few hours means you never finish.
    start_from = args.model
    if args.resume:
        last = project_dir / args.name / "weights" / "last.pt"
        if not last.exists():
            print(f"--resume given but no checkpoint at {last}; starting fresh.",
                  file=sys.stderr)
        else:
            print(f"resuming from {last}")
            start_from = str(last)

    model = YOLO(start_from)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.resume,
        resume=args.resume,
        # augmentation tuned for tight person crops
        mosaic=0.4,
        close_mosaic=15,
        scale=0.4,
        translate=0.08,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.45,
        erasing=0.2,
        seed=0,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nbest checkpoint: {best}")

    metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)
    try:
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"precision: {metrics.box.mp:.4f}")
        print(f"recall   : {metrics.box.mr:.4f}")
    except AttributeError:
        print(f"metrics: {metrics}")

    if args.export_to and best.exists():
        dest = Path(args.export_to)
        if not dest.is_absolute():
            dest = PROJECT_ROOT / dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"copied -> {dest}  (config.yaml models.weapon_weights points here)")

    print(
        "\nBefore trusting this model: look at the confusion matrix and the "
        "val batch images in the run directory. On a small weapon dataset a "
        "high mAP number can still mean the model has memorised backgrounds "
        "rather than learned the object."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
