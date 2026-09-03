"""Turn a person+weapon detection dataset into a *person-crop* weapon dataset.

The idea
--------
At inference the stage-2 model only ever sees expanded person crops. If we
train it on full frames instead, train and test distributions disagree: it
learns to find guns in wide scenes and is then asked to find them in tight
upscaled crops. Training directly on crops removes that gap, and the upscaling
means small distant weapons occupy many more pixels than they would full-frame.

What this script does, per source image:
  1. read the YOLO label file
  2. for every person box, expand it by the same margins the detector uses
  3. crop, and re-express every weapon box inside that crop in crop coordinates
  4. write the crop as a training image, with a single class: ``weapon``

Crops containing no weapon are kept as **negatives** (empty label file). These
matter more than the positives do: they are what teaches the model that an
ordinary human - arms out, holding a stick, carrying a bag - is not armed.
Without them the model says "gun" for every person crop it sees.

Run:
    python -m src.datasets.build_crop_dataset --src data/raw --dst data/weapon_crops
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

PERSON_PATTERNS = [r"^person", r"^people", r"^human", r"^pedestrian"]
WEAPON_PATTERNS = [
    r"gun", r"pistol", r"handgun", r"rifle", r"firearm", r"weapon",
    r"shotgun", r"revolver", r"knife", r"machete",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classify_names(names: dict[int, str]) -> tuple[set[int], set[int]]:
    person_ids, weapon_ids = set(), set()
    for idx, raw in names.items():
        name = str(raw).strip().lower()
        if any(re.search(p, name) for p in PERSON_PATTERNS):
            person_ids.add(int(idx))
        elif any(re.search(p, name) for p in WEAPON_PATTERNS):
            weapon_ids.add(int(idx))
    return person_ids, weapon_ids


def load_names(data_yaml: Path) -> dict[int, str]:
    with open(data_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return dict(enumerate(names))
    raise ValueError(f"cannot read 'names' from {data_yaml}")


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
    return out


def yolo_to_xyxy(box, w: int, h: int):
    _, cx, cy, bw, bh = box
    return (
        (cx - bw / 2) * w, (cy - bh / 2) * h,
        (cx + bw / 2) * w, (cy + bh / 2) * h,
    )


def build_split(
    src_split: Path,
    dst_split: Path,
    person_ids: set[int],
    weapon_ids: set[int],
    margin_x: float,
    margin_y: float,
    min_size: int,
    overlap_keep: float,
    neg_ratio: float,
    rng: random.Random,
) -> Counter:
    img_dir, lbl_dir = src_split / "images", src_split / "labels"
    if not img_dir.exists():
        return Counter()

    out_img, out_lbl = dst_split / "images", dst_split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    pending_negatives: list[tuple[Path, "cv2.Mat"]] = []

    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        labels = read_labels(lbl_dir / f"{img_path.stem}.txt")
        if not labels:
            stats["images_no_labels"] += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            stats["images_unreadable"] += 1
            continue
        h, w = image.shape[:2]

        persons = [yolo_to_xyxy(b, w, h) for b in labels if b[0] in person_ids]
        weapons = [yolo_to_xyxy(b, w, h) for b in labels if b[0] in weapon_ids]
        stats["src_persons"] += len(persons)
        stats["src_weapons"] += len(weapons)

        if not persons:
            # No person box. If there is a weapon, the full image is still a
            # usable positive - it is roughly "a crop of the whole scene".
            if weapons:
                _write_sample(
                    out_img, out_lbl, f"{img_path.stem}_full", image,
                    [(_clip(wb, 0, 0, w, h), w, h) for wb in weapons], w, h,
                )
                stats["pos_fullframe"] += 1
            continue

        for pi, pbox in enumerate(persons):
            x1, y1, x2, y2 = _expand(pbox, margin_x, margin_y, w, h)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            if max(x2 - x1, y2 - y1) < min_size:
                stats["crops_too_small"] += 1
                continue

            crop = image[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]

            kept = []
            for wb in weapons:
                inter = _intersect(wb, (x1, y1, x2, y2))
                if inter is None:
                    continue
                warea = max(1e-6, (wb[2] - wb[0]) * (wb[3] - wb[1]))
                iarea = (inter[2] - inter[0]) * (inter[3] - inter[1])
                if iarea / warea < overlap_keep:
                    continue  # only a sliver of the weapon is in this crop
                kept.append((
                    (inter[0] - x1, inter[1] - y1, inter[2] - x1, inter[3] - y1),
                    cw, ch,
                ))

            stem = f"{img_path.stem}_p{pi}"
            if kept:
                _write_sample(out_img, out_lbl, stem, crop, kept, cw, ch)
                stats["positives"] += 1
            else:
                pending_negatives.append((out_img / f"{stem}.jpg", crop))

    # Cap negatives so the split does not become 95% empty images.
    n_pos = stats["positives"] + stats["pos_fullframe"]
    cap = int(n_pos * neg_ratio) if n_pos else len(pending_negatives)
    rng.shuffle(pending_negatives)
    for path, crop in pending_negatives[:cap]:
        cv2.imwrite(str(path), crop)
        (out_lbl / f"{path.stem}.txt").write_text("", encoding="utf-8")
        stats["negatives"] += 1
    stats["negatives_dropped"] += max(0, len(pending_negatives) - cap)

    return stats


def _expand(box, mx: float, my: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(round(x1 - bw * mx))),
        max(0, int(round(y1 - bh * my))),
        min(w, int(round(x2 + bw * mx))),
        min(h, int(round(y2 + bh * my))),
    )


def _clip(box, x1, y1, x2, y2):
    return (max(box[0], x1), max(box[1], y1), min(box[2], x2), min(box[3], y2))


def _intersect(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _write_sample(out_img: Path, out_lbl: Path, stem: str, image, boxes, w: int, h: int):
    cv2.imwrite(str(out_img / f"{stem}.jpg"), image)
    lines = []
    for (bx1, by1, bx2, by2), bw_ref, bh_ref in boxes:
        cx = ((bx1 + bx2) / 2) / bw_ref
        cy = ((by1 + by2) / 2) / bh_ref
        bw = (bx2 - bx1) / bw_ref
        bh = (by2 - by1) / bh_ref
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(bw, 1.0), min(bh, 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (out_lbl / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a person-crop weapon dataset")
    ap.add_argument("--src", default="data/raw", help="YOLO dataset root (has data.yaml)")
    ap.add_argument("--dst", default="data/weapon_crops")
    ap.add_argument("--margin-x", type=float, default=0.45)
    ap.add_argument("--margin-y", type=float, default=0.18)
    ap.add_argument("--min-size", type=int, default=64)
    ap.add_argument("--overlap-keep", type=float, default=0.55,
                    help="fraction of the weapon box that must fall inside the crop")
    ap.add_argument("--neg-ratio", type=float, default=1.5,
                    help="max negative crops per positive crop")
    ap.add_argument("--person-classes", default=None,
                    help="comma-separated class ids to treat as person (override)")
    ap.add_argument("--weapon-classes", default=None,
                    help="comma-separated class ids to treat as weapon (override)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    data_yaml = src / "data.yaml"
    if not data_yaml.exists():
        print(f"no data.yaml in {src} - run src/datasets/download.py first", file=sys.stderr)
        return 2

    names = load_names(data_yaml)
    person_ids, weapon_ids = classify_names(names)
    if args.person_classes:
        person_ids = {int(v) for v in args.person_classes.split(",")}
    if args.weapon_classes:
        weapon_ids = {int(v) for v in args.weapon_classes.split(",")}

    print(f"class names        : {names}")
    print(f"person class ids   : {sorted(person_ids) or 'NONE'}")
    print(f"weapon class ids   : {sorted(weapon_ids) or 'NONE'}")
    if not weapon_ids:
        print("ERROR: no weapon-like class found. Pass --weapon-classes explicitly.",
              file=sys.stderr)
        return 2
    if not person_ids:
        print("WARNING: no person-like class found. Every image will be kept "
              "full-frame, which loses the crop-distribution benefit.", file=sys.stderr)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    rng = random.Random(args.seed)
    totals = Counter()
    splits_present = []
    for split in ("train", "valid", "val", "test"):
        src_split = src / split
        if not (src_split / "images").exists():
            continue
        out_name = "valid" if split == "val" else split
        print(f"\n-- {split} -> {out_name}")
        stats = build_split(
            src_split, dst / out_name, person_ids, weapon_ids,
            args.margin_x, args.margin_y, args.min_size,
            args.overlap_keep, args.neg_ratio, rng,
        )
        for k, v in sorted(stats.items()):
            print(f"   {k:22s} {v}")
        totals.update(stats)
        if stats:
            splits_present.append(out_name)

    out_yaml = dst / "data.yaml"
    spec = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val": ("valid/images" if "valid" in splits_present else "train/images"),
        "names": {0: "weapon"},
    }
    if "test" in splits_present:
        spec["test"] = "test/images"
    with open(out_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)

    print(f"\nTOTALS: {dict(totals)}")
    print(f"wrote {out_yaml}")
    if totals["positives"] + totals["pos_fullframe"] == 0:
        print("ERROR: zero positive crops produced - check the class mapping.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
