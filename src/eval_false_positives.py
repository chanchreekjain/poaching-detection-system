"""Measure false positives on footage you know contains NO weapons.

Why this exists
---------------
Validation mAP is computed on a split of the same dataset the model trained on,
so it shares that distribution and cannot tell you how the model behaves on
YOUR footage. The question that actually decides whether this system is
deployable is different and much simpler:

    Point it at an hour of ordinary people carrying nothing dangerous.
    How often does it cry wolf?

This script answers that, and it answers it at two levels:

  * per-crop  - how often the raw stage-2 detector fires. Expect this to be
                non-trivial; it is what the clustering layer exists to absorb.
  * per-alert - how often a false hit actually survives clustering and scoring
                to wake somebody up. This is the number that matters.

The gap between those two is precisely the value the evidence layer adds, and
it is worth knowing rather than assuming.

It also sweeps the weapon confidence threshold, so you can pick
``models.weapon_conf`` from measured behaviour instead of a guess.

    python -m src.eval_false_positives --source data/unarmed_footage.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from .config import load_config, resolve
from .detector import Detection, TwoStageDetector
from .evidence import EvidenceTracker
from .motion import MotionGate

log = logging.getLogger("eval-fp")

DEFAULT_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="False-positive rate on weapon-free footage")
    ap.add_argument("--source", required=True,
                    help="video containing people but NO weapons")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sweep", default=",".join(str(t) for t in DEFAULT_SWEEP),
                    help="comma-separated weapon confidence thresholds")
    ap.add_argument("--no-motion-gate", action="store_true",
                    help="examine every frame (recommended for evaluation)")
    ap.add_argument("--json-out", default=None, help="write results as JSON")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s | %(message)s")

    thresholds = sorted(float(t) for t in args.sweep.split(","))
    lowest = thresholds[0]

    if args.device:
        cfg["models"]["device"] = args.device
    # Run stage 2 wide open, then filter offline. One pass, whole sweep.
    cfg["models"]["weapon_conf"] = lowest
    if args.no_motion_gate:
        cfg["motion"]["enabled"] = False

    detector = TwoStageDetector(cfg)
    if not detector.stage2_available:
        log.error("no stage-2 weapon model - nothing to evaluate. Train it first.")
        return 2

    cap = cv2.VideoCapture(str(resolve(args.source)))
    if not cap.isOpened():
        log.error("could not open %s", args.source)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    gate = MotionGate(cfg.motion)
    # per-frame records: list of (person_box, person_conf, [weapon confidences])
    timeline: list[list[tuple[tuple, float, list[float]]]] = []
    n_frames = n_examined = n_crops = 0

    log.info("pass 1: running detector at weapon_conf=%.2f", lowest)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n_frames >= args.max_frames:
            break
        state = gate.update(frame)
        row = []
        if state.active:
            n_examined += 1
            for det in detector.detect(frame):
                n_crops += 1
                row.append((det.xyxy, det.conf, [w["conf"] for w in det.weapons]))
        timeline.append(row)
        n_frames += 1
        if n_frames % 200 == 0:
            log.info("  %d frames…", n_frames)
    cap.release()

    if n_crops == 0:
        log.error("no person crops found - stage 1 detected nobody, so this "
                  "footage cannot measure stage-2 false positives.")
        return 1

    log.info("pass 2: replaying %d frames at %d thresholds", n_frames, len(thresholds))
    rows = []
    for thr in thresholds:
        crops_firing = 0
        total_hits = 0
        tracker = EvidenceTracker(cfg.cluster, cfg.scoring)
        alerts = 0
        for idx, row in enumerate(timeline):
            dets = []
            for box, pconf, wconfs in row:
                kept = [c for c in wconfs if c >= thr]
                if kept:
                    crops_firing += 1
                    total_hits += len(kept)
                dets.append(Detection(
                    xyxy=box, conf=pconf,
                    weapons=[{"xyxy": box, "conf": c, "cls": 0, "label": "weapon"}
                             for c in kept],
                ))
            _, new = tracker.update(dets, idx, None)
            alerts += len(new)
        rows.append({
            "weapon_conf": thr,
            "crops_firing": crops_firing,
            "crop_fp_rate": crops_firing / n_crops,
            "raw_hits": total_hits,
            "false_alerts": alerts,
            "alerts_per_hour": alerts / (n_frames / fps / 3600) if n_frames else 0.0,
        })

    minutes = n_frames / fps / 60
    print("\n" + "=" * 78)
    print(f"FALSE-POSITIVE EVALUATION  ({Path(args.source).name})")
    print(f"footage    : {n_frames} frames, {minutes:.1f} min @ {fps:.1f} fps")
    print(f"examined   : {n_examined} frames, {n_crops} person crops")
    print("ASSUMPTION : this footage contains NO weapons. Every hit below is a "
          "false positive.")
    print("-" * 78)
    print(f"{'conf':>6} | {'crops firing':>13} | {'crop FP rate':>12} | "
          f"{'false alerts':>12} | {'alerts/hr':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['weapon_conf']:>6.2f} | {r['crops_firing']:>13} | "
              f"{r['crop_fp_rate']:>11.2%} | {r['false_alerts']:>12} | "
              f"{r['alerts_per_hour']:>9.1f}")
    print("=" * 78)

    zero = [r for r in rows if r["false_alerts"] == 0]
    if zero:
        best = min(zero, key=lambda r: r["weapon_conf"])
        print(f"\nLowest threshold with zero false alerts: {best['weapon_conf']:.2f} "
              f"(raw detector still fired on {best['crop_fp_rate']:.1%} of crops - "
              f"the clustering layer absorbed all of them).")
        print("Prefer the LOWEST such threshold: raising it costs you recall on "
              "real weapons, which is the failure that matters.")
    else:
        print("\nEvery threshold produced false alerts. Options, in order of "
              "preference:")
        print("  1. raise cluster.min_armed_hits (costs alert latency, not recall)")
        print("  2. add hard negatives to training (phones/tools held in hand)")
        print("  3. raise weapon_conf - last resort, it costs real-weapon recall")

    print("\nThis measures FALSE POSITIVES ONLY. It says nothing about whether "
          "the model detects real weapons - that needs armed footage.")

    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "source": str(args.source), "frames": n_frames,
            "person_crops": n_crops, "fps": fps, "results": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
