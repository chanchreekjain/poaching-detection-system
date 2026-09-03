"""Verification tests.

These exist because the interesting failure modes of this system are silent.
A coordinate remap that is off by the crop origin still produces plausible
boxes; a clustering rule that never promotes still runs without error and just
never alerts. Both would look fine in a demo.

    python -m pytest tests -q          (or: python tests/test_pipeline.py)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Namespace, load_config  # noqa: E402
from src.detector import Detection, _expand_box  # noqa: E402
from src.evidence import EvidenceTracker, iou  # noqa: E402
from src.motion import MotionGate  # noqa: E402
from src.recorder import AlertRecorder  # noqa: E402


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # half-overlap: inter 50, union 150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_expand_box_clips_to_frame():
    # box hugging the left edge must not produce negative coordinates
    x1, y1, x2, y2 = _expand_box((5, 5, 105, 205), 0.45, 0.18, 640, 480)
    assert (x1, y1) == (0, 0)
    assert x2 == 150 and y2 == 241
    # and must not exceed the frame on the other side
    x1, y1, x2, y2 = _expand_box((600, 400, 639, 479), 0.45, 0.18, 640, 480)
    assert x2 == 640 and y2 == 480


# --------------------------------------------------------------------------
# clustering / alert promotion - the false-alert filter
# --------------------------------------------------------------------------
def _tracker(min_armed_hits=5, window=45, min_person_hits=6):
    cluster = Namespace(
        iou_match=0.25, max_age=30, window=window,
        min_armed_hits=min_armed_hits, min_person_hits=min_person_hits,
        alert_cooldown=300,
    )
    scoring = Namespace(
        w_weapon=0.60, w_person=0.20, w_persist=0.10, w_night=0.10,
        night_hours=[18, 6], alert_threshold=0.55,
    )
    return EvidenceTracker(cluster, scoring)


def _person(armed: bool, conf=0.9, box=(100, 100, 200, 400)):
    weapons = [{"xyxy": (150, 200, 180, 215), "conf": 0.62,
                "cls": 0, "label": "weapon"}] if armed else []
    return Detection(xyxy=box, conf=conf, weapons=weapons)


def test_persistent_armed_person_alerts():
    tr = _tracker()
    alerts = []
    for f in range(30):
        _, new = tr.update([_person(armed=True)], f, datetime(2026, 8, 25, 22, 0))
        alerts += new
    assert alerts, "a person armed in 30 consecutive frames must raise an alert"
    a = alerts[0]
    assert a.armed_hits >= 5
    assert a.score >= 0.55
    assert "weapon" in a.weapon_labels


def test_sporadic_weapon_hits_do_not_alert():
    """Two isolated false positives on an otherwise unarmed person: no alert.
    This is the whole point of the evidence window."""
    tr = _tracker()
    alerts = []
    for f in range(60):
        armed = f in (7, 31)  # two spurious frames, far apart
        _, new = tr.update([_person(armed=armed)], f, datetime(2026, 8, 25, 22, 0))
        alerts += new
    assert not alerts, f"sporadic hits must not alert, got {len(alerts)}"


def test_unarmed_person_never_alerts():
    tr = _tracker()
    alerts = []
    for f in range(120):
        _, new = tr.update([_person(armed=False)], f, datetime(2026, 8, 25, 22, 0))
        alerts += new
    assert not alerts


def test_separate_people_get_separate_tracks():
    tr = _tracker()
    left = (100, 100, 200, 400)
    right = (500, 100, 600, 400)
    for f in range(10):
        tracks, _ = tr.update(
            [_person(False, box=left), _person(False, box=right)], f
        )
    assert len(tracks) == 2, f"expected 2 tracks, got {len(tracks)}"
    assert len({t.track_id for t in tracks}) == 2


def test_alert_cooldown_prevents_repeat_spam():
    tr = _tracker()
    alerts = []
    for f in range(400):
        _, new = tr.update([_person(armed=True)], f, datetime(2026, 8, 25, 22, 0))
        alerts += new
    # cooldown is 300 frames, so at most 2 alerts in 400 frames
    assert 1 <= len(alerts) <= 2, f"cooldown failed, got {len(alerts)} alerts"


def test_night_bonus_raises_score():
    day = _tracker()
    night = _tracker()
    for f in range(20):
        day.update([_person(armed=True)], f, datetime(2026, 8, 25, 12, 0))
        night.update([_person(armed=True)], f, datetime(2026, 8, 25, 23, 0))
    assert night.tracks[0].score > day.tracks[0].score


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
def test_clip_has_preroll_and_correct_length():
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = Namespace(clip_seconds=2.0, pre_seconds=0.5, out_dir=str(tmp),
                        write_metadata=True, annotate_clip=False)
        fps, size = 10.0, (64, 48)
        rec = AlertRecorder(cfg, fps, size)
        assert rec.clip_frames == 20 and rec.pre_frames == 5

        frames = [np.full((48, 64, 3), i, np.uint8) for i in range(60)]
        for i, fr in enumerate(frames):
            rec.push(fr)
            if i == 20:
                rec.start_clip(1, {"track_id": 1}, datetime(2026, 8, 25, 22, 0))
        rec.flush()

        clips = sorted(tmp.glob("*.mp4"))
        assert len(clips) == 1, f"expected 1 clip, found {clips}"
        assert clips[0].with_suffix(".json").exists(), "metadata sidecar missing"

        cap = cv2.VideoCapture(str(clips[0]))
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        assert n == 20, f"clip should be 20 frames (2s @ 10fps), got {n}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# motion gate
# --------------------------------------------------------------------------
def test_motion_gate_opens_on_movement_and_closes_after_hold():
    cfg = Namespace(enabled=True, history=50, var_threshold=25,
                    detect_shadows=True, min_area_ratio=0.002, blur_kernel=3,
                    dilate_iter=1, open_iter=1, hold_frames=5,
                    keyframe_interval=0)
    gate = MotionGate(cfg)
    still = np.full((120, 160, 3), 70, np.uint8)
    for _ in range(60):
        gate.update(still)
    assert not gate.update(still).active, "static scene should keep the gate shut"

    moving = still.copy()
    cv2.rectangle(moving, (30, 30), (110, 100), (250, 250, 250), -1)
    assert gate.update(moving).reason == "motion"

    # after the object leaves, the gate holds open for hold_frames then closes
    reasons = [gate.update(still).reason for _ in range(12)]
    assert "hold" in reasons
    assert reasons[-1] == "idle", f"gate never closed: {reasons}"


# --------------------------------------------------------------------------
# crop-dataset builder - coordinate remapping is the silent-failure risk
# --------------------------------------------------------------------------
def test_build_crop_dataset_remaps_boxes_and_keeps_negatives():
    from src.datasets.build_crop_dataset import main as build_main

    tmp = Path(tempfile.mkdtemp())
    try:
        src = tmp / "raw"
        (src / "train" / "images").mkdir(parents=True)
        (src / "train" / "labels").mkdir(parents=True)
        W, H = 640, 480

        def write(stem, boxes):
            img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
            cv2.imwrite(str(src / "train" / "images" / f"{stem}.jpg"), img)
            lines = []
            for cls, (x1, y1, x2, y2) in boxes:
                lines.append(
                    f"{cls} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} "
                    f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
                )
            (src / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines))

        # armed: person (100,100)-(300,400), gun (250,200)-(310,230)
        write("armed", [(0, (100, 100, 300, 400)), (1, (250, 200, 310, 230))])
        # unarmed: person only -> must become a negative crop
        write("unarmed", [(0, (100, 100, 300, 400))])

        with open(src / "data.yaml", "w") as fh:
            yaml.safe_dump({"names": {0: "person", 1: "gun"},
                            "train": "train/images"}, fh)

        dst = tmp / "crops"
        rc = build_main(["--src", str(src), "--dst", str(dst),
                         "--margin-x", "0.45", "--margin-y", "0.18",
                         "--min-size", "32", "--neg-ratio", "2.0"])
        assert rc == 0

        lbl_dir = dst / "train" / "labels"
        pos = lbl_dir / "armed_p0.txt"
        neg = lbl_dir / "unarmed_p0.txt"
        assert pos.exists(), f"positive crop missing; have {list(lbl_dir.iterdir())}"
        assert neg.exists(), "negative (unarmed) crop was dropped - these are needed"
        assert neg.read_text().strip() == "", "negative must have an empty label file"

        # crop = (10,46)-(390,454) => 380x408
        # gun in crop coords = (240,154)-(300,184)
        cls, cx, cy, bw, bh = pos.read_text().split()
        assert cls == "0", "output must be single-class 'weapon'"
        assert abs(float(cx) - 270 / 380) < 1e-4, f"cx wrong: {cx}"
        assert abs(float(cy) - 169 / 408) < 1e-4, f"cy wrong: {cy}"
        assert abs(float(bw) - 60 / 380) < 1e-4, f"bw wrong: {bw}"
        assert abs(float(bh) - 30 / 408) < 1e-4, f"bh wrong: {bh}"

        crop_img = cv2.imread(str(dst / "train" / "images" / "armed_p0.jpg"))
        assert crop_img.shape[:2] == (408, 380), f"crop size {crop_img.shape}"

        spec = yaml.safe_load((dst / "data.yaml").read_text())
        assert spec["names"] == {0: "weapon"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
def test_config_loads_and_is_self_consistent():
    cfg = load_config()
    assert cfg.alert.pre_seconds <= cfg.alert.clip_seconds
    assert cfg.cluster.min_armed_hits <= cfg.cluster.window
    assert cfg.models.weapon_imgsz > 0
    assert 0.0 < cfg.scoring.alert_threshold < 1.0
    weights = (cfg.scoring.w_weapon + cfg.scoring.w_person
               + cfg.scoring.w_persist + cfg.scoring.w_night)
    assert abs(weights - 1.0) < 1e-6, f"scoring weights sum to {weights}, not 1.0"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
