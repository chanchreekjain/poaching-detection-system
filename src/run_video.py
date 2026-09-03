"""Entry point: run the poaching-detection pipeline over a video file.

    python -m src.run_video --source data/clip.mp4 --save-render out.mp4

Pipeline per frame:
    motion gate -> stage 1 person -> stage 2 weapon on crops
                -> temporal clustering -> threat score -> alert clip
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from .config import load_config, resolve
from .detector import TwoStageDetector
from .evidence import EvidenceTracker
from .motion import MotionGate
from .recorder import AlertRecorder
from .viz import draw_frame

log = logging.getLogger("poaching")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poaching detection - video inference")
    p.add_argument("--source", required=True,
                   help="path to a video file, or a webcam index like 0")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--save-render", default=None,
                   help="write a full annotated render to this path")
    p.add_argument("--show", action="store_true",
                   help="display a live window (needs a GUI-capable OpenCV)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = whole video)")
    p.add_argument("--start-time", default=None,
                   help="ISO timestamp the video starts at, e.g. 2026-08-25T21:40:00. "
                        "Used only for the night-time term in the threat score.")
    p.add_argument("--device", default=None, help="override models.device")
    p.add_argument("--no-motion-gate", action="store_true",
                   help="run the detector on every frame")
    p.add_argument("--debug-force-armed", action="store_true",
                   help="DEBUG ONLY: treat every detected person as armed. Exercises "
                        "the clustering -> scoring -> alert-clip path before the "
                        "stage-2 model exists. Never use this for real footage.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, str(cfg.output.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.device:
        cfg["models"]["device"] = args.device
    if args.no_motion_gate:
        cfg["motion"]["enabled"] = False

    source = int(args.source) if str(args.source).isdigit() else str(resolve(args.source))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("could not open source: %s", args.source)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log.info("source %s | %dx%d @ %.2f fps | %s frames",
             args.source, width, height, fps, total or "unknown")

    start_time = datetime.fromisoformat(args.start_time) if args.start_time else None

    detector = TwoStageDetector(cfg)
    gate = MotionGate(cfg.motion)
    tracker = EvidenceTracker(cfg.cluster, cfg.scoring)

    alert_cfg = cfg.alert
    alert_dir = resolve(alert_cfg.out_dir)
    alert_cfg["out_dir"] = str(alert_dir)
    recorder = AlertRecorder(alert_cfg, fps, (width, height))

    events_path = resolve(cfg.output.events_log)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_fh = open(events_path, "a", encoding="utf-8")

    render = None
    render_path = args.save_render or cfg.output.annotated_video
    if render_path:
        render_path = resolve(render_path)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        render = cv2.VideoWriter(
            str(render_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        log.info("writing annotated render to %s", render_path)

    frame_idx = 0
    n_detect_frames = 0
    n_alerts = 0
    detections = []
    tracks = []
    t0 = time.perf_counter()
    inference_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            state = gate.update(frame)
            now = (start_time + timedelta(seconds=frame_idx / fps)) if start_time else None

            if state.active:
                t_inf = time.perf_counter()
                detections = detector.detect(frame)
                inference_time += time.perf_counter() - t_inf
                n_detect_frames += 1
                if args.debug_force_armed:
                    for det in detections:
                        x1, y1, x2, y2 = det.xyxy
                        det.weapons = [{
                            "xyxy": (x1, (y1 + y2) / 2, x2, (y1 + y2) / 2 + 12),
                            "conf": 0.99, "cls": 0, "label": "DEBUG_FAKE",
                        }]
            else:
                detections = []

            tracks, new_alerts = tracker.update(detections, frame_idx, now)

            annotated = None
            if render is not None or args.show or recorder.annotate:
                elapsed = time.perf_counter() - t0
                annotated = draw_frame(
                    frame, tracks, detections, state, frame_idx,
                    fps=(frame_idx + 1) / elapsed if elapsed > 0 else None,
                    stage2=detector.stage2_available,
                )

            recorder.push(frame, annotated)

            for track in new_alerts:
                n_alerts += 1
                meta = {
                    "track_id": track.track_id,
                    "frame": frame_idx,
                    "video_time_s": round(frame_idx / fps, 2),
                    "wall_time": now.isoformat() if now else None,
                    "threat_score": round(track.score, 3),
                    "armed_hits": track.armed_hits,
                    "window_len": len(track.armed_window),
                    "armed_ratio": round(track.armed_ratio, 3),
                    "person_conf": round(track.conf, 3),
                    "weapons": track.weapon_labels,
                    "person_box_xyxy": [round(v, 1) for v in track.box],
                    "source": str(args.source),
                }
                clip_path = recorder.start_clip(track.track_id, meta, now or datetime.now())
                meta["clip_path"] = str(clip_path)
                events_fh.write(json.dumps(meta, default=str) + "\n")
                events_fh.flush()
                log.warning(
                    "ALERT track #%d at %.1fs | score %.2f | %s",
                    track.track_id, frame_idx / fps, track.score,
                    track.weapon_labels or "no weapon labels",
                )

            if render is not None and annotated is not None:
                render.write(annotated)
            if args.show and annotated is not None:
                cv2.imshow("poaching-detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if cfg.output.show_fps and frame_idx % 100 == 0:
                el = time.perf_counter() - t0
                log.info("… %d frames | %.1f fps overall | %d detector passes | %d alerts",
                         frame_idx, frame_idx / el if el else 0, n_detect_frames, n_alerts)

    except KeyboardInterrupt:
        log.warning("interrupted by user")
    finally:
        recorder.flush()
        cap.release()
        if render is not None:
            render.release()
        events_fh.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    skipped = frame_idx - n_detect_frames
    log.info("=" * 72)
    log.info("frames processed      : %d", frame_idx)
    log.info("detector passes       : %d (%d frames skipped by motion gate, %.1f%%)",
             n_detect_frames, skipped, 100.0 * skipped / frame_idx if frame_idx else 0.0)
    log.info("mean detector latency : %.1f ms",
             1000.0 * inference_time / n_detect_frames if n_detect_frames else 0.0)
    log.info("wall clock            : %.1f s (%.1f fps end-to-end)",
             elapsed, frame_idx / elapsed if elapsed else 0.0)
    log.info("alerts raised         : %d  -> %s", n_alerts, alert_dir)
    if args.debug_force_armed:
        log.warning(
            "--debug-force-armed was ON: every person was marked armed. These "
            "%d alerts test the plumbing only and mean NOTHING about accuracy.",
            n_alerts,
        )
    elif not detector.stage2_available:
        log.warning(
            "stage 2 was DISABLED - person-only run. Humans were tracked but no "
            "weapon check ran, so no armed alert could ever fire."
        )
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
