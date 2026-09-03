"""Overlay drawing. Kept separate so the pipeline can run headless with
annotation switched off and pay nothing for it."""

from __future__ import annotations

import cv2
import numpy as np

GREEN = (80, 200, 80)
AMBER = (0, 190, 255)
RED = (40, 40, 235)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)


def _label(img, text, org, colour, scale=0.5, thickness=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 6, y + 2), colour, -1)
    cv2.putText(
        img, text, (x + 3, y - base + 1), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (20, 20, 20), thickness, cv2.LINE_AA,
    )


def draw_frame(
    frame: np.ndarray,
    tracks,
    detections,
    motion_state=None,
    frame_idx: int = 0,
    fps: float | None = None,
    stage2: bool = True,
) -> np.ndarray:
    out = frame.copy()

    # Weapon boxes come from the current frame's detections.
    for det in detections:
        for w in det.weapons:
            x1, y1, x2, y2 = (int(v) for v in w["xyxy"])
            cv2.rectangle(out, (x1, y1), (x2, y2), RED, 2)
            _label(out, f"{w['label']} {w['conf']:.2f}", (x1, y1 - 4), RED, 0.45)

    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.box)
        if track.alerted:
            colour, tag = RED, "ARMED"
        elif track.armed_hits > 0:
            colour, tag = AMBER, "suspect"
        else:
            colour, tag = GREEN, "person"
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        _label(
            out,
            f"#{track.track_id} {tag} s={track.score:.2f} w={track.armed_hits}/{len(track.armed_window)}",
            (x1, y1 - 4),
            colour,
        )

    # Status strip
    bits = [f"frame {frame_idx}"]
    if motion_state is not None:
        bits.append(f"motion:{motion_state.reason} {motion_state.area_ratio*100:.2f}%")
    if fps:
        bits.append(f"{fps:.1f} fps")
    if not stage2:
        bits.append("STAGE-2 OFF (person-only)")
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (35, 35, 35), -1)
    cv2.putText(
        out, "  |  ".join(bits), (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, WHITE if stage2 else AMBER, 1, cv2.LINE_AA,
    )
    return out
