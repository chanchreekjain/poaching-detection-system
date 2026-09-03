"""Alert clip capture with pre-roll.

The point of the pre-roll is that by the time the evidence tracker is confident
enough to alert, the interesting part - the subject entering frame - has
already happened. A clip that starts at the moment of the alert shows you a
person standing still. So the recorder keeps a ring buffer of the last
``pre_seconds`` of frames and dumps it as the head of the clip.

One recorder can hold several clips open at once (two subjects, two alerts) and
each is closed independently once it has collected its full duration.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ActiveClip:
    path: Path
    writer: "cv2.VideoWriter"
    frames_remaining: int
    meta: dict = field(default_factory=dict)


class AlertRecorder:
    def __init__(self, cfg, fps: float, frame_size: tuple[int, int]) -> None:
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.frame_size = frame_size  # (width, height)
        self.clip_frames = int(round(float(cfg.clip_seconds) * self.fps))
        self.pre_frames = int(round(float(cfg.pre_seconds) * self.fps))
        self.pre_frames = min(self.pre_frames, self.clip_frames)
        self.write_metadata = bool(cfg.write_metadata)
        self.annotate = bool(cfg.annotate_clip)
        # Four poachers walking together are ONE incident, not four. Without
        # this, a group triggers one full-length clip per person - near
        # identical footage, four times the disk and four times the pager.
        self.merge_concurrent = bool(cfg.get("merge_concurrent", True))

        self._ring: deque[np.ndarray] = deque(maxlen=max(1, self.pre_frames))
        self._active: list[ActiveClip] = []
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # -- per-frame ----------------------------------------------------------
    def push(self, raw_frame: np.ndarray, annotated_frame: np.ndarray | None = None) -> None:
        """Feed every frame here, alert or not."""
        buffered = annotated_frame if (self.annotate and annotated_frame is not None) else raw_frame
        self._ring.append(buffered.copy())

        still_open: list[ActiveClip] = []
        for clip in self._active:
            if clip.frames_remaining > 0:
                clip.writer.write(buffered)
                clip.frames_remaining -= 1
            if clip.frames_remaining > 0:
                still_open.append(clip)
            else:
                self._close(clip)
        self._active = still_open

    # -- alerts -------------------------------------------------------------
    def start_clip(self, track_id: int, meta: dict, when: datetime | None = None) -> Path:
        when = when or datetime.now()

        # Same incident already being recorded -> attach this subject to it.
        if self.merge_concurrent and self._active:
            clip = self._active[0]
            clip.meta.setdefault("subjects", [clip.meta.get("track_id")])
            clip.meta["subjects"].append(track_id)
            clip.meta["subjects"] = sorted(
                {t for t in clip.meta["subjects"] if t is not None}
            )
            clip.meta.setdefault("merged_detail", []).append(meta)
            # A second subject is new information, so extend the clip a little
            # rather than letting it end mid-incident.
            clip.frames_remaining = max(clip.frames_remaining, self.clip_frames // 2)
            log.warning(
                "ALERT track #%d merged into in-progress clip %s (subjects now %s)",
                track_id, clip.path.name, clip.meta["subjects"],
            )
            return clip.path

        stamp = when.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"alert_{stamp}_track{track_id:03d}.mp4"

        writer = cv2.VideoWriter(str(path), self._fourcc, self.fps, self.frame_size)
        if not writer.isOpened():
            log.error("could not open VideoWriter for %s - clip skipped", path)
            return path

        # Dump the pre-roll first.
        pre = list(self._ring)
        for frame in pre:
            writer.write(frame)

        remaining = max(0, self.clip_frames - len(pre))
        clip = ActiveClip(
            path=path,
            writer=writer,
            frames_remaining=remaining,
            meta={**meta, "clip": path.name, "started_at": when.isoformat(),
                  "pre_roll_frames": len(pre), "fps": self.fps},
        )
        self._active.append(clip)
        log.warning("ALERT clip opened: %s (%d pre-roll frames)", path.name, len(pre))
        return path

    def _close(self, clip: ActiveClip) -> None:
        clip.writer.release()
        if self.write_metadata:
            meta_path = clip.path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(clip.meta, fh, indent=2, default=str)
        log.info("alert clip written: %s", clip.path.name)

    def flush(self) -> None:
        """Close any clip still open when the video ends."""
        for clip in self._active:
            self._close(clip)
        self._active = []

    @property
    def active_count(self) -> int:
        return len(self._active)
