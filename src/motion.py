"""Cheap motion gate.

Camera-trap and static-CCTV footage is mostly empty frames. Running a YOLO
pass on every one of them is wasted compute, and on an edge device it is the
difference between real-time and not.

This gate uses MOG2 background subtraction, which handles gradual illumination
change (sun moving, clouds) far better than naive frame differencing, and can
label shadow pixels so they do not count as motion.

Design note: the gate is *permissive by construction*. Once it fires it holds
the detector open for ``hold_frames``, and a keyframe interval forces a
detection pass periodically no matter what. A gate that misses a slow-moving
human is far more costly than one that occasionally wakes up for a branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionState:
    active: bool          # should the detector run on this frame?
    area_ratio: float     # fraction of the frame that changed
    reason: str           # 'motion' | 'hold' | 'keyframe' | 'idle' | 'disabled'


class MotionGate:
    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg.enabled)
        self.min_area_ratio = float(cfg.min_area_ratio)
        self.blur_kernel = int(cfg.blur_kernel)
        self.dilate_iter = int(cfg.dilate_iter)
        self.open_iter = int(cfg.open_iter)
        self.hold_frames = int(cfg.hold_frames)
        self.keyframe_interval = int(cfg.keyframe_interval)

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.history),
            varThreshold=float(cfg.var_threshold),
            detectShadows=bool(cfg.detect_shadows),
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._hold = 0
        self._frame_idx = -1
        # MOG2 needs a few frames to build a background model; everything looks
        # like motion until then, so we ignore the first ``_warmup`` frames.
        self._warmup = 12

    def update(self, frame: np.ndarray) -> MotionState:
        self._frame_idx += 1

        if not self.enabled:
            return MotionState(True, 0.0, "disabled")

        keyframe = (
            self.keyframe_interval > 0
            and self._frame_idx % self.keyframe_interval == 0
        )

        blurred = frame
        if self.blur_kernel > 1:
            k = self.blur_kernel | 1  # must be odd
            blurred = cv2.GaussianBlur(frame, (k, k), 0)

        mask = self._bg.apply(blurred)
        # MOG2 marks shadows as 127; keep only hard foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        if self.open_iter:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, self._kernel, iterations=self.open_iter
            )
        if self.dilate_iter:
            mask = cv2.dilate(mask, self._kernel, iterations=self.dilate_iter)

        area_ratio = float(np.count_nonzero(mask)) / float(mask.size)

        if self._frame_idx < self._warmup:
            return MotionState(keyframe, area_ratio, "keyframe" if keyframe else "idle")

        if area_ratio >= self.min_area_ratio:
            self._hold = self.hold_frames
            return MotionState(True, area_ratio, "motion")

        if self._hold > 0:
            self._hold -= 1
            return MotionState(True, area_ratio, "hold")

        return MotionState(keyframe, area_ratio, "keyframe" if keyframe else "idle")
