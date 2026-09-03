"""Two-stage detector: person (stage 1) -> weapon on person crops (stage 2).

Why two stages rather than one model with an extra class
-------------------------------------------------------
1. The stock COCO person detector was trained on ~118k annotated images. Any
   public weapon dataset is 2-3 orders of magnitude smaller. Fine-tuning one
   model on a merged set trades away a very strong person detector for a
   marginal latency gain.
2. A carried firearm is a *small, thin, frequently occluded* object. At 640px
   full-frame input a rifle across a distant human is a handful of pixels.
   Cropping the person and upscaling gives stage 2 a far higher effective
   resolution on exactly the object that is hard to see. This is the single
   biggest recall lever in the whole system.
3. It encodes the semantics we actually want: a weapon *being carried by a
   person*, not a weapon-shaped thing somewhere in the frame.
4. On empty footage - which is most footage - stage 2 never runs at all, so
   the average cost is lower than a merged model despite the second pass.

The cost is that a weapon on a human stage 1 misses is invisible. Since the
alert logic requires a human anyway, that is an acceptable trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import resolve

log = logging.getLogger(__name__)

COCO_PERSON_CLASS_ID = 0


@dataclass
class Detection:
    """A person detection, plus whatever stage 2 found on their crop."""

    xyxy: tuple[float, float, float, float]
    conf: float
    weapons: list[dict] = field(default_factory=list)  # frame-coord weapon boxes

    @property
    def armed(self) -> bool:
        return bool(self.weapons)

    @property
    def best_weapon_conf(self) -> float:
        return max((w["conf"] for w in self.weapons), default=0.0)


def _expand_box(
    xyxy: tuple[float, float, float, float],
    margin_x: float,
    margin_y: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin_x
    x2 += bw * margin_x
    y1 -= bh * margin_y
    y2 += bh * margin_y
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


class TwoStageDetector:
    def __init__(self, cfg) -> None:
        from ultralytics import YOLO  # imported lazily so --help stays fast

        self.cfg = cfg
        m = cfg.models
        self.device = m.device
        # 'half' is deprecated in recent ultralytics and only meaningful on
        # CUDA, so it is passed only when explicitly switched on.
        self._extra = {"half": True} if bool(m.half) else {}

        log.info("loading stage-1 person model: %s", m.person_weights)
        self.person_model = YOLO(str(m.person_weights))

        self.weapon_model = None
        self.weapon_names: dict[int, str] = {}
        weapon_path = resolve(m.weapon_weights)
        if weapon_path.exists():
            log.info("loading stage-2 weapon model: %s", weapon_path)
            self.weapon_model = YOLO(str(weapon_path))
            self.weapon_names = dict(self.weapon_model.names)
        else:
            log.warning(
                "STAGE 2 DISABLED - no weapon weights at %s. Running in "
                "PERSON-ONLY mode: every human will look unarmed, so no armed "
                "alerts can ever fire. Train the weapon model first "
                "(see src/train_weapon.py).",
                weapon_path,
            )

    # -- properties ---------------------------------------------------------
    @property
    def stage2_available(self) -> bool:
        return self.weapon_model is not None

    # -- inference ----------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        m = self.cfg.models

        results = self.person_model.predict(
            frame,
            classes=[COCO_PERSON_CLASS_ID],
            conf=float(m.person_conf),
            iou=float(m.person_iou),
            imgsz=int(m.person_imgsz),
            device=self.device,
            verbose=False,
            **self._extra,
        )[0]

        detections: list[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            det = Detection(xyxy=tuple(float(v) for v in box), conf=float(conf))
            if self.weapon_model is not None:
                det.weapons = self._detect_weapons(frame, det.xyxy, w, h)
            detections.append(det)

        return detections

    def _detect_weapons(
        self,
        frame: np.ndarray,
        person_box: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> list[dict]:
        c = self.cfg.crop
        m = self.cfg.models

        x1, y1, x2, y2 = _expand_box(
            person_box, float(c.margin_x), float(c.margin_y), width, height
        )
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        if max(x2 - x1, y2 - y1) < int(c.min_size):
            return []  # too small for stage 2 to say anything meaningful

        crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        # Upscale small crops so the weapon occupies more pixels.
        scale = 1.0
        target = int(c.upscale_to)
        longest = max(crop_h, crop_w)
        if longest < target:
            scale = target / float(longest)
            crop = cv2.resize(
                crop,
                (int(round(crop_w * scale)), int(round(crop_h * scale))),
                interpolation=cv2.INTER_LINEAR,
            )

        res = self.weapon_model.predict(
            crop,
            conf=float(m.weapon_conf),
            iou=float(m.weapon_iou),
            imgsz=int(m.weapon_imgsz),
            device=self.device,
            verbose=False,
            **self._extra,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return []

        out: list[dict] = []
        wboxes = res.boxes.xyxy.cpu().numpy()
        wconfs = res.boxes.conf.cpu().numpy()
        wclss = res.boxes.cls.cpu().numpy().astype(int)
        for wb, wc, wk in zip(wboxes, wconfs, wclss):
            # crop coords -> original-crop coords -> frame coords
            fx1 = x1 + wb[0] / scale
            fy1 = y1 + wb[1] / scale
            fx2 = x1 + wb[2] / scale
            fy2 = y1 + wb[3] / scale
            out.append(
                {
                    "xyxy": (float(fx1), float(fy1), float(fx2), float(fy2)),
                    "conf": float(wc),
                    "cls": int(wk),
                    "label": self.weapon_names.get(int(wk), str(int(wk))),
                }
            )
        return out
