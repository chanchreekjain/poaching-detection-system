"""Temporal-spatial clustering and threat scoring.

The false-alert problem
-----------------------
A per-frame weapon detector at conf 0.3 will fire spuriously on branches,
straps, tool handles, walking sticks and phone-in-hand poses. Alerting on a
single frame is unusable in the field - rangers stop trusting the system after
the second false callout, and then it may as well not exist.

So detections are not alerts. Detections are *evidence*. Each person detection
is linked into a track by single-link clustering over (IoU, time gap): a
detection joins the nearest existing cluster whose last box overlaps it above
``iou_match`` and which is no older than ``max_age`` frames, otherwise it seeds
a new cluster. A cluster only promotes to an alert once it holds
``min_armed_hits`` weapon-positive frames inside a rolling window of
``window`` frames, and clears the composite threat score.

That window requirement is the actual false-alert filter. A branch that looks
like a barrel for two frames never reaches five hits on one persistent track;
a human walking through carrying a rifle does so easily.

[Inference] The specific thresholds here are starting points chosen to be
conservative, not values validated against field data - they need tuning on
real footage from the deployment site before any accuracy claim is made.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """One clustered human subject followed across frames."""

    track_id: int
    box: Box
    conf: float
    first_frame: int
    last_frame: int
    hits: int = 1
    misses: int = 0
    # rolling windows of booleans/confidences, capped at cluster.window
    armed_window: deque = field(default_factory=deque)
    person_window: deque = field(default_factory=deque)
    weapon_labels: dict = field(default_factory=dict)
    last_alert_frame: int | None = None
    alerted: bool = False
    score: float = 0.0

    @property
    def armed_hits(self) -> int:
        return sum(self.armed_window)

    @property
    def armed_ratio(self) -> float:
        return (self.armed_hits / len(self.armed_window)) if self.armed_window else 0.0

    @property
    def age(self) -> int:
        return self.last_frame - self.first_frame + 1


class EvidenceTracker:
    """Single-link IoU clustering over time + threat scoring."""

    def __init__(self, cluster_cfg, scoring_cfg) -> None:
        self.iou_match = float(cluster_cfg.iou_match)
        self.max_age = int(cluster_cfg.max_age)
        self.window = int(cluster_cfg.window)
        self.min_armed_hits = int(cluster_cfg.min_armed_hits)
        self.min_person_hits = int(cluster_cfg.min_person_hits)
        self.alert_cooldown = int(cluster_cfg.alert_cooldown)

        self.w_weapon = float(scoring_cfg.w_weapon)
        self.w_person = float(scoring_cfg.w_person)
        self.w_persist = float(scoring_cfg.w_persist)
        self.w_night = float(scoring_cfg.w_night)
        self.night_start, self.night_end = (int(v) for v in scoring_cfg.night_hours)
        self.alert_threshold = float(scoring_cfg.alert_threshold)

        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    # -- helpers ------------------------------------------------------------
    def _is_night(self, when: datetime | None) -> bool:
        if when is None:
            return False
        h = when.hour
        if self.night_start <= self.night_end:
            return self.night_start <= h < self.night_end
        return h >= self.night_start or h < self.night_end  # wraps midnight

    def _push(self, track: Track, armed: bool, conf: float) -> None:
        track.armed_window.append(bool(armed))
        track.person_window.append(float(conf))
        while len(track.armed_window) > self.window:
            track.armed_window.popleft()
        while len(track.person_window) > self.window:
            track.person_window.popleft()

    def _score(self, track: Track, night: bool) -> float:
        person_conf = (
            sum(track.person_window) / len(track.person_window)
            if track.person_window
            else 0.0
        )
        persist = min(1.0, track.hits / max(1, self.min_person_hits * 2))
        return (
            self.w_weapon * track.armed_ratio
            + self.w_person * person_conf
            + self.w_persist * persist
            + self.w_night * (1.0 if night else 0.0)
        )

    # -- main update --------------------------------------------------------
    def update(
        self,
        detections,
        frame_idx: int,
        timestamp: datetime | None = None,
    ) -> tuple[list[Track], list[Track]]:
        """Link detections into tracks. Returns (active_tracks, new_alerts)."""
        night = self._is_night(timestamp)
        unmatched = list(range(len(detections)))
        matched_tracks: set[int] = set()

        # Greedy single-link assignment, highest IoU first.
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di in unmatched:
                score = iou(track.box, detections[di].xyxy)
                if score >= self.iou_match:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_dets: set[int] = set()
        for _score, ti, di in pairs:
            if ti in matched_tracks or di in used_dets:
                continue
            det = detections[di]
            track = self.tracks[ti]
            track.box = det.xyxy
            track.conf = det.conf
            track.last_frame = frame_idx
            track.hits += 1
            track.misses = 0
            self._push(track, det.armed, det.conf)
            for wdet in det.weapons:
                track.weapon_labels[wdet["label"]] = max(
                    track.weapon_labels.get(wdet["label"], 0.0), wdet["conf"]
                )
            matched_tracks.add(ti)
            used_dets.add(di)

        # Unmatched detections seed new clusters.
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            track = Track(
                track_id=next(self._ids),
                box=det.xyxy,
                conf=det.conf,
                first_frame=frame_idx,
                last_frame=frame_idx,
            )
            self._push(track, det.armed, det.conf)
            for wdet in det.weapons:
                track.weapon_labels[wdet["label"]] = wdet["conf"]
            self.tracks.append(track)

        # Age out unmatched tracks.
        survivors: list[Track] = []
        for track in self.tracks:
            if track.last_frame != frame_idx:
                track.misses += 1
                if track.misses > self.max_age:
                    continue
            survivors.append(track)
        self.tracks = survivors

        # Promote clusters that clear both the hit count and the score.
        alerts: list[Track] = []
        for track in self.tracks:
            track.score = self._score(track, night)
            if track.hits < self.min_person_hits:
                continue
            if track.armed_hits < self.min_armed_hits:
                continue
            if track.score < self.alert_threshold:
                continue
            if (
                track.last_alert_frame is not None
                and frame_idx - track.last_alert_frame < self.alert_cooldown
            ):
                continue
            track.last_alert_frame = frame_idx
            track.alerted = True
            alerts.append(track)

        return self.tracks, alerts
