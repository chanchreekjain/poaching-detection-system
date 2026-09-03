"""Synthesise a test video so the pipeline can be exercised without field data.

Builds a clip with three phases:
  1. static scene       -> the motion gate should stay closed
  2. a person-containing region pans in -> gate opens, stage 1 fires, a track
     forms and accumulates person hits
  3. static again       -> the gate closes after the hold window

Real people come from the Ultralytics sample image (bus.jpg), so stage 1
produces genuine detections rather than synthetic boxes.

    python tools/make_test_video.py --out data/test_clip.mp4
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SAMPLE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"


def fetch_sample(cache: Path) -> np.ndarray:
    """Prefer the copy that ships inside the installed ultralytics package;
    fall back to downloading it only if that is missing."""
    if not cache.exists():
        try:
            import ultralytics
            local = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
            if local.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(local.read_bytes())
                print(f"using bundled sample image: {local}")
        except Exception:  # noqa: BLE001
            pass
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SAMPLE_URL}")
        urllib.request.urlretrieve(SAMPLE_URL, cache)  # noqa: S310
    img = cv2.imread(str(cache))
    if img is None:
        raise RuntimeError(f"could not read {cache}")
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/test_clip.mp4")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--static-seconds", type=float, default=2.0)
    ap.add_argument("--move-seconds", type=float, default=12.0)
    ap.add_argument("--tail-seconds", type=float, default=4.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    people = fetch_sample(Path("data/_bus.jpg"))
    ph, pw = people.shape[:2]
    scale = args.height / ph * 0.95
    people = cv2.resize(people, (int(pw * scale), int(args.height * 0.95)))
    ph, pw = people.shape[:2]

    # A plain green-brown "bush" background, plus mild per-frame sensor noise so
    # the background model has something realistic to converge on.
    rng = np.random.default_rng(0)
    base = np.zeros((args.height, args.width, 3), np.uint8)
    base[:, :] = (58, 82, 46)
    for _ in range(400):
        x, y = rng.integers(0, args.width), rng.integers(0, args.height)
        r = int(rng.integers(6, 26))
        c = tuple(int(v) for v in rng.integers(30, 110, 3))
        cv2.circle(base, (int(x), int(y)), r, c, -1)
    base = cv2.GaussianBlur(base, (9, 9), 0)

    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        print(f"could not open writer for {out}", file=sys.stderr)
        return 1

    n_static = int(args.static_seconds * args.fps)
    n_move = int(args.move_seconds * args.fps)
    n_tail = int(args.tail_seconds * args.fps)

    def noisy() -> np.ndarray:
        noise = rng.normal(0, 2.0, base.shape)
        return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    for _ in range(n_static):
        writer.write(noisy())

    x_start, x_end = args.width, -pw
    for i in range(n_move):
        frame = noisy()
        t = i / max(1, n_move - 1)
        x = int(x_start + (x_end - x_start) * t)
        y = int((args.height - ph) / 2)
        sx1, sx2 = max(0, -x), min(pw, args.width - x)
        dx1, dx2 = max(0, x), min(args.width, x + pw)
        if sx2 > sx1 and dx2 > dx1:
            frame[y:y + ph, dx1:dx2] = people[:, sx1:sx2]
        writer.write(frame)

    for _ in range(n_tail):
        writer.write(noisy())

    writer.release()
    total = n_static + n_move + n_tail
    print(f"wrote {out} - {total} frames, {total/args.fps:.1f}s, "
          f"{args.width}x{args.height} @ {args.fps}fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
