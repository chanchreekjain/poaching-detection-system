"""Generates notebooks/train_weapon_colab.ipynb.

The notebook is generated rather than hand-written so the cell contents stay
easy to edit and always produce valid ipynb JSON.

    python tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
(MD, """# Stage-2 Weapon Detector — Colab (free tier)

Trains the weapon detector for the Poaching Detection System. Stage 1 (people)
needs no training; this notebook produces the model that decides whether a
detected human is **armed**.

**Free-tier reality:** sessions disconnect, often within a few hours and
without warning. Checkpoints are therefore written to **Google Drive**, and the
training cell is **resumable** — if you get disconnected, re-run cells 1–6 and
then re-run the training cell with `RESUME = True`. Nothing is lost.

**Runtime → Change runtime type → T4 GPU** before you start. On CPU this will
not finish.

Order: setup → data → **look at the data** → train → **measure false
positives** → download."""),

(CODE, """#@title 1. GPU check — stop here if this fails
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout or
      'NO GPU. Runtime -> Change runtime type -> T4 GPU, then re-run.')"""),

(CODE, """#@title 2. Mount Drive (checkpoints survive disconnects)
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = '/content/drive/MyDrive/poaching'   #@param {type:"string"}
os.makedirs(DRIVE_ROOT, exist_ok=True)
print('checkpoints ->', DRIVE_ROOT)"""),

(CODE, """#@title 3. Install dependencies
!pip -q install "ultralytics>=8.4.0" roboflow
import ultralytics, torch
print('ultralytics', ultralytics.__version__, '| torch', torch.__version__,
      '| cuda', torch.cuda.is_available())"""),

(MD, """## 4. Upload the project

Upload **`poaching_project.zip`** from your `poaching` folder. This keeps one
source of truth: the crop-building code that runs here is the same code the
detector uses at inference, so the training and inference distributions cannot
drift apart.

The zip is cached to Drive, so on a reconnect this cell restores it without
re-uploading."""),

(CODE, """#@title 4. Upload / restore the project
import os, shutil, zipfile

WORK = '/content/poaching'
cached_zip = os.path.join(DRIVE_ROOT, 'poaching_project.zip')

if os.path.exists(cached_zip):
    print('restoring cached zip from Drive')
    src = cached_zip
else:
    from google.colab import files
    up = files.upload()
    src = list(up.keys())[0]
    shutil.copy(src, cached_zip)
    print('cached to Drive for next time')

if os.path.exists(WORK):
    shutil.rmtree(WORK)
with zipfile.ZipFile(src) as z:
    z.extractall('/content')

# the zip may or may not contain a top-level folder
if not os.path.exists(os.path.join(WORK, 'src')):
    for d in os.listdir('/content'):
        p = os.path.join('/content', d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'src', 'detector.py')):
            WORK = p
            break

os.chdir(WORK)
print('working dir:', os.getcwd())
print(sorted(os.listdir()))"""),

(CODE, """#@title 5. Roboflow API key + dataset download
from getpass import getpass
import os

if not os.environ.get('ROBOFLOW_API_KEY'):
    os.environ['ROBOFLOW_API_KEY'] = getpass('Roboflow API key (hidden): ')

!python -m src.datasets.download --out data/raw

print()
print(open('data/raw/data.yaml').read())"""),

(MD, """### Check the class names above

`build_crop_dataset` matches class names by regex, but datasets get re-versioned
and renamed. If the printed names are not what you expect, pass explicit ids to
the next cell with `--person-classes` / `--weapon-classes` rather than letting
it guess."""),

(CODE, """#@title 6. Build the person-crop training set
# Margins MUST match config.yaml crop.margin_x / crop.margin_y, otherwise the
# model trains on a different framing than it sees at inference.
import yaml
cfg = yaml.safe_load(open('config.yaml'))
MX, MY = cfg['crop']['margin_x'], cfg['crop']['margin_y']
IMGSZ = cfg['models']['weapon_imgsz']
print(f'using margins from config.yaml: margin_x={MX} margin_y={MY} imgsz={IMGSZ}')

!python -m src.datasets.build_crop_dataset \\
    --src data/raw --dst data/weapon_crops \\
    --margin-x {MX} --margin-y {MY} --neg-ratio 1.5"""),

(MD, """## 7. Look at the data before you spend GPU hours

A coordinate-remap bug produces boxes that are plausible but wrong, trains
without error, and yields a model that is quietly useless. Two minutes of
looking now is worth more than any metric later.

**Green boxes must sit on the weapon**, and the crops should look like tight
shots of a person — the same framing the detector produces at inference."""),

(CODE, """#@title 7. Visual sanity check
import glob, random, cv2, matplotlib.pyplot as plt

pos = [p for p in glob.glob('data/weapon_crops/train/labels/*.txt')
       if open(p).read().strip()]
neg = [p for p in glob.glob('data/weapon_crops/train/labels/*.txt')
       if not open(p).read().strip()]
print(f'{len(pos)} positive crops, {len(neg)} negative crops '
      f'({len(neg)/max(1,len(pos)+len(neg)):.0%} negative)')

random.seed(0)
sample = random.sample(pos, min(8, len(pos)))
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
for ax, lp in zip(axes.ravel(), sample):
    img = cv2.imread(lp.replace('/labels/', '/images/').replace('.txt', '.jpg'))
    h, w = img.shape[:2]
    for line in open(lp).read().strip().splitlines():
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
        x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis('off')
for ax in axes.ravel()[len(sample):]:
    ax.axis('off')
plt.tight_layout(); plt.show()"""),

(MD, """## 8. Train

`RESUME = False` for a fresh run. If Colab disconnects, re-run cells 1–6, set
`RESUME = True`, and re-run this cell — it picks up from `last.pt` on Drive.

Batch 32 fits a T4 at 448px. Drop to 16 if you hit CUDA OOM."""),

(CODE, """#@title 8. Train the weapon detector
RESUME  = False   #@param {type:"boolean"}
EPOCHS  = 80      #@param {type:"integer"}
BATCH   = 32      #@param {type:"integer"}
MODEL   = "yolo11s.pt"  #@param ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]

resume_flag = "--resume" if RESUME else ""
!python -m src.train_weapon \\
    --data data/weapon_crops/data.yaml \\
    --model {MODEL} --epochs {EPOCHS} --imgsz {IMGSZ} --batch {BATCH} \\
    --device 0 --project "{DRIVE_ROOT}/runs" --name weapon_yolo11s \\
    {resume_flag}"""),

(CODE, """#@title 9. Training curves and confusion matrix
from IPython.display import Image, display
import os
run = f'{DRIVE_ROOT}/runs/weapon_yolo11s'
for f in ['results.png', 'confusion_matrix_normalized.png',
          'val_batch0_pred.jpg', 'PR_curve.png']:
    p = os.path.join(run, f)
    if os.path.exists(p):
        print(f); display(Image(p, width=900))"""),

(MD, """### Reading these honestly

High mAP on this val split means the model learned *this dataset*. It does not
mean it will work on your footage — the val split shares the training
distribution. Look at `val_batch0_pred.jpg`: are the boxes on the weapons, or
on hands and torsos that merely correlate with weapons in this dataset?

The next cell is the check that actually generalises."""),

(CODE, """#@title 10. False positives on weapon-free footage
# The number that decides deployability: how often does it cry wolf on
# ordinary people carrying nothing? Uses the synthetic clip by default;
# point --source at your own unarmed footage for a real answer.
!python tools/make_test_video.py --out data/test_clip.mp4
!python -m src.eval_false_positives --source data/test_clip.mp4 \\
    --no-motion-gate --device 0 --json-out data/fp_eval.json"""),

(CODE, """#@title 11. Download the trained weights
import shutil, os
best = f'{DRIVE_ROOT}/runs/weapon_yolo11s/weights/best.pt'
if os.path.exists(best):
    shutil.copy(best, f'{DRIVE_ROOT}/weapon_yolo11s.pt')
    print('on Drive at:', f'{DRIVE_ROOT}/weapon_yolo11s.pt')
    from google.colab import files
    files.download(best)
else:
    print('no best.pt found — did training finish?')"""),

(MD, """## Done — back on your laptop

Put the downloaded `best.pt` at:

```
C:\\Users\\CJ\\Downloads\\poaching\\models\\weapon_yolo11s.pt
```

Then run — the `STAGE-2 OFF` warning should be gone:

```
python -m src.run_video --source data\\test_clip.mp4 --save-render data\\render.mp4
```

### What to do with the false-positive numbers

- **Zero false alerts at a low threshold** → set `models.weapon_conf` to the
  lowest such value in `config.yaml`. Prefer low: raising it costs recall on
  real weapons, which is the failure that matters.
- **False alerts at every threshold** → raise `cluster.min_armed_hits` first
  (costs latency, not recall). If that is not enough, *then* add the Sohas
  hard negatives — phones, purses, cards held in hand. That is the ablation
  worth running, and now you have a baseline to compare it against.

Still unmeasured: whether it detects **real** weapons. That needs armed
footage, and public handgun/CCTV data is a poor proxy for rifles at distance in
the field."""),
]


def main() -> None:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == MD:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            cells.append({"cell_type": "code", "metadata": {},
                          "execution_count": None, "outputs": [], "source": src})

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    out = Path(__file__).resolve().parent.parent / "notebooks"
    out.mkdir(exist_ok=True)
    path = out / "train_weapon_colab.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
