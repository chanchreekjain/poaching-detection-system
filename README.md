# Poaching Detection System — YOLO11

Two-stage detector for wildlife-protection camera footage. Stage 1 finds
**people**, stage 2 checks whether they are **carrying a weapon**, and an
evidence-clustering layer decides whether that is worth waking a ranger for.

Human presence alone is a useless alarm — rangers, tourists, researchers and
villagers all trip it. The discriminator is *human **and** weapon*, sustained
across frames.

> **Status: stage 2 is not trained yet.** Stage 1, the motion gate, clustering,
> scoring and alert capture are implemented and tested (12/12 passing, verified
> on Windows and Linux). The weapon detector has not been trained, so the system
> currently runs in person-only mode and cannot flag anyone as armed. There are
> no accuracy numbers anywhere in this repo, and none should be inferred.
> `notebooks/train_weapon_colab.ipynb` is the next step.

```
video ─▶ motion gate ─▶ [1] person detector ─▶ crop+upscale ─▶ [2] weapon detector
                                                                      │
                        alert clip ◀── threat score ◀── temporal clustering
```

---

## Read this before anything else

### 1. You cannot "add a gun class to the COCO file"

`coco.yaml` is a *label-name list*. The pretrained `yolo11*.pt` weights carry an
80-neuron head bound to those names. Appending `80: gun` changes nothing —
there is no output neuron for it and the model has never seen a labelled gun.
The gun class only exists once you **train** a head on annotated gun images.
That is what `src/train_weapon.py` does, and it is the only step in this repo
that requires a GPU and real time.

### 2. About the "ammo" class

I could not find a public object-detection dataset with annotated ammunition
(cartridges, magazines, belts). Beyond availability: at realistic camera-trap
distances a cartridge is a handful of pixels, frequently pocketed, and would
contribute almost nothing over the firearm detection itself. **My
recommendation is to drop it** and put that effort into weapon *recall*
instead, which is where this system will actually fail. If you want it anyway
it means hand-annotating your own set — say so and I'll write the annotation
tooling.

### 3. The dataset domain gap is the biggest risk in this project

[Inference] Public weapon datasets are overwhelmingly **CCTV robbery footage:
handguns, indoors, close range, urban**. Poaching involves **long guns,
machetes, bows and snares, outdoors, at distance, often at night or in IR**.
A model trained on the former and deployed on the latter will underperform in
ways that a validation-set mAP number will not show you, because the val set
shares the training distribution. This is a reasoned expectation from the
distribution mismatch, not a measured result — but treat any mAP you get from
the public dataset as an upper bound, not a field estimate. Collecting even a
few hundred labelled frames from your actual deployment site and fine-tuning on
them will matter more than any architecture choice here.

### 4. Nothing here has been validated on real footage

The code is tested (see `tests/`, 12 passing) and verified end to end on a
synthetic clip. There are **no accuracy numbers** in this repo and every
threshold in `config.yaml` is a reasoned starting point, not a tuned value.

---

## Why two stages instead of one merged model

| | Two-stage (this repo) | One model, person + weapon classes |
|---|---|---|
| Person accuracy | Stock COCO head, ~118k training images | Degraded — retrained on a ~4k-image set |
| Small-weapon recall | **High** — crop is upscaled to 448px | Low — a rifle at 640px full-frame is a few pixels |
| Cost on empty frames | Stage 2 never runs | Full model always runs |
| Cost on busy frames | 1 + N passes | 1 pass |
| Semantics | "weapon carried by a person" | "weapon somewhere in frame" |

Small-object recall is the deciding factor. A carried firearm is thin, often
occluded by the body, and small at range; cropping the person and upscaling is
the single largest recall lever available. The trade-off is that a weapon on a
human stage 1 misses is invisible — acceptable, since the alert requires a
human anyway.

If you later need this on a Raspberry Pi and stage 2 proves too slow, the merged
model is the fallback; `build_crop_dataset.py` would be swapped for a plain
class-remap. Benchmark before switching — on mostly-empty footage two-stage is
usually *cheaper* on average.

---

## Setup

```bash
pip install -r requirements.txt
```

Stage-1 weights (`yolo11s.pt`) download automatically on first run.
Verified against **ultralytics 8.4.128**, torch 2.13, Python 3.11.

---

## Run it right now (no training needed)

```bash
python tools/make_test_video.py --out data/test_clip.mp4
python -m src.run_video --source data/test_clip.mp4 --save-render data/render.mp4
```

This runs in **person-only mode** and says so loudly — humans are detected and
tracked, but with no weapon model nothing can ever be flagged as armed.

To prove the alert path (clustering → scoring → clip capture) works before you
have a weapon model:

```bash
python -m src.run_video --source data/test_clip.mp4 --debug-force-armed
```

`--debug-force-armed` marks **every** person as armed. It tests plumbing only
and means nothing about accuracy. Never point it at real footage.

---

## Training the weapon detector

**1. Get the data.**

Recommended: [Person-Gun Detection-1](https://universe.roboflow.com/gun-detection-nv4pf/person-gun-detection-1)
on Roboflow Universe — 4,111 images, **both** person and gun annotated,
CC BY 4.0, exports in YOLOv11 layout. Having the person boxes is what makes the
person-crop training set possible without pseudo-labelling anything.

```bash
export ROBOFLOW_API_KEY=xxxx
python -m src.datasets.download --out data/raw
```

Or download the zip manually from the Universe page (format: YOLOv11) and unzip
to `data/raw/`. Attribution is required by CC BY 4.0 — keep it in your paper.

> Alternative: [OD-WeaponDetection / Sohas](https://github.com/ari-dasci/OD-WeaponDetection)
> (academic, CC BY-**SA** 4.0 — note the share-alike). Its value is the
> distractor classes: phone, purse, card, bill, i.e. *objects held in the hand
> the same way a pistol is*. Folding those in as hard negatives is the most
> direct way to cut false alerts on "ranger holding a phone". The share-alike
> licence is viral, so check it against your intended use.
>
> [Unverified] Image counts and licences above are from those pages as of
> August 2026 — re-check before citing them.

**2. Build the person-crop training set.**

```bash
python -m src.datasets.build_crop_dataset --src data/raw --dst data/weapon_crops
```

This crops each annotated person (with the same margins the detector uses at
inference), remaps weapon boxes into crop coordinates, and emits a single-class
`weapon` dataset. Person crops with no weapon are kept as **negatives** — these
matter more than the positives, because they are what stops the model calling
every human armed. Class names are matched by regex, so it adapts to whatever
the dataset calls things; override with `--person-classes` / `--weapon-classes`.

**3. Train.**

On a local GPU:

```bash
python -m src.train_weapon --data data/weapon_crops/data.yaml --epochs 80 --device 0
```

**On Colab free tier — use `notebooks/train_weapon_colab.ipynb`.** It clones
this repo, so there is nothing to upload; note that it clones what you have
**pushed**, so commit local changes first. Checkpoints go to Google Drive and
the training cell is resumable, because free-tier sessions disconnect without
warning; on a reconnect, re-run cells 1–6 and set `RESUME = True`. The notebook
also makes you *look at the training crops* before spending GPU hours — a
coordinate bug produces plausible-but-wrong boxes that train cleanly and yield a
quietly useless model.

Copies `best.pt` to `models/weapon_yolo11s.pt`, which `config.yaml` already
points at. On CPU this is impractically slow.

**4. Measure false positives — do not skip this.**

```bash
python -m src.eval_false_positives --source data/unarmed_footage.mp4 --no-motion-gate
```

Point it at footage containing people but **no weapons**. It sweeps
`weapon_conf` and reports, at each threshold, how often the raw detector fires
per crop *and* how many of those survive clustering to become an alert. The gap
between those two numbers is what the evidence layer is worth.

Validation mAP cannot answer this — it is computed on a split of the training
distribution. "How often does it cry wolf on ordinary people?" is the question
that decides whether this is deployable, and it needs your footage.

Prefer the **lowest** threshold that gives zero false alerts. Raising
`weapon_conf` costs recall on real weapons, which is the failure that matters.

**4. Run for real.**

```bash
python -m src.run_video --source data/field_clip.mp4 \
    --save-render out.mp4 --start-time 2026-08-25T21:40:00
```

`--start-time` only feeds the night-time term in the threat score.

---

## How false alerts are actually suppressed

Three independent layers, because no single one is enough:

1. **Motion gate** (`src/motion.py`) — MOG2 background subtraction. Skips the
   detector on empty frames. Deliberately permissive: once it fires it holds
   open for `hold_frames`, and a keyframe interval forces a pass regardless.
2. **Person gating** — stage 2 only ever runs on a person crop, so a
   weapon-shaped branch in open scrub is never even examined.
3. **Temporal-spatial clustering** (`src/evidence.py`) — detections are
   *evidence*, not alerts. Each person is linked into a track by IoU over time;
   a track promotes only after `min_armed_hits` weapon-positive frames inside a
   rolling `window`, and only if the composite threat score clears
   `alert_threshold`. Two spurious frames never reach five hits on one track.

Plus `merge_concurrent`: four people walking together are one incident and get
one clip, not four near-identical ones.

Threat score = `0.60·armed_ratio + 0.20·person_conf + 0.10·persistence + 0.10·night`.
Weights are in `config.yaml` because the right values are site-specific.

---

## Alert output

Each incident produces, in `alerts/`:

- `alert_<timestamp>_track<id>.mp4` — 20 s, of which the first 5 s is
  **pre-roll** from a ring buffer. Without pre-roll the clip starts after the
  system became confident, by which time the subject is just standing there.
- `alert_....json` — score, hit counts, boxes, weapon labels, merged subjects.
- `events.jsonl` — one line per alert, appended across runs.

---

## Files

```
config.yaml                          every threshold, nothing hard-coded
src/run_video.py                     entry point
src/detector.py                      two-stage detection + crop remapping
src/motion.py                        MOG2 motion gate
src/evidence.py                      clustering, tracking, threat scoring
src/recorder.py                      ring buffer + alert clip writer
src/viz.py                           overlays
src/train_weapon.py                  stage-2 fine-tuning (resumable)
src/eval_false_positives.py          FP rate + threshold sweep on unarmed footage
src/datasets/download.py             dataset fetch
src/datasets/build_crop_dataset.py   person-crop dataset builder
notebooks/train_weapon_colab.ipynb   Colab free-tier training, Drive checkpoints
tools/make_test_video.py             synthetic test clip
tools/build_notebook.py              regenerates the notebook
tests/test_pipeline.py               12 tests
```

---

## Tuning order when you get real footage

1. **`person_conf`** — get stage 1 catching everyone first. Missed humans are
   invisible to everything downstream.
2. **`motion.min_area_ratio`** — check the skip percentage in the run summary.
   If a human ever walks through without waking the gate, lower it immediately;
   a wasted detector pass costs milliseconds, a missed poacher costs a rhino.
3. **`weapon_conf`** — deliberately low (0.30). The clustering layer, not this
   threshold, is what filters false positives.
4. **`min_armed_hits` / `window`** — the real false-alert dial. Raise if you get
   noise, lower if armed subjects pass through too fast to accumulate hits.
5. **`crop.margin_x`** — 0.45 is set wide because a carried long gun extends
   well past the person box. If your weapons are handguns, tighten it.

---

## Testing

```bash
python tests/test_pipeline.py        # or: python -m pytest tests -q
```

Covers IoU and box-expansion geometry, crop-coordinate remapping against
hand-computed values, negative-sample retention, alert promotion, sporadic-hit
rejection, cooldown, night scoring, motion gate open/hold/close, and clip
length including pre-roll.

These exist because this system's failure modes are silent: a coordinate remap
off by the crop origin still yields plausible boxes, and a clustering rule that
never promotes runs cleanly and just never alerts. Both would look fine in a
demo.

---

## Known limitations

- No field validation; no accuracy numbers.
- Domain gap between public handgun/CCTV data and real poaching imagery (§3).
- No IR/night-vision handling — public datasets are daylight RGB. IR footage
  will likely need its own fine-tuning set.
- Occluded or shouldered weapons are the dominant expected failure mode.
- The IoU tracker is deliberately simple; it will swap IDs when people cross.
  Ultralytics' ByteTrack is the upgrade path if that matters.
- Single-process, synchronous. Fine for files; a live multi-camera deployment
  needs a queue.

## Licence and attribution

This project is licensed **AGPL-3.0** — see `LICENSE`.

That is not a free choice. This code depends on the `ultralytics` package, which
is AGPL-3.0, and Ultralytics' [licensing page](https://www.ultralytics.com/license)
states that publishing a project built on it requires you to "publicly release
the complete corresponding source code for the entire derivative work." They
also offer a paid Enterprise licence for anyone who needs to avoid that — which
is the route to look at if this is ever commercialised or embedded in a product.
[Unverified] That is Ultralytics' stated position as of August 2026; I am not a
lawyer and this is not legal advice.


Dataset terms are separate from the code licence and travel with the data:

- Roboflow *Person-Gun Detection-1* — **CC BY 4.0**: attribution required.
- OD-WeaponDetection / Sohas — **CC BY-SA 4.0**: attribution **and** share-alike.

Neither dataset is redistributed here; both are downloaded at build time.
