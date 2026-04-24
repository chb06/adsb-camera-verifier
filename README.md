# ADS-B + camera passive aircraft verification (Mac-first conservative scaffold)

This project is a **careful, incremental scaffold** for collecting synchronized camera + ADS-B sessions and then reviewing them offline.

It keeps the parts that are already working:

- **Mac development first**
- **Arducam B0498 (USB3 8.3MP)** camera path preserved
- **OpenCV AVFoundation Mac camera path preserved**
- **ADS-B SBS/BaseStation ingest** preserved
- **Raspberry Pi feed at `adsbexchange.local:30003`** preserved
- **Audio remains disabled by default**
- **Jetson stays a later deployment target**

The priority in this update is not fancy architecture. The priority is a **stable path to collecting usable synchronized sessions**, replaying them, and starting offline YOLO experiments safely.

---

## What is fully implemented now

### Working / practical now

- Mac camera smoke test
- ADS-B smoke test
- Realtime run with the working Mac camera override config
- Session logging with:
  - raw recorded video
  - frame timestamp index
  - ADS-B JSONL log
  - per-frame projected track / ROI log
  - ROI image crops
  - run metadata and summary
- Replay tool for recorded sessions
- Offline detection runner for recorded sessions
- Dataset manifest export for review / labeling prep
- Intrinsics checkerboard calibration helper
- Plain-English config warnings when site / calibration are still placeholders

### Still scaffold / future work

- Real camera/site calibration refinement beyond the current practical helpers
- Extrinsics fitting from labeled aircraft pixels to ADS-B
- TensorRT runtime on Jetson
- Mature live realtime verification logic
- Production-quality aircraft detector trained on your site data
- Threshold sweep / full evaluation framework
- Audio fusion (intentionally still secondary / disabled)

---

## Important safety and scope notes

This project is for **passive verification only**.

Included:
- passive reception from an existing ADS-B decoder feed
- local logging
- offline replay
- offline detector experiments

Not included:
- RF transmit
- spoofing
- jamming
- interference

---

## Quick start on your Mac

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you want to run offline YOLO on recorded sessions, also install Ultralytics in this same environment:

```bash
pip install ultralytics
```

If you want the alternate FFmpeg AVFoundation camera path on macOS, install FFmpeg:

```bash
brew install ffmpeg
```

---

## The exact Mac commands to use

### 1) List cameras on macOS

```bash
python3 -m app.list_cameras
```

### 2) Camera smoke test using the Mac default path

```bash
python3 -m app.smoke_camera \
  --source 0 \
  --mode 720p15 \
  --display \
  --seconds 60
```

On macOS this uses the OpenCV AVFoundation path. If the Arducam works in QuickTime but not from the terminal, check that Terminal or iTerm has Camera permission in macOS Privacy & Security settings.

### 3) ADS-B smoke test using your Pi feed

```bash
python3 -m app.smoke_adsb --host adsbexchange.local --port 30003 --seconds 10
```

### 4) Realtime run using the existing Mac override

This preserves the working camera and ADS-B settings and starts logging a session into `runs/`:

```bash
python3 -m app.run_realtime \
  --config configs/default.yaml \
  --override configs/mac_dev.yaml \
  --seconds 60
```

### 5) Replay a recorded session

```bash
python -m app.run_replay_eval --run-dir runs/run_YYYYMMDD_HHMMSS --display
```

Useful replay options:

```bash
python -m app.run_replay_eval \
  --run-dir runs/run_YYYYMMDD_HHMMSS \
  --display \
  --speed 1.0 \
  --summary-interval-s 2.0
```

### 6) Run offline YOLO on a recorded session

Start with ROI-guided offline detection on a recorded run:

```bash
python -m app.run_offline_detection \
  --run-dir runs/run_YYYYMMDD_HHMMSS \
  --mode roi \
  --backend ultralytics \
  --model-path /path/to/your_model.pt \
  --allow-class airplane \
  --save-preview-video
```

If you only want to test the pipeline without a real model yet:

```bash
python -m app.run_offline_detection \
  --run-dir runs/run_YYYYMMDD_HHMMSS \
  --mode roi \
  --backend none
```

### 7) Build a review / labeling manifest from a run

```bash
python scripts/dataset_make_manifest.py --run-dir runs/run_YYYYMMDD_HHMMSS
```

This writes a CSV to:

```text
runs/run_YYYYMMDD_HHMMSS/exports/dataset_manifest.csv
```

---

## Session folder layout

Each logged run now contains a more useful, inspectable structure:

```text
runs/run_YYYYMMDD_HHMMSS/
  metadata.json
  run_summary.json
  video.mp4
  frame_index.jsonl
  adsb.jsonl
  track_frames.jsonl
  roi_index.jsonl
  decisions.jsonl
  audio_scores.jsonl
  detections/
  exports/
  roi/
```

### What each file means

- `video.mp4` — the raw recorded camera video used for replay
- `frame_index.jsonl` — timestamp for every saved frame
- `adsb.jsonl` — raw ADS-B states with receive timestamps
- `track_frames.jsonl` — per-frame projected track/ROI records, range, ROI box, and any live detections
- `roi_index.jsonl` — index of saved ROI crop images
- `metadata.json` — config used for the run
- `run_summary.json` — counts and session duration
- `detections/*.jsonl` — offline detector outputs
- `exports/dataset_manifest.csv` — review / labeling manifest from the helper script

This is the main improvement in this conservative update: recorded sessions are now much more useful for later replay, debugging, and dataset building.

---

## Replay and inspection

The replay tool now does more than just count files.

It can now:

- open the saved video
- sync it with the saved frame timestamps
- show projected ROI boxes from `track_frames.jsonl`
- show offline detections if you have already run `app.run_offline_detection`
- print a simple ADS-B summary during playback

Keyboard controls in replay window:

- `q` = quit
- `space` = pause / resume
- `n` = step one frame while paused

---

## YOLO plan for passive aircraft verification

Do **not** start by trying to build the final Jetson model right away.

Use this order instead:

### Stage 1 — prove the offline pipeline

- Record several short real sessions
- Run `app.run_offline_detection` on those sessions
- Use any reasonable Ultralytics `.pt` model first, just to prove the workflow
- Expect general-purpose models to be weak on small / distant aircraft

### Stage 2 — build your site-specific training set

Use your recorded sessions to collect:

- ROI crops from projected ADS-B tracks
- frame timestamps
- ADS-B association (`icao24`, range, time)
- offline detections and confidence

Then review those ROI images and label the true aircraft examples.

### Stage 3 — train a simple custom model

For the first custom model, keep it small and simple:

- one class: `aircraft`
- train on your own site data
- focus on the camera angle and aircraft sizes you really see
- use recorded ROI crops and difficult negatives from your site

### Stage 4 — only then think about Jetson deployment

Once the offline detector is useful on recorded sessions:

- export ONNX
- build TensorRT later on the Jetson
- keep the Mac workflow as the safe development path until the data pipeline is solid

---

## What data you should collect next

For the next few sessions, collect boring, clean data rather than trying to be clever.

Collect:

1. **Daytime sessions** with good visibility
2. **10 to 20 minute runs** pointing at a known traffic direction
3. Sessions where aircraft move across the field of view slowly enough to inspect
4. Runs with your real site position filled in
5. A few checkerboard photos for camera intrinsics calibration

Try to avoid:

- camera movement during a session
- changing camera source / backend while testing
- mixing too many experiments in one run

---

## Physical things you need to measure or write down

You do not need survey-grade precision yet.

Write down these five things:

1. camera latitude / longitude
2. camera altitude (or approximate elevation)
3. camera mounting height above ground
4. camera facing direction (rough compass heading / yaw)
5. whether the camera is tilted up or down (pitch)

If the horizon looks slanted, also note camera roll.

---

## Calibration workflow

### Intrinsics first

Take 20 to 40 checkerboard photos with the full board visible at different angles and distances, then run:

```bash
python -m app.calibrate_intrinsics \
  --images /path/to/checkerboard_images \
  --cols 9 \
  --rows 6 \
  --square-mm 25
```

Paste the printed `fx`, `fy`, `cx`, `cy`, and `dist` values into your config.

### Extrinsics later

For now, start with rough yaw / pitch / roll estimates.

The code now warns you when your site / calibration values are still placeholders. That is intentional so you know why ADS-B ROI alignment may still be rough.

---

## Config notes for your setup

`configs/mac_dev.yaml` preserves the Mac path:

- camera backend: `opencv`
- camera source: `0` (the Arducam index seen in `python3 -m app.list_cameras`)
- ADS-B host: `adsbexchange.local`
- ADS-B port: `30003`
- audio disabled

If you need to add your real site position and rough camera pointing, copy the values from `configs/site_template.yaml` into your own Mac override file.

---

## Beginner-friendly workflow for the next few days

### First goal

Make sure these three commands work on your Mac:

```bash
python3 -m app.smoke_camera --source 0 --mode 720p15 --display --seconds 20
python3 -m app.smoke_adsb --host adsbexchange.local --port 30003 --seconds 10
python3 -m app.run_realtime --config configs/default.yaml --override configs/mac_dev.yaml --seconds 60
```

### Second goal

After one session is recorded:

```bash
python -m app.run_replay_eval --run-dir runs/run_YYYYMMDD_HHMMSS --display
```

### Third goal

Once you install Ultralytics and have a `.pt` model to try:

```bash
python -m app.run_offline_detection \
  --run-dir runs/run_YYYYMMDD_HHMMSS \
  --mode roi \
  --backend ultralytics \
  --model-path /path/to/your_model.pt \
  --allow-class airplane
```

### Fourth goal

Build the manifest you will use for review / labeling:

```bash
python scripts/dataset_make_manifest.py --run-dir runs/run_YYYYMMDD_HHMMSS
```

---

## Known limitations in this conservative update

- The live realtime path is still mainly a logging / inspection tool, not a final verifier.
- General-purpose YOLO models will probably miss small distant aircraft at your site.
- Accurate ADS-B projection depends heavily on correct site location and camera pointing.
- TensorRT / Jetson deployment is intentionally not the focus yet.

---

## Recommended next milestone

The next milestone is:

**collect 5 to 10 good synchronized sessions, replay them, run offline YOLO, and build the first review manifest**.

That gives you the data you need to decide whether to:

- improve calibration first,
- improve the ROI sizing / projection first, or
- start fine-tuning a custom aircraft model.
