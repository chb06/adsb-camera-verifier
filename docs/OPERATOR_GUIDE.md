# Operator guide (plain English)

## Before you start

Make sure these two physical things are true:

- The Arducam is plugged into your Mac.
- The Raspberry Pi ADS-B feed is reachable at `adsbexchange.local:30003`.

## The safest way to work

1. Do **not** change the camera backend unless you have to.
2. Keep using the default Mac path through OpenCV AVFoundation.
3. Record short sessions first.
4. Replay every session right away so you catch problems early.

## One simple working routine

### Step 1 — confirm the camera

```bash
python3 -m app.smoke_camera --source 0 --mode 720p15 --display --seconds 20
```

### Step 2 — confirm ADS-B

```bash
python3 -m app.smoke_adsb --host adsbexchange.local --port 30003 --seconds 10
```

### Step 3 — record one session

```bash
python3 -m app.run_realtime --config configs/default.yaml --override configs/mac_dev.yaml --seconds 60
```

### Step 4 — replay that same session

```bash
python3 -m app.run_replay_eval --run-dir runs/run_YYYYMMDD_HHMMSS --display
```

### Step 5 — create a review manifest

```bash
python scripts/dataset_make_manifest.py --run-dir runs/run_YYYYMMDD_HHMMSS
```

## What “good data” looks like

A good run is:

- camera view stable
- no source/backend changes during the run
- aircraft visible in the camera view
- ADS-B feed active for the whole run
- session replay opens without errors

## What to collect next

Collect a few short runs in:

- clear daytime sky
- the same camera position
- the same camera mode
- a known traffic direction if possible

## What to write down in a notebook or text file

- date and time of run
- approximate weather / visibility
- camera direction
- any obvious aircraft seen during the run
- anything unusual, like dropped frames or unplug/replug events
