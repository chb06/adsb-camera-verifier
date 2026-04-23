# Conservative update changelog

## Main goal of this update

Keep the existing working Mac camera path and ADS-B ingest path intact, while making recorded sessions much more useful for replay, debugging, and dataset building.

## High-value changes

- added synchronized `frame_index.jsonl`
- added `track_frames.jsonl` for per-frame ADS-B / ROI association
- improved run summaries with `run_summary.json`
- preserved raw video logging while keeping display overlays separate
- implemented replay tooling in `app.run_replay_eval`
- implemented offline detection in `app.run_offline_detection`
- upgraded `scripts/dataset_make_manifest.py`
- added beginner-facing docs and config warnings
- fixed `app.smoke_adsb` timeout behavior
- fixed `CameraStream.read()` so OpenCV video files work correctly in the same API as the ffmpeg path

## Files changed

- `README.md`
- `CHANGELOG.md`
- `docs/OPERATOR_GUIDE.md`
- `configs/site_template.yaml`
- `app/run_realtime.py`
- `app/run_replay_eval.py`
- `app/run_offline_detection.py`
- `app/smoke_adsb.py`
- `data/logger.py`
- `data/replay.py`
- `geo/projection.py`
- `geo/config_checks.py`
- `scripts/dataset_make_manifest.py`
- `sensors/camera_capture.py`
- `tests/test_projection.py`
- `tests/test_session_pipeline.py`
