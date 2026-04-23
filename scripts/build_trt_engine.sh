#!/usr/bin/env bash
set -euo pipefail

# Build a TensorRT engine from an ONNX model.
#
# Usage:
#   ./scripts/build_trt_engine.sh yolov8s_640.onnx yolov8s_640_fp16.engine
#
# Notes:
# - Run this ON THE JETSON (engine is not portable across TRT major versions).
# - This uses FP16. Add --int8 later once you have calibration.

ONNX=${1:-""}
ENGINE=${2:-""}

if [[ -z "$ONNX" || -z "$ENGINE" ]]; then
  echo "Usage: $0 <model.onnx> <out.engine>"
  exit 1
fi

# Adjust input tensor name if your ONNX exporter uses a different name.
# Ultralytics often uses 'images'.
INPUT_NAME=${INPUT_NAME:-images}

trtexec \
  --onnx="$ONNX" \
  --fp16 \
  --saveEngine="$ENGINE" \
  --minShapes=${INPUT_NAME}:1x3x640x640 \
  --optShapes=${INPUT_NAME}:4x3x640x640 \
  --maxShapes=${INPUT_NAME}:8x3x640x640
