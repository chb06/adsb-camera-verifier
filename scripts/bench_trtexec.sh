#!/usr/bin/env bash
set -euo pipefail

# Benchmark an ONNX model or TensorRT engine.
#
# Examples:
#   ./scripts/bench_trtexec.sh --onnx yolov8s_640.onnx
#   ./scripts/bench_trtexec.sh --engine yolov8s_640_fp16.engine

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 --onnx <model.onnx> | --engine <model.engine>"
  exit 1
fi

MODE=$1
FILE=$2

if [[ "$MODE" == "--onnx" ]]; then
  trtexec --onnx="$FILE" --fp16 --shapes=images:1x3x640x640 --avgRuns=100 --warmUp=200
elif [[ "$MODE" == "--engine" ]]; then
  trtexec --loadEngine="$FILE" --avgRuns=200 --warmUp=200
else
  echo "Unknown mode: $MODE"
  exit 1
fi
