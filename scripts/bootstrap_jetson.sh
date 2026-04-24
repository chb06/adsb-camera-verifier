#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] Installing system packages (Jetson / Ubuntu)..."

sudo apt update

# Core tools
sudo apt install -y \
  git \
  python3-venv python3-pip \
  build-essential pkg-config \
  curl ca-certificates \
  v4l-utils \
  ffmpeg

# OpenCV python bindings (use Ubuntu package on Jetson)
sudo apt install -y python3-opencv

# GStreamer runtime utilities and common plugins
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav

# Audio (PortAudio for sounddevice)
sudo apt install -y portaudio19-dev

echo "[bootstrap] Done. Next: create venv and pip install -r requirements.txt"
