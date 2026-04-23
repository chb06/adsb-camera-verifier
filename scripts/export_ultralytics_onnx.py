from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='trained .pt file')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--outdir', default='exports')
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(
            'ultralytics is not installed. Install it in your training environment: pip install ultralytics'
        ) from e

    m = YOLO(args.weights)
    m.export(format='onnx', imgsz=args.imgsz, opset=13, simplify=True, dynamic=True, half=True, device='cpu', project=args.outdir)


if __name__ == '__main__':
    main()
