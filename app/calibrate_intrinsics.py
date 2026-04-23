from __future__ import annotations

import argparse
from pathlib import Path

from geo.calibration_tools import calibrate_intrinsics_checkerboard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='folder with checkerboard images (jpg/png)')
    ap.add_argument('--cols', type=int, required=True, help='checkerboard inner corners (columns)')
    ap.add_argument('--rows', type=int, required=True, help='checkerboard inner corners (rows)')
    ap.add_argument('--square-mm', type=float, required=True, help='square size in millimeters')
    args = ap.parse_args()

    img_dir = Path(args.images)
    paths = sorted([p for p in img_dir.glob('*.jpg')] + [p for p in img_dir.glob('*.png')])
    if not paths:
        raise SystemExit(f"No jpg/png images found in {img_dir}")

    res = calibrate_intrinsics_checkerboard(
        image_paths=paths,
        checkerboard_size=(args.cols, args.rows),
        square_size_m=args.square_mm / 1000.0,
    )

    print("=== Intrinsics calibration result ===")
    print(f"Used images: {res.used_images}/{len(paths)}")
    print(f"Image size (w,h): {res.image_size}")
    print(f"RMS reprojection error: {res.rms_reproj_error:.3f} px")
    print("K:")
    print(res.K)
    print("dist (k1,k2,p1,p2,k3,...):")
    print(res.dist)

    print("\nPaste into configs/default.yaml under calibration:")
    print(f"  fx: {res.K[0,0]:.6f}")
    print(f"  fy: {res.K[1,1]:.6f}")
    print(f"  cx: {res.K[0,2]:.6f}")
    print(f"  cy: {res.K[1,2]:.6f}")
    print(f"  dist: [{', '.join(f'{x:.6g}' for x in res.dist[:5])}]")


if __name__ == '__main__':
    main()
