"""
gen_test_frames.py  —  Genere des frames synthetiques pour tester le codec
Usage : python gen_test_frames.py -o sample_frames -n 12
"""
import argparse
import os
import cv2
import numpy as np


def gen_frames(out_dir, n=12, h=128, w=128):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Fond degrade
        for y in range(h):
            frame[y, :] = [int(y * 0.6), int(y * 0.4), 60]
        # Cercle qui se deplace
        cx = 20 + (i * (w - 50)) // max(1, n - 1)
        cy = h // 2 + int(20 * np.sin(i * 0.5))
        cv2.circle(frame, (cx, cy), 18, (200, 80, 50), -1)
        # Rectangle fixe
        cv2.rectangle(frame, (w//2, h//3), (w//2+30, h//3+30), (50, 180, 200), -1)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), frame)
    print(f"{n} frames generees dans '{out_dir}' ({w}x{h})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="sample_frames")
    ap.add_argument("-n", "--num",    type=int, default=12)
    ap.add_argument("--height",       type=int, default=128)
    ap.add_argument("--width",        type=int, default=128)
    args = ap.parse_args()
    gen_frames(args.output, args.num, args.height, args.width)
