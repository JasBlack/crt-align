"""Benchmark the previous Python algorithm (crt_align.pin_align) on the synthetic set.

Usage: python run_old.py SYNTH_DIR
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import crt_align  # noqa: E402


def stats(a):
    a = np.sort(np.asarray(a))
    return {"median": float(a[len(a) // 2]), "p90": float(a[int(0.9 * len(a))]), "max": float(a[-1])}


def main():
    d = sys.argv[1]
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(d, "*_meta.json"))):
        name = os.path.basename(meta_path)[:-len("_meta.json")]
        meta = json.load(open(meta_path))
        ref = cv2.imread(os.path.join(d, f"{name}_orig.png"), cv2.IMREAD_COLOR)
        mov = cv2.imread(os.path.join(d, f"{name}_film.jpg"), cv2.IMREAD_COLOR)
        truth = np.load(os.path.join(d, f"{name}_truth.npy"))
        t0 = time.time()
        try:
            _, (mx, my), _ = crt_align.pin_align(ref, mov, verbose=False)
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s} FAILED: {e}")
            continue
        secs = time.time() - t0
        W, H, g = meta["W"], meta["H"], meta["grid"]
        th, tw = truth.shape[:2]
        errs, errs_all, errs_core = [], [], []
        for gy in range(th):
            for gx in range(tw):
                x, y = gx * g, gy * g
                u, v = (x + 0.5) / W, (y + 0.5) / H
                gx2 = min(tw - 1, gx + 1)
                sc = np.hypot(*(truth[gy, gx2] - truth[gy, gx])) / g or 1.0
                e = float(np.hypot(mx[y, x] - truth[gy, gx, 0], my[y, x] - truth[gy, gx, 1]) / sc)
                errs_all.append(e)
                if 0.1 < u < 0.9 and 0.1 < v < 0.9:
                    errs.append(e)
                if 0.2 < u < 0.8 and 0.2 < v < 0.8:
                    errs_core.append(e)
        si, sc = stats(errs), stats(errs_core)
        rows.append({"interior": si, "core": sc})
        print(f"{name:18s} core med {sc['median']:.2f} p90 {sc['p90']:.2f} max {sc['max']:.1f} | "
              f"10-90% med {si['median']:.2f} p90 {si['p90']:.2f} | {secs:.1f}s", flush=True)
    for reg in ("core", "interior"):
        med = lambda k: float(np.median([r[reg][k] for r in rows]))  # noqa: E731
        print(f"OLD {reg:8s}: median {med('median'):.2f}px, median p90 {med('p90'):.2f}px, "
              f"worst p90 {max(r[reg]['p90'] for r in rows):.2f}px, worst max {max(r[reg]['max'] for r in rows):.1f}px")

if __name__ == "__main__":
    main()
