"""Convert synthetic cases to raw float32 luma + truth for the Node benchmark.

The original is stored at <= 1536 px wide and the filmed photo at <= 2048 px wide,
which is what the browser app decodes for estimation.

Usage: python prep_raw.py SYNTH_DIR
"""
import glob
import json
import os
import sys

import cv2
import numpy as np


def luma(path, max_w):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    if W > max_w:
        img = cv2.resize(img, (max_w, round(H * max_w / W)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g, W, H


def main():
    d = sys.argv[1]
    for meta_path in sorted(glob.glob(os.path.join(d, "*_meta.json"))):
        name = os.path.basename(meta_path)[:-len("_meta.json")]
        meta = json.load(open(meta_path))
        o, W, H = luma(os.path.join(d, f"{name}_orig.png"), 1536)
        f, FW, FH = luma(os.path.join(d, f"{name}_film.jpg"), 2048)
        truth = np.load(os.path.join(d, f"{name}_truth.npy")).astype(np.float32)
        o.tofile(os.path.join(d, f"{name}_orig.f32"))
        f.tofile(os.path.join(d, f"{name}_film.f32"))
        truth.tofile(os.path.join(d, f"{name}_truth.f32"))
        meta.update({"W": W, "H": H, "FW": FW, "FH": FH,
                     "ow": o.shape[1], "oh": o.shape[0], "fw": f.shape[1], "fh": f.shape[0],
                     "tw": truth.shape[1], "th": truth.shape[0]})
        json.dump(meta, open(meta_path, "w"))
        print("prepped", name)


if __name__ == "__main__":
    main()
