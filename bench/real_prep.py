"""Prepare real original/filmed pairs as raw luma for the Node engine.
Usage: python real_prep.py OUT_DIR name=orig.jpg,filmed.jpg [...]"""
import json, os, sys
import cv2
import numpy as np

def luma(path, max_w):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    if W > max_w:
        img = cv2.resize(img, (max_w, round(H * max_w / W)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, W, H

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for arg in sys.argv[2:]:
    name, pair = arg.split("=")
    o_path, f_path = [os.path.join(root, p) for p in pair.split(",")]
    o, W, H = luma(o_path, 1536)
    f, FW, FH = luma(f_path, 2048)
    o.tofile(os.path.join(out, f"{name}_orig.f32"))
    f.tofile(os.path.join(out, f"{name}_film.f32"))
    json.dump({"W": W, "H": H, "FW": FW, "FH": FH, "ow": o.shape[1], "oh": o.shape[0],
               "fw": f.shape[1], "fh": f.shape[0], "orig": o_path, "film": f_path},
              open(os.path.join(out, f"{name}_meta.json"), "w"))
    print("prepped", name)
