"""Raw float32 luma (<= 2048 px long side, what the browser decodes for stack mode)
for every jpg/png in a folder, plus sizes in index.json. Usage: python prep.py DIR"""
import cv2, glob, json, os, sys
import numpy as np
d = sys.argv[1]; idx = []
for p in sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))):
    img = cv2.imread(p); H, W = img.shape[:2]; k = min(1, 2048 / max(W, H))
    if k < 1: img = cv2.resize(img, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g.tofile(p + ".f32"); idx.append({"file": os.path.basename(p), "W": W, "H": H, "w": g.shape[1], "h": g.shape[0]})
json.dump(idx, open(os.path.join(d, "index.json"), "w"))
print(d, len(idx))
