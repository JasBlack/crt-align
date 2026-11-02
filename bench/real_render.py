"""Render aligned previews + QC overlays from real_align.mjs maps.
Usage: python real_render.py DIR"""
import glob, json, os, sys
import cv2
import numpy as np
d = sys.argv[1]
for mp in sorted(glob.glob(os.path.join(d, "*_model.json"))):
    name = os.path.basename(mp)[:-len("_model.json")]
    meta = json.load(open(os.path.join(d, f"{name}_meta.json")))
    mdl = json.load(open(mp))
    pw, ph = mdl["pw"], mdl["ph"]
    m = np.fromfile(os.path.join(d, f"{name}_map.f32"), np.float32).reshape(ph, pw, 2)
    film = cv2.imread(meta["film"], cv2.IMREAD_COLOR)
    orig = cv2.resize(cv2.imread(meta["orig"], cv2.IMREAD_COLOR), (pw, ph), interpolation=cv2.INTER_AREA)
    aligned = cv2.remap(film, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(d, f"{name}_aligned.jpg"), aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # QC: original edges (cyan) drawn over the aligned glitch layer
    g = cv2.GaussianBlur(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    e = cv2.Canny(g, 40, 110)
    qc = aligned.copy()
    qc[e > 0] = (255, 255, 0)
    tile = max(32, min(ph, pw) // 10)
    yy, xx = np.mgrid[0:ph, 0:pw]
    checker = (((yy // tile) + (xx // tile)) % 2 == 0)[..., None]
    board = np.where(checker, orig, aligned)
    cv2.imwrite(os.path.join(d, f"{name}_qc.jpg"), np.hstack([qc, board]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("rendered", name)
