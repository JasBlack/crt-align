"""Synthetic handheld bursts with ground truth. For each source photo, N shots:
random camera rotation/scale/shift/perspective, exposure + white balance change,
noise, JPEG. Writes shot_k.jpg and truth.json (H_k maps shot pixels -> source pixels)."""
import cv2, numpy as np, json, sys, os
rng = np.random.default_rng(7)
def burst(src, out, n=6, W=2400, H=1800, hard=False):
    os.makedirs(out, exist_ok=True)
    img = cv2.imread(src); sh, sw = img.shape[:2]
    truth = {"source": os.path.basename(src), "sw": sw, "sh": sh, "w": W, "h": H, "shots": []}
    for k in range(n):
        ang = np.deg2rad(rng.uniform(-6, 6) * (2 if hard else 1))
        s = rng.uniform(0.95, 1.25) * min(sw / W, sh / H) * 0.9
        cx, cy = sw / 2 + rng.uniform(-0.08, 0.08) * sw, sh / 2 + rng.uniform(-0.08, 0.08) * sh
        # shot pixel (u,v) -> source: centre, rotate, scale, perspective
        T0 = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        S = np.diag([s, s, 1])
        p = (0.6 if hard else 0.25) * 1e-4
        P = np.array([[1, 0, 0], [0, 1, 0], [rng.uniform(-p, p), rng.uniform(-p, p), 1]])
        T1 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        Hm = T1 @ S @ R @ P @ T0
        Hm /= Hm[2, 2]
        shot = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        f = shot.astype(np.float32)
        f *= rng.uniform(0.75, 1.3); f *= rng.uniform(0.9, 1.1, 3)[None, None, :]
        f += rng.normal(0, 4, f.shape)
        cv2.imwrite(f"{out}/shot_{k}.jpg", np.clip(f, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
        truth["shots"].append({"file": f"shot_{k}.jpg", "H": Hm.tolist()})
    json.dump(truth, open(f"{out}/truth.json", "w"), indent=1)

def sun(out, n=6, W=2400, H=1600):
    """Sun moving across a sky over a horizon; camera also moves. Truth = sun position per shot."""
    os.makedirs(out, exist_ok=True); shots = []
    yy, xx = np.mgrid[0:H, 0:W]
    for k in range(n):
        dx, dy = rng.uniform(-200, 200), rng.uniform(-120, 120)
        sx, sy = W * (0.3 + 0.08 * k) + dx, H * (0.35 - 0.02 * k) + dy
        sky = np.dstack([90 + 60 * yy / H, 140 + 50 * yy / H, 230 - 20 * yy / H]).astype(np.float32)
        r = np.hypot(xx - sx, yy - sy)
        glow = np.exp(-r / 160)[..., None] * np.array([60, 160, 255]) + (r < 38)[..., None] * 255
        im = sky + glow
        hor = H * 0.72 + dy + 30 * np.sin(xx / 170.0)
        im[yy > hor] = (im[yy > hor] * 0.25 + np.array([20, 50, 30]))
        im += rng.normal(0, 3, im.shape)
        cv2.imwrite(f"{out}/sun_{k}.jpg", np.clip(im, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
        shots.append({"file": f"sun_{k}.jpg", "sun": [sx, sy]})
    json.dump({"w": W, "h": H, "shots": shots}, open(f"{out}/truth.json", "w"), indent=1)

R = sys.argv[1]; O = sys.argv[2]
burst(f"{R}/base_29.jpg", f"{O}/petronas")
burst(f"{R}/base_49.jpg", f"{O}/chicago")
burst(f"{R}/base_49.jpg", f"{O}/chicago_hard", hard=True)
burst(f"{R}/base_26.jpg", f"{O}/leopard")
sun(f"{O}/sun")
