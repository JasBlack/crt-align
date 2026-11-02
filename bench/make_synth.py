"""Synthetic CRT-capture benchmark with known ground truth.

For each original image, simulate filming a glitched CRT:
  original -> raster (overscan zoom, anisotropic scale, small rotation, offset)
           -> CRT glass barrel distortion
           -> camera perspective into a 4000x3000 photo with dark bezel
           -> glitch: false colour, emboss mix, horizontal line jitter,
              scanlines, blur, noise

Writes per case:
  <case>_orig.png, <case>_film.jpg      the input pair
  <case>_truth.npy                      float32 (gh, gw, 2): on a 1/8-res grid of
                                        ORIGINAL pixel centres, where that content
                                        lands in the filmed photo
  <case>_meta.json                      grid spacing + true raster corners

Usage: python make_synth.py OUT_DIR
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCES = ["base_26.jpg", "base_29.jpg", "base_49.jpg", "base_68.jpg", "new_00367_.png"]
FILM_W, FILM_H = 4000, 3000
GRID = 8


class Geometry:
    """original px -> screen (s,t) -> barrel -> camera photo px, with exact inverse."""

    def __init__(self, W, H, rng, severity):
        self.W, self.H = W, H
        self.zx = rng.uniform(1.06, 1.15)
        self.zy = rng.uniform(1.06, 1.15)
        self.rot = np.deg2rad(rng.uniform(-1.0, 1.0))
        self.off = rng.uniform(-0.02, 0.02, size=2)
        self.barrel = rng.uniform(0.04, 0.10)
        cx = FILM_W / 2 + rng.uniform(-150, 150)
        cy = FILM_H / 2 + rng.uniform(-120, 120)
        sw, sh = rng.uniform(2600, 3200), rng.uniform(2000, 2400)
        j = 60 + 120 * severity
        quad = np.float32([
            [cx - sw / 2 + rng.uniform(-j, j), cy - sh / 2 + rng.uniform(-j, j)],
            [cx + sw / 2 + rng.uniform(-j, j), cy - sh / 2 + rng.uniform(-j, j)],
            [cx + sw / 2 + rng.uniform(-j, j), cy + sh / 2 + rng.uniform(-j, j)],
            [cx - sw / 2 + rng.uniform(-j, j), cy + sh / 2 + rng.uniform(-j, j)],
        ])
        unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.Hc = cv2.getPerspectiveTransform(unit, quad).astype(np.float64)
        self.Hci = np.linalg.inv(self.Hc)

    # --- original normalized (u,v) <-> screen (s,t) (affine) ---
    def to_screen(self, u, v):
        c, s = np.cos(self.rot), np.sin(self.rot)
        du, dv = u - 0.5, v - 0.5
        return (0.5 + (c * du - s * dv) * self.zx + self.off[0],
                0.5 + (s * du + c * dv) * self.zy + self.off[1])

    def from_screen(self, s_, t_):
        c, s = np.cos(self.rot), np.sin(self.rot)
        a = (s_ - 0.5 - self.off[0]) / self.zx
        b = (t_ - 0.5 - self.off[1]) / self.zy
        return 0.5 + c * a + s * b, 0.5 - s * a + c * b

    # --- CRT glass barrel ---
    def _k(self, x, y):
        return self.barrel * (1 - 0.25 * (x * x + y * y))

    def barrel_fwd(self, s, t):
        x, y = 2 * s - 1, 2 * t - 1
        k = self._k(x, y)
        return 0.5 + 0.5 * x * (1 + 0.9 * k * (1 - y * y)), 0.5 + 0.5 * y * (1 + 0.9 * k * (1 - x * x))

    def barrel_inv(self, bs, bt):
        xp, yp = 2 * bs - 1, 2 * bt - 1
        x, y = xp.copy(), yp.copy()
        for _ in range(25):
            k = self._k(x, y)
            x = xp / (1 + 0.9 * k * (1 - y * y))
            y = yp / (1 + 0.9 * k * (1 - x * x))
        return 0.5 + 0.5 * x, 0.5 + 0.5 * y

    # --- camera homography ---
    def cam_fwd(self, s, t):
        Hm = self.Hc
        d = Hm[2, 0] * s + Hm[2, 1] * t + Hm[2, 2]
        return (Hm[0, 0] * s + Hm[0, 1] * t + Hm[0, 2]) / d, (Hm[1, 0] * s + Hm[1, 1] * t + Hm[1, 2]) / d

    def cam_inv(self, X, Y):
        Hm = self.Hci
        d = Hm[2, 0] * X + Hm[2, 1] * Y + Hm[2, 2]
        return (Hm[0, 0] * X + Hm[0, 1] * Y + Hm[0, 2]) / d, (Hm[1, 0] * X + Hm[1, 1] * Y + Hm[1, 2]) / d

    def orig_to_film(self, x, y):
        s, t = self.to_screen((x + 0.5) / self.W, (y + 0.5) / self.H)
        return self.cam_fwd(*self.barrel_fwd(s, t))

    def film_to_screen(self, X, Y):
        """photo px -> pre-barrel screen (s,t)."""
        return self.barrel_inv(*self.cam_inv(X, Y))


def glitch(img, rng, severity):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    M = np.eye(3) * rng.uniform(0.3, 0.9) + rng.uniform(-0.5, 0.8, (3, 3)) * severity
    out = np.clip(out @ M.T.astype(np.float32), 0, 255)
    if rng.random() < 0.5 * severity + 0.2:
        ch = int(rng.integers(0, 3))
        out[..., ch] = 255 - out[..., ch]
    gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    emb = cv2.filter2D(gray, -1, np.float32([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]))
    mix = 0.35 + 0.4 * severity
    out = out * (1 - mix) + (emb[..., None] * 1.2 + 128) * mix
    # horizontal line jitter (time-base wobble), smooth in y; zero-mean glitch
    jit = cv2.GaussianBlur(rng.normal(0, 1, (h, 1)).astype(np.float32), (1, 0), sigmaX=0,
                           sigmaY=6 * h / 1000).ravel()
    jit = jit / (np.abs(jit).max() + 1e-6) * (2 + 5 * severity) * w / 1000
    if os.environ.get("NOJITTER"):
        jit[:] = 0
    xs = np.arange(w, dtype=np.float32)[None, :] + jit[:, None]
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    out = cv2.remap(np.clip(out, 0, 255).astype(np.uint8), xs, ys, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    return out


def make_case(name, src_path, out_dir, rng, severity):
    orig = cv2.imread(src_path, cv2.IMREAD_COLOR)
    H, W = orig.shape[:2]
    g = Geometry(W, H, rng, severity)
    glitched = glitch(orig, rng, severity)

    # render filmed photo: per photo pixel, invert to original px
    Y, X = np.mgrid[0:FILM_H, 0:FILM_W].astype(np.float64)
    s, t = g.film_to_screen(X, Y)
    inside = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    u, v = g.from_screen(s, t)
    mx = (u * W - 0.5).astype(np.float32)
    my = (v * H - 0.5).astype(np.float32)
    del X, Y, u, v
    film = cv2.remap(glitched, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0).astype(np.float32)
    del mx, my

    # raster mask with rounded corners, soft edge; dark bezel outside
    m = inside.astype(np.uint8) * 255
    del inside, s, t
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    m = cv2.dilate(cv2.erode(m, k), k)
    rm = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 2.5)[..., None]
    bezel = 16 + rng.normal(0, 3, film.shape).astype(np.float32)
    film = film * rm + bezel * (1 - rm)
    del rm, bezel

    yy = np.arange(FILM_H, dtype=np.float32)[:, None, None]
    film *= 0.80 + 0.20 * np.cos(yy * 2 * np.pi / rng.uniform(5.5, 7.5))
    film = cv2.GaussianBlur(film, (0, 0), 1.2)
    film += rng.normal(0, 5, film.shape).astype(np.float32)
    film = np.clip(film, 0, 255).astype(np.uint8)

    # truth on a coarse grid of original pixel centres
    gy, gx = np.mgrid[0:H:GRID, 0:W:GRID].astype(np.float64)
    TX, TY = g.orig_to_film(gx, gy)
    truth = np.stack([TX, TY], -1).astype(np.float32)
    corners = [list(map(float, g.cam_fwd(*g.barrel_fwd(np.float64(a), np.float64(b)))))
               for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]]

    cv2.imwrite(os.path.join(out_dir, f"{name}_orig.png"), orig)
    cv2.imwrite(os.path.join(out_dir, f"{name}_film.jpg"), film, [cv2.IMWRITE_JPEG_QUALITY, 90])
    np.save(os.path.join(out_dir, f"{name}_truth.npy"), truth)
    with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as fh:
        json.dump({"W": W, "H": H, "grid": GRID, "corners": corners,
                   "zoom": [g.zx, g.zy], "severity": severity}, fh)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(1234)
    only = os.environ.get("ONLY")
    for sev_name, sev in (("mild", 0.3), ("heavy", 1.0)):
        for src in SOURCES:
            if only and only not in f"{src}_{sev_name}":
                rng.random(64)
                continue
            name = f"{os.path.splitext(src)[0]}_{sev_name}"
            make_case(name, os.path.join(ROOT, src), out_dir, rng, sev)
            print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
