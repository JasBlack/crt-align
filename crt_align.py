#!/usr/bin/env python3
"""crt_align — align a filmed CRT capture (photo or video) to its source image/video.

The filmed capture is corner-pinned and warped so it sits on top of the original,
ready to use as a glitch layer in compositing.

Usage:
    python crt_align.py ORIGINAL FILMED [-o OUTPUT] [options]

    ORIGINAL  source image or video that was sent to the CRT
    FILMED    photo or video of the CRT screen showing (a glitched version of) it

Modes:
    pin (default)  crop into the CRT screen, pin its corners to the canvas corners,
                   then elastically warp the interior to match the original.
                   Full-bleed output, no black edges.
    exact          geometric mapping only: every filmed pixel goes exactly where its
                   content belongs. Leaves black where the camera saw nothing.

Examples:
    python crt_align.py base_26.jpg IDG_1234.JPG --overlay
    python crt_align.py clip.mp4 filmed.mp4 -o aligned.mp4
    python crt_align.py still.png filmed.mp4 -o layer.mp4

The output always has the exact pixel dimensions of ORIGINAL.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

MATCH_MAX_DIM = 1600      # working resolution for feature matching
FLOW_MAX_DIM = 1200       # working resolution for dense refinement
MIN_INLIERS = 12


def media_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Unrecognized media extension: {path}")


def scale_to_fit(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so max(h, w) <= max_dim. Returns (image, scale) with scale <= 1."""
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def prep_variants(gray: np.ndarray, blur_sigma: float) -> dict[str, np.ndarray]:
    """Different tonal renderings to give the matcher several chances on glitched footage.

    blur_sigma suppresses CRT scanline / moire texture before feature detection.
    """
    g = cv2.GaussianBlur(gray, (0, 0), blur_sigma) if blur_sigma > 0 else gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    mag = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(mag)
    return {"clahe": clahe, "gradient": mag}


def root_sift(des: np.ndarray | None) -> np.ndarray | None:
    """RootSIFT: Hellinger-kernel variant, markedly better matching than raw SIFT."""
    if des is None:
        return None
    des = des / (np.abs(des).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des)


def sane_homography(H: np.ndarray, mov_shape: tuple[int, int], ref_shape: tuple[int, int]) -> bool:
    """Reject degenerate/flipped/absurd homographies."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    h, w = mov_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    area = cv2.contourArea(warped.astype(np.float32), oriented=True)
    ref_area = ref_shape[0] * ref_shape[1]
    if area <= 0.05 * ref_area or area >= 20.0 * ref_area:
        return False
    if not cv2.isContourConvex(warped.astype(np.float32)):
        return False
    return True


def match_features(ref_gray: np.ndarray, mov_gray: np.ndarray, label: str, verbose: bool):
    """Try RootSIFT then ORB on one preprocessing variant. Returns (H, inliers) or (None, 0)."""
    best = (None, 0)
    for det_name in ("sift", "orb"):
        if det_name == "sift":
            det = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.01)
            norm = cv2.NORM_L2
        else:
            det = cv2.ORB_create(nfeatures=12000, fastThreshold=10)
            norm = cv2.NORM_HAMMING
        kp1, des1 = det.detectAndCompute(ref_gray, None)
        kp2, des2 = det.detectAndCompute(mov_gray, None)
        if det_name == "sift":
            des1, des2 = root_sift(des1), root_sift(des2)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            continue
        matcher = cv2.BFMatcher(norm)
        knn = matcher.knnMatch(des2, des1, k=2)  # query = moving, train = reference
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
        if len(good) < 8:
            if verbose:
                print(f"    [{label}/{det_name}] only {len(good)} ratio-test matches")
            continue
        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0,
                                     maxIters=10000, confidence=0.999)
        inliers = int(mask.sum()) if mask is not None else 0
        if verbose:
            print(f"    [{label}/{det_name}] {len(good)} matches, {inliers} MAGSAC inliers")
        if inliers > best[1] and inliers >= 8 and \
                sane_homography(H, mov_gray.shape[:2], ref_gray.shape[:2]):
            best = (H, inliers)
    return best


def feature_homography(ref_bgr: np.ndarray, mov_bgr: np.ndarray, verbose: bool = True):
    """Feature-based full-res homography FILMED -> ORIGINAL. Returns (H or None, inliers)."""
    ref_small, s_ref = scale_to_fit(ref_bgr, MATCH_MAX_DIM)
    mov_small, s_mov = scale_to_fit(mov_bgr, MATCH_MAX_DIM)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(mov_small, cv2.COLOR_BGR2GRAY)

    best = (None, 0)
    # filmed CRT footage carries scanline/moire texture -> heavier blur helps
    for blur_ref, blur_mov in ((1.0, 2.5), (0.0, 1.2)):
        ref_variants = prep_variants(ref_gray, blur_ref)
        mov_variants = prep_variants(mov_gray, blur_mov)
        for name in ref_variants:
            label = f"{name} blur={blur_mov}"
            H, inliers = match_features(ref_variants[name], mov_variants[name], label, verbose)
            if inliers > best[1]:
                best = (H, inliers)
        if best[1] >= 3 * MIN_INLIERS:
            break  # confident enough, skip further attempts
    H_small, inliers = best
    if H_small is None:
        return None, 0
    S_ref = np.diag([s_ref, s_ref, 1.0])
    S_mov = np.diag([s_mov, s_mov, 1.0])
    return np.linalg.inv(S_ref) @ H_small @ S_mov, inliers


def detect_screen_quad(mov_bgr: np.ndarray) -> np.ndarray | None:
    """Find the bright CRT raster against the dark surround. Returns 4x2 corners
    (TL, TR, BR, BL) in full-resolution filmed coords, or None."""
    small, s = scale_to_fit(mov_bgr, 1200)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.05 * small.shape[0] * small.shape[1]:
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.2, 40):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
    # order TL, TR, BR, BL
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    quad = quad[np.argsort(ang)]
    start = int(np.argmin(quad.sum(axis=1)))
    quad = np.roll(quad, -start, axis=0)
    if quad[1][1] > quad[3][1]:  # second point should be the top-right
        quad = quad[[0, 3, 2, 1]]
    return (quad / s).astype(np.float32)


def ecc_rep(bgr: np.ndarray, blur: float = 3.0, mode: str = "grad") -> np.ndarray:
    """Representation for ECC: blurred gradient magnitude, or equalized intensity."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (0, 0), blur)
    if mode == "int":
        g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
        return (g.astype(np.float32)) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), blur)
    mag /= (mag.max() + 1e-6)
    return mag.astype(np.float32)


def ecc_refine(ref_bgr: np.ndarray, mov_bgr: np.ndarray, W_init: np.ndarray,
               levels: tuple[int, ...] = (350, 700, 1400), verbose: bool = True,
               mov_mask: np.ndarray | None = None, rep: str = "grad"):
    """Pyramid ECC. W maps ORIGINAL coords -> FILMED coords (template -> input).
    mov_mask (uint8, filmed full-res) marks pixels of the filmed image ECC may use.
    Returns (W_refined, rho) or (None, -1) on divergence."""
    W = W_init.astype(np.float64).copy()
    W /= W[2, 2]
    rho = -1.0
    for max_dim in levels:
        ref_small, s_ref = scale_to_fit(ref_bgr, max_dim)
        mov_small, s_mov = scale_to_fit(mov_bgr, max_dim)
        t = ecc_rep(ref_small, mode=rep)
        i = ecc_rep(mov_small, mode=rep)
        mask_small = None
        if mov_mask is not None:
            mask_small = cv2.resize(mov_mask, (mov_small.shape[1], mov_small.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        S_ref = np.diag([s_ref, s_ref, 1.0])
        S_mov = np.diag([s_mov, s_mov, 1.0])
        W_lvl = (S_mov @ W @ np.linalg.inv(S_ref)).astype(np.float32)
        W_lvl /= W_lvl[2, 2]
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        try:
            rho, W_lvl = cv2.findTransformECC(t, i, W_lvl, cv2.MOTION_HOMOGRAPHY,
                                              criteria, mask_small, 5)
        except cv2.error:
            return None, -1.0
        W = np.linalg.inv(S_mov) @ W_lvl.astype(np.float64) @ S_ref
        W /= W[2, 2]
        if verbose:
            print(f"    [ecc @{max_dim}px] rho={rho:.3f}")
    return W, float(rho)


def coarse_similarity_init(ref_bgr: np.ndarray, mov_bgr: np.ndarray,
                           W_quad: np.ndarray) -> np.ndarray | None:
    """Brute-force scale + shift search at thumbnail resolution.

    W_quad maps ORIGINAL -> FILMED. The filmed image is first quad-rectified into the
    original's frame; a residual anisotropic scale + translation is then found by
    exhaustive template matching on gradient thumbnails. Returns improved W or None.
    """
    h, w = ref_bgr.shape[:2]
    mov_q = cv2.warpPerspective(mov_bgr, W_quad, (w, h),
                                flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
    tiny = 220
    ref_t, s = scale_to_fit(ref_bgr, tiny)
    mov_t = cv2.resize(mov_q, (ref_t.shape[1], ref_t.shape[0]), interpolation=cv2.INTER_AREA)
    g_ref = ecc_rep(ref_t, blur=2.0)
    g_mov = ecc_rep(mov_t, blur=2.0)

    th, tw = g_ref.shape
    pad_y, pad_x = th // 2, tw // 2
    best = None  # (score, sx, sy, dx, dy)
    for sx in np.arange(0.60, 1.46, 0.05):
        for sy in np.arange(0.60, 1.46, 0.05):
            scaled = cv2.resize(g_mov, (max(8, round(tw * sx)), max(8, round(th * sy))))
            padded = cv2.copyMakeBorder(scaled, pad_y, pad_y, pad_x, pad_x,
                                        cv2.BORDER_CONSTANT, value=0.0)
            res = cv2.matchTemplate(padded, g_ref, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, sx, sy, loc[0] - pad_x, loc[1] - pad_y)
    if best is None or best[0] < 0.05:
        return None
    _, sx, sy, dx, dy = best
    T = np.array([[1.0 / sx, 0, dx / sx],
                  [0, 1.0 / sy, dy / sy],
                  [0, 0, 1]], dtype=np.float64)
    S = np.diag([s, s, 1.0])
    T_full = np.linalg.inv(S) @ T @ S
    return W_quad @ T_full


def find_homography(ref_bgr: np.ndarray, mov_bgr: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Full-resolution homography mapping FILMED pixel coords -> ORIGINAL pixel coords."""
    H_feat, inliers = feature_homography(ref_bgr, mov_bgr, verbose=verbose)
    if H_feat is not None and inliers >= 2 * MIN_INLIERS:
        if verbose:
            print(f"  homography locked with {inliers} inliers")
        return H_feat

    # --- fallback: screen-quad init + pyramid ECC on gradient images -------------
    if verbose:
        print(f"  feature matching weak ({inliers} inliers) -> ECC fallback")
    h_ref, w_ref = ref_bgr.shape[:2]
    ref_corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]])
    inits: list[tuple[str, np.ndarray]] = []
    if H_feat is not None:
        inits.append(("features", np.linalg.inv(H_feat)))
    quad = detect_screen_quad(mov_bgr)
    mov_mask = None
    if quad is not None:
        W_quad = cv2.getPerspectiveTransform(ref_corners, quad).astype(np.float64)
        W_search = coarse_similarity_init(ref_bgr, mov_bgr, W_quad)
        if W_search is not None:
            inits.append(("quad + coarse search", W_search))
        for inset in (0.0, 0.05, 0.10):
            c = quad.mean(axis=0)
            q = quad * (1 - inset) + c * inset
            W0 = cv2.getPerspectiveTransform(ref_corners, q.astype(np.float32))
            inits.append((f"quad inset {int(inset * 100)}%", W0.astype(np.float64)))
        # mask off the raster edges: glitch sync bars there would anchor ECC wrongly
        mov_mask = np.zeros(mov_bgr.shape[:2], np.uint8)
        c = quad.mean(axis=0)
        inner = (quad * 0.86 + c * 0.14).astype(np.int32)
        cv2.fillConvexPoly(mov_mask, inner, 255)
    if not inits:
        raise RuntimeError(
            "Feature matching failed and no CRT screen quad could be detected. "
            "Check that the inputs correspond."
        )

    # coarse-score every init x representation, then fully refine the best few
    combos = []
    for name, W0 in inits:
        for rep in ("grad", "int"):
            W1, rho = ecc_refine(ref_bgr, mov_bgr, W0, levels=(350,), verbose=False,
                                 mov_mask=mov_mask, rep=rep)
            if W1 is not None:
                combos.append((rho, f"{name}/{rep}", W1, rep))
            if verbose:
                print(f"    init [{name}/{rep}] coarse rho={rho:.3f}")
    combos.sort(key=lambda x: -x[0])
    scored = []
    for rho0, name, W1, rep in combos[:3]:
        W2, rho = ecc_refine(ref_bgr, mov_bgr, W1, levels=(700, 1400), verbose=False,
                             mov_mask=mov_mask, rep=rep)
        if W2 is not None and sane_homography(np.linalg.inv(W2), mov_bgr.shape[:2],
                                              ref_bgr.shape[:2]):
            scored.append((rho, name, W2))
        if verbose:
            print(f"    refine [{name}] final rho={rho:.3f}")
    if not scored:
        raise RuntimeError("ECC fallback diverged for all initializations.")
    rho, name, W2 = max(scored, key=lambda x: x[0])
    if verbose:
        print(f"  best init: [{name}]")
    if rho < 0.15:
        raise RuntimeError(f"ECC alignment too weak (rho={rho:.3f}); giving up.")
    H = np.linalg.inv(W2)
    if verbose:
        print(f"  homography locked via ECC (rho={rho:.3f})")
    return H


def refine_flow(ref_bgr: np.ndarray, warped_bgr: np.ndarray,
                smooth_sigma: float = 21.0, max_disp: float = 60.0,
                verbose: bool = True, work_dim: int = FLOW_MAX_DIM
                ) -> tuple[np.ndarray, np.ndarray]:
    """Dense residual warp (CRT glass curvature, lens distortion) via smoothed optical flow.

    Returns full-resolution remap grids (map_x, map_y) to be applied to warped_bgr.
    """
    H, W = ref_bgr.shape[:2]
    ref_small, s = scale_to_fit(ref_bgr, work_dim)
    warped_small = cv2.resize(warped_bgr, (ref_small.shape[1], ref_small.shape[0]),
                              interpolation=cv2.INTER_AREA)
    g_ref = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY))
    g_wrp = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(warped_small, cv2.COLOR_BGR2GRAY))

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    flow = dis.calc(g_ref, g_wrp, None)  # ref(x) ~= warped(x + flow(x))

    max_disp_small = max_disp * s
    mag = np.linalg.norm(flow, axis=2)
    flow[mag > max_disp_small] = 0.0
    sigma = smooth_sigma * s
    k = int(sigma * 6) | 1
    flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (k, k), sigma)
    flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (k, k), sigma)
    if verbose:
        m = float(np.median(np.linalg.norm(flow, axis=2))) / s
        print(f"  residual curvature correction: median {m:.1f}px at full res")

    hs, ws = g_ref.shape
    xs, ys = np.meshgrid(np.arange(ws, dtype=np.float32), np.arange(hs, dtype=np.float32))
    map_x_small = xs + flow[..., 0]
    map_y_small = ys + flow[..., 1]
    map_x = cv2.resize(map_x_small, (W, H), interpolation=cv2.INTER_LINEAR) / s
    map_y = cv2.resize(map_y_small, (W, H), interpolation=cv2.INTER_LINEAR) / s
    return map_x, map_y


# ------------------------------------------------------------------ pin mode


def residual_homography(ref_bgr: np.ndarray, pinned_bgr: np.ndarray, verbose: bool):
    """Alignment between the corner-pinned filmed image and the original.
    Returns A: backward map homography (ORIGINAL coords -> PINNED coords), near identity."""
    H_feat, inliers = feature_homography(ref_bgr, pinned_bgr, verbose=verbose)
    if H_feat is not None and inliers >= 2 * MIN_INLIERS and \
            sane_homography(H_feat, pinned_bgr.shape[:2], ref_bgr.shape[:2]):
        if verbose:
            print(f"  interior match via features ({inliers} inliers)")
        return np.linalg.inv(H_feat)
    if verbose:
        print(f"  feature matching weak ({inliers} inliers) -> ECC from identity")
    inits = [("identity", np.eye(3))]
    W_search = coarse_similarity_init(ref_bgr, pinned_bgr, np.eye(3))
    if W_search is not None:
        inits.append(("coarse search", W_search))
    best = (None, -1.0)
    for init_name, W0 in inits:
        for rep in ("grad", "int"):
            W, rho = ecc_refine(ref_bgr, pinned_bgr, W0, levels=(350, 700, 1400),
                                verbose=False, rep=rep)
            if verbose:
                print(f"    [ecc {init_name}/{rep}] rho={rho:.3f}")
            if W is not None and rho > best[1] and \
                    sane_homography(np.linalg.inv(W), pinned_bgr.shape[:2], ref_bgr.shape[:2]):
                best = (W, rho)
    if best[0] is None:
        if verbose:
            print("  WARNING: interior match failed, keeping plain corner pin")
        return np.eye(3)
    if verbose:
        print(f"  interior match via ECC (rho={best[1]:.3f})")
    return best[0]


def pin_corners_exact(map_x: np.ndarray, map_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add a bilinear correction field so the 4 canvas corners map exactly to
    themselves — like dragging the corner handles of a mesh warp back onto the pins.
    The rest of the field keeps following the content match, smoothly adjusted."""
    Hh, Ww = map_x.shape
    corners = ((0, 0), (Ww - 1, 0), (Ww - 1, Hh - 1), (0, Hh - 1))
    ex = [cx - map_x[cy, cx] for (cx, cy) in corners]
    ey = [cy - map_y[cy, cx] for (cx, cy) in corners]
    u, v = np.meshgrid(np.linspace(0, 1, Ww, dtype=np.float32),
                       np.linspace(0, 1, Hh, dtype=np.float32))
    w_tl, w_tr = (1 - u) * (1 - v), u * (1 - v)
    w_br, w_bl = u * v, (1 - u) * v
    map_x = map_x + w_tl * ex[0] + w_tr * ex[1] + w_br * ex[2] + w_bl * ex[3]
    map_y = map_y + w_tl * ey[0] + w_tr * ey[1] + w_br * ey[2] + w_bl * ey[3]
    return map_x, map_y


def tps_fit(src: np.ndarray, off: np.ndarray, lam: float) -> np.ndarray:
    """Thin-plate spline through anchor points. src Nx2 (normalized coords),
    off Nx2 (offsets). Returns params (N+3, 2)."""
    n = len(src)
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = d2 * np.log(d2 + 1e-12)
    K += lam * np.eye(n)
    P = np.hstack([np.ones((n, 1)), src])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    Y = np.zeros((n + 3, 2))
    Y[:n] = off
    return np.linalg.solve(L, Y)


def tps_eval(query: np.ndarray, src: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate fitted TPS offsets at query points (Mx2). Returns Mx2 offsets."""
    n = len(src)
    d2 = ((query[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    U = d2 * np.log(d2 + 1e-12)
    return U @ params[:n] + np.hstack([np.ones((len(query), 1)), query]) @ params[n:]


def landmark_rep(bgr: np.ndarray) -> np.ndarray:
    """Representation for landmark template matching: gradient magnitude of
    lightly blurred CLAHE gray — robust to the glitch's tonal shifts."""
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return (mag / (mag.max() + 1e-6)).astype(np.float32)


def _collect_landmarks(g_ref, g_wrp, corners, patch, radius, min_score):
    """One template-matching sweep. Returns candidate (src, dst) lists in working px."""
    h, w = g_ref.shape
    half = patch // 2
    cand_src, cand_dst = [], []
    for (cx, cy) in corners.reshape(-1, 2):
        cx, cy = int(round(cx)), int(round(cy))
        if cx < half or cy < half or cx >= w - half or cy >= h - half:
            continue
        tmpl = g_ref[cy - half: cy + half + 1, cx - half: cx + half + 1]
        if float(tmpl.std()) < 0.01:
            continue  # flat patch, nothing distinctive
        x0, x1 = max(0, cx - half - radius), min(w, cx + half + radius + 1)
        y0, y1 = max(0, cy - half - radius), min(h, cy + half + radius + 1)
        win = g_wrp[y0:y1, x0:x1]
        if win.shape[0] <= patch or win.shape[1] <= patch:
            continue
        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < min_score:
            continue
        # uniqueness: second peak away from the first must be clearly worse
        res2 = res.copy()
        rx0, rx1 = max(0, loc[0] - 5), min(res.shape[1], loc[0] + 6)
        ry0, ry1 = max(0, loc[1] - 5), min(res.shape[0], loc[1] + 6)
        res2[ry0:ry1, rx0:rx1] = -1
        _, second, _, _ = cv2.minMaxLoc(res2)
        if second > 0.92 * score:
            continue
        cand_src.append((cx, cy))
        cand_dst.append((x0 + loc[0] + half, y0 + loc[1] + half))
    return cand_src, cand_dst


def find_landmarks(ref_bgr: np.ndarray, warped_bgr: np.ndarray, verbose: bool,
                   attempts=((25, 40, 0.45), (31, 60, 0.30)), work_dim: int = 1200):
    """Pick distinctive points in the original (corners, eyes, building tops) and find
    each one in the roughly-aligned filmed image by normalized cross-correlation.
    attempts: sequence of (patch, search radius, min score) tried until enough anchors.
    work_dim: matching resolution — coarser is more tolerant of heavy glitch.

    Returns (pts_ref Nx2, pts_found Nx2) in full-resolution canvas coords, plus the
    rejected candidates for QC display."""
    small, s = scale_to_fit(ref_bgr, work_dim)
    warped_small = cv2.resize(warped_bgr, (small.shape[1], small.shape[0]),
                              interpolation=cv2.INTER_AREA)
    g_ref = landmark_rep(small)
    g_wrp = landmark_rep(warped_small)
    h, w = g_ref.shape

    gray_ref = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    corners = cv2.goodFeaturesToTrack(gray_ref, maxCorners=250, qualityLevel=0.01,
                                      minDistance=max(12, min(h, w) // 30), blockSize=7)
    if corners is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), []

    best = (np.zeros((0, 2)), np.zeros((0, 2)), [])
    for patch, radius, min_score in attempts:
        cand_src, cand_dst = _collect_landmarks(g_ref, g_wrp, corners,
                                                patch, radius, min_score)
        if len(cand_src) < 4:
            continue
        src = np.array(cand_src, np.float64)
        dst = np.array(cand_dst, np.float64)
        # global consistency: the true warp is nearly affine + gentle curvature, so
        # every honest anchor must roughly agree with one affine transform (RANSAC)
        M, inl = cv2.estimateAffine2D(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                      method=cv2.RANSAC, ransacReprojThreshold=10.0,
                                      confidence=0.995, maxIters=5000)
        keep = np.zeros(len(src), bool) if M is None else inl.ravel().astype(bool)
        rejected = [(src[i] / s, dst[i] / s) for i in range(len(src)) if not keep[i]]
        result = (src[keep] / s, dst[keep] / s, rejected)
        if len(result[0]) > len(best[0]):
            best = result
        if len(best[0]) >= 12:
            break
    if verbose:
        print(f"  landmarks: {len(best[0])} matched ({len(best[2])} outliers dropped)")
    return best


def alignment_score(ref_bgr: np.ndarray, warped_bgr: np.ndarray) -> float:
    """Global gradient-correlation score used to verify that a warp round helped."""
    small, _ = scale_to_fit(ref_bgr, 600)
    w_small = cv2.resize(warped_bgr, (small.shape[1], small.shape[0]),
                         interpolation=cv2.INTER_AREA)
    a = landmark_rep(small).ravel()
    b = landmark_rep(w_small).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def draw_landmarks_qc(ref_bgr, pts_ref, pts_found, rejected, out_path):
    """QC sheet: original with kept anchors (green, line to matched pos) and rejects (red)."""
    vis, s = scale_to_fit(ref_bgr.copy(), 1600)
    for p, q in rejected:
        cv2.drawMarker(vis, tuple((p * s).astype(int)), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 12, 2)
    for p, q in zip(pts_ref, pts_found):
        p_i = tuple((p * s).astype(int))
        q_i = tuple((q * s).astype(int))
        cv2.line(vis, p_i, q_i, (0, 255, 255), 2)
        cv2.circle(vis, p_i, 5, (0, 255, 0), 2)
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])


def distill_to_mesh(map_cx: np.ndarray, map_cy: np.ndarray, mesh: int,
                    verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a dense warp field to a coarse control-point mesh (like a manual mesh
    warp): sample the measured displacement at a small grid of nodes, then rebuild a
    globally smooth field from just those nodes. Kills local rubber-sheet wobble."""
    Hh, Ww = map_cx.shape
    nx = max(4, mesh)
    ny = max(4, round(mesh * Hh / Ww)) if Ww >= Hh else max(4, round(mesh * Ww / Hh))
    if Ww < Hh:
        nx, ny = ny, max(4, mesh)
    xs, ys = np.meshgrid(np.arange(Ww, dtype=np.float32), np.arange(Hh, dtype=np.float32))
    ox = map_cx - xs
    oy = map_cy - ys
    node_x = np.linspace(0, Ww - 1, nx)
    node_y = np.linspace(0, Hh - 1, ny)
    rad = max(4, int(0.25 * min(Ww / (nx - 1), Hh / (ny - 1))))
    grid_ox = np.zeros((ny, nx), np.float32)
    grid_oy = np.zeros((ny, nx), np.float32)
    for j, yn in enumerate(node_y):
        for i, xn in enumerate(node_x):
            x0, x1 = max(0, int(xn) - rad), min(Ww, int(xn) + rad + 1)
            y0, y1 = max(0, int(yn) - rad), min(Hh, int(yn) + rad + 1)
            grid_ox[j, i] = np.median(ox[y0:y1, x0:x1])
            grid_oy[j, i] = np.median(oy[y0:y1, x0:x1])
    limit = 0.10 * min(Hh, Ww)
    np.clip(grid_ox, -limit, limit, out=grid_ox)
    np.clip(grid_oy, -limit, limit, out=grid_oy)
    if verbose:
        m = float(np.median(np.hypot(grid_ox, grid_oy)))
        print(f"  mesh {nx}x{ny}: median node displacement {m:.1f}px")
    field_x = cv2.resize(grid_ox, (Ww, Hh), interpolation=cv2.INTER_CUBIC)
    field_y = cv2.resize(grid_oy, (Ww, Hh), interpolation=cv2.INTER_CUBIC)
    return xs + field_x, ys + field_y


def pin_align(ref_bgr: np.ndarray, mov_bgr: np.ndarray, inset: float = 0.02,
              refine: bool = True, stiffness: float = 0.003,
              manual_points: np.ndarray | None = None, verbose: bool = True,
              maps: tuple[np.ndarray, np.ndarray] | None = None):
    """Corner-pin the CRT screen content onto the full original canvas, then anchor
    distinctive landmark points (eyes, building tops, corners of shapes) to their exact
    positions in the original; a thin-plate spline pulls everything in between smoothly.
    Returns (aligned, (map_x, map_y), qc) — maps are reusable per frame."""
    Hh, Ww = ref_bgr.shape[:2]
    if maps is not None:
        aligned = cv2.remap(mov_bgr, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        return aligned, maps, None

    quad = detect_screen_quad(mov_bgr)
    if quad is None:
        raise RuntimeError("Could not detect the CRT screen in the filmed input. "
                           "Pin mode needs the bright screen against a darker surround.")
    c = quad.mean(axis=0)
    q = (quad * (1 - inset) + c * inset).astype(np.float32)
    ref_corners = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]])
    H_pin = cv2.getPerspectiveTransform(ref_corners, q)  # canvas -> filmed
    if verbose:
        print(f"  screen quad found, corner-pinned with {inset * 100:.0f}% inset")

    # estimation image: screen content rectified onto the canvas
    pinned = cv2.warpPerspective(mov_bgr, H_pin, (Ww, Hh),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    # ---- smooth analytic prior: residual homography only (NO dense flow — flow
    # melts the glitch and its folds corrupt anchor targets)
    A = residual_homography(ref_bgr, pinned, verbose)
    xs, ys = np.meshgrid(np.arange(Ww, dtype=np.float32), np.arange(Hh, dtype=np.float32))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
    res = cv2.perspectiveTransform(pts, A.astype(np.float64)).reshape(Hh, Ww, 2)
    map_cx = res[..., 0].astype(np.float32)
    map_cy = res[..., 1].astype(np.float32)
    map_cx, map_cy = pin_corners_exact(map_cx, map_cy)

    corner_pts = np.float64([[0, 0], [Ww - 1, 0], [Ww - 1, Hh - 1], [0, Hh - 1]])
    L = float(max(Ww, Hh))

    def tps_offset_field(src, dst):
        """Full-res smooth offset field through anchors (src -> dst) + pinned corners."""
        s_all = np.vstack([src, corner_pts])
        d_all = np.vstack([dst, corner_pts])
        params = tps_fit(s_all / L, (d_all - s_all) / L, lam=stiffness)
        gw = 240
        gh = max(24, round(gw * Hh / Ww))
        gxs, gys = np.meshgrid(np.linspace(0, Ww - 1, gw), np.linspace(0, Hh - 1, gh))
        qpts = np.stack([gxs.ravel(), gys.ravel()], axis=1)
        off = tps_eval(qpts / L, s_all / L, params) * L
        off_x = cv2.resize(off[:, 0].reshape(gh, gw).astype(np.float32), (Ww, Hh),
                           interpolation=cv2.INTER_CUBIC)
        off_y = cv2.resize(off[:, 1].reshape(gh, gw).astype(np.float32), (Ww, Hh),
                           interpolation=cv2.INTER_CUBIC)
        return off_x, off_y

    qc = None
    if manual_points is not None and len(manual_points):
        # user-picked anchors: ref x,y <-> filmed x,y. Filmed coords -> pinned space,
        # then one direct TPS: each anchor lands exactly, corners stay pinned.
        ref_pts = manual_points[:, :2].astype(np.float64)
        film_pts = manual_points[:, 2:].astype(np.float32).reshape(-1, 1, 2)
        pin_pts = cv2.perspectiveTransform(
            film_pts, np.linalg.inv(H_pin.astype(np.float64))).reshape(-1, 2)
        off_x, off_y = tps_offset_field(ref_pts, pin_pts.astype(np.float64))
        map_cx = xs + off_x
        map_cy = ys + off_y
        map_cx, map_cy = pin_corners_exact(map_cx, map_cy)
        if verbose:
            print(f"  manual TPS through {len(ref_pts)} user anchors, corners pinned")
    else:
        # ---- iterative landmark anchoring, like doing it by hand: find obvious
        # points, pin them, let the TPS pull everything between, repeat tighter.
        # Every round must IMPROVE the global alignment score or it is reverted.
        rounds = ((600, ((21, 60, 0.28), (27, 90, 0.22))),   # coarse: long reach
                  (1200, ((31, 45, 0.30),)),                 # mid: normal matching
                  (1200, ((25, 25, 0.40),)))                 # tight verification
        for rnd, (work_dim, attempts) in enumerate(rounds):
            prior_warped = cv2.remap(pinned, map_cx, map_cy,
                                     interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            score_before = alignment_score(ref_bgr, prior_warped)
            pts_ref, pts_found, rejected = find_landmarks(ref_bgr, prior_warped,
                                                          verbose, attempts=attempts,
                                                          work_dim=work_dim)
            if len(pts_ref) < 6:
                if verbose:
                    print(f"  round {rnd + 1}: only {len(pts_ref)} landmarks, stopping")
                break

            # o(ref) = found - ref, o(corner) = 0; new map = old map at x + o(x)
            off_x, off_y = tps_offset_field(pts_ref, pts_found)
            nx = np.clip(xs + off_x, 0, Ww - 1)
            ny = np.clip(ys + off_y, 0, Hh - 1)
            cand_cx = cv2.remap(map_cx, nx, ny, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            cand_cy = cv2.remap(map_cy, nx, ny, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            cand_cx, cand_cy = pin_corners_exact(cand_cx, cand_cy)

            cand_warped = cv2.remap(pinned, cand_cx, cand_cy,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            score_after = alignment_score(ref_bgr, cand_warped)
            if score_after < score_before - 0.002:
                if verbose:
                    print(f"  round {rnd + 1}: TPS rejected "
                          f"(score {score_before:.3f} -> {score_after:.3f}), reverting")
                break
            map_cx, map_cy = cand_cx, cand_cy
            qc = (pts_ref, pts_found, rejected)
            if verbose:
                m = float(np.median(np.linalg.norm(pts_found - pts_ref, axis=1)))
                print(f"  round {rnd + 1}: TPS through {len(pts_ref)} anchors "
                      f"(median adjust {m:.1f}px, score {score_before:.3f} -> "
                      f"{score_after:.3f})")

    # stay inside the pinned screen content: never sample bezel/surround
    # (pinned space [0..W)x[0..H) IS the inset screen quad)
    map_cx = np.clip(map_cx, 0, Ww - 1)
    map_cy = np.clip(map_cy, 0, Hh - 1)

    # compose with the corner pin: canvas -> filmed coords, single resampling
    pts = np.stack([map_cx, map_cy], axis=-1).reshape(-1, 1, 2)
    total = cv2.perspectiveTransform(pts, H_pin.astype(np.float64)).reshape(Hh, Ww, 2)
    h_mov, w_mov = mov_bgr.shape[:2]
    total_x = np.clip(total[..., 0], 0, w_mov - 1).astype(np.float32)
    total_y = np.clip(total[..., 1], 0, h_mov - 1).astype(np.float32)

    aligned = cv2.remap(mov_bgr, total_x, total_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return aligned, (total_x, total_y), qc


# ------------------------------------------------------------------ exact mode

def align_image(ref_bgr: np.ndarray, mov_bgr: np.ndarray, refine: bool = True,
                verbose: bool = True, H: np.ndarray | None = None,
                maps: tuple[np.ndarray, np.ndarray] | None = None):
    """Exact mode: warp mov_bgr into ref_bgr's pixel frame. Returns (aligned, H, maps)."""
    Hh, Ww = ref_bgr.shape[:2]
    if H is None:
        H = find_homography(ref_bgr, mov_bgr, verbose=verbose)
    warped = cv2.warpPerspective(mov_bgr, H, (Ww, Hh), flags=cv2.INTER_LINEAR)
    if refine:
        if maps is None:
            maps = refine_flow(ref_bgr, warped, verbose=verbose)
        warped = cv2.remap(warped, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
    return warped, H, maps


def make_overlay(ref_bgr: np.ndarray, aligned_bgr: np.ndarray) -> np.ndarray:
    """Side-by-side QC sheet: 50/50 blend and checkerboard mix."""
    blend = cv2.addWeighted(ref_bgr, 0.5, aligned_bgr, 0.5, 0)
    h, w = ref_bgr.shape[:2]
    tile = max(32, min(h, w) // 12)
    yy, xx = np.meshgrid(np.arange(h) // tile, np.arange(w) // tile, indexing="ij")
    checker = ((yy + xx) % 2 == 0)[..., None]
    board = np.where(checker, ref_bgr, aligned_bgr)
    return np.hstack([blend, board])


# ---------------------------------------------------------------- video helpers

def luma_signature(path: str, max_frames: int = 100000) -> np.ndarray:
    """Per-frame mean luma of the (center-cropped) frame, for temporal sync."""
    cap = cv2.VideoCapture(path)
    sig = []
    while len(sig) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        crop = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        sig.append(float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()
    return np.array(sig, dtype=np.float64)


def sync_offset_candidates(sig_ref: np.ndarray, fps_ref: float,
                           sig_mov: np.ndarray, fps_mov: float,
                           k: int = 5) -> list[float]:
    """Candidate time offsets o (seconds) such that mov at time t + o shows ref at
    time t — the k best correlation peaks (1s apart), strongest first. Works across
    different frame rates by resampling both signatures to 25 Hz."""
    R = 25.0
    def resample(sig, fps):
        t = np.arange(len(sig)) / fps
        tt = np.arange(0.0, t[-1] if len(t) else 0.0, 1.0 / R)
        return np.interp(tt, t, sig)
    a = resample(sig_ref, fps_ref)
    b = resample(sig_mov, fps_mov)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    corr = np.correlate(b, a, mode="full")
    order = np.argsort(corr)[::-1]
    picks: list[int] = []
    for idx in order:
        if all(abs(idx - p) > R for p in picks):
            picks.append(int(idx))
        if len(picks) >= k:
            break
    return [(p - (len(a) - 1)) / R for p in picks]


def quick_pair_score(ref_bgr: np.ndarray, mov_bgr: np.ndarray, inset: float) -> float:
    """Cheap spatial-alignment plausibility for one frame pair: corner-pin, coarse
    scale/shift search, one coarse ECC level. Returns rho (-1 if hopeless)."""
    quad = detect_screen_quad(mov_bgr)
    if quad is None:
        return -1.0
    Hh, Ww = ref_bgr.shape[:2]
    c = quad.mean(axis=0)
    q = (quad * (1 - inset) + c * inset).astype(np.float32)
    ref_corners = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]])
    H_pin = cv2.getPerspectiveTransform(ref_corners, q)
    pinned = cv2.warpPerspective(mov_bgr, H_pin, (Ww, Hh),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    best = -1.0
    inits = [np.eye(3)]
    W_s = coarse_similarity_init(ref_bgr, pinned, np.eye(3))
    if W_s is not None:
        inits.append(W_s)
    for W0 in inits:
        _, rho = ecc_refine(ref_bgr, pinned, W0, levels=(350,), verbose=False)
        best = max(best, rho)
    return best


def read_frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from {path}")
    return frame


def video_meta(path: str) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def align_video(original: str, filmed: str, out_path: str, mode: str, refine: bool,
                per_frame_every: int, inset: float, verbose: bool):
    ref_kind = media_kind(original)
    n_mov, fps_mov, _, _ = video_meta(filmed)

    dur_mov = n_mov / fps_mov
    if ref_kind == "image":
        ref_bgr = cv2.imread(original, cv2.IMREAD_COLOR)
        if ref_bgr is None:
            raise RuntimeError(f"Cannot read {original}")
        off, t0, fps_out, n_out = 0.0, 0.0, fps_mov, n_mov
        get_ref = lambda t: ref_bgr
    else:
        n_ref, fps_ref, _, _ = video_meta(original)
        dur_ref = n_ref / fps_ref
        if verbose:
            print("Computing temporal sync ...")
        cands = sync_offset_candidates(luma_signature(original), fps_ref,
                                       luma_signature(filmed), fps_mov)
        # brightness correlation alone can lock onto the wrong repeat; verify each
        # candidate offset by how well its mid-frame pair aligns spatially
        scored = []
        for o in cands:
            ta = max(0.0, -o)
            tb = min(dur_ref, dur_mov - o)
            if tb - ta < 1.0:
                continue
            tm = (ta + tb) / 2
            try:
                r = read_frame(original, min(n_ref - 1, int(round(tm * fps_ref))))
                m = read_frame(filmed, min(n_mov - 1, int(round((tm + o) * fps_mov))))
            except RuntimeError:
                continue
            rho = quick_pair_score(r, m, inset)
            scored.append((rho, o))
            if verbose:
                print(f"  offset candidate {o:+.2f}s: spatial rho={rho:.3f}")
        if not scored:
            raise RuntimeError("Videos do not overlap after temporal sync.")
        _, off = max(scored)
        t0 = max(0.0, -off)
        t1 = min(dur_ref, dur_mov - off)
        if verbose:
            print(f"  filmed video offset: {off:+.2f}s, overlap {t1 - t0:.1f}s")
        fps_out = fps_ref
        n_out = int((t1 - t0) * fps_ref)
        def get_ref(t, _fps=fps_ref, _n=n_ref):
            return read_frame(original, min(_n - 1, int(round(t * _fps))))

    t_mid = t0 + (n_out / fps_out) / 2
    ref_mid = get_ref(t_mid)
    mov_mid = read_frame(filmed, min(n_mov - 1, int(round((t_mid + off) * fps_mov))))
    if verbose:
        print("Estimating alignment on mid-sequence frame ...")
    if mode == "pin":
        _, maps, _ = pin_align(ref_mid, mov_mid, inset=inset, refine=refine,
                               verbose=verbose)
        H = None
    else:
        _, H, maps = align_image(ref_mid, mov_mid, refine=refine, verbose=verbose)

    h_out, w_out = ref_mid.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    # read the filmed video forward, mapping each output time to a filmed frame
    cap = cv2.VideoCapture(filmed)
    first_target = max(0, int(round((t0 + off) * fps_mov)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_target)
    cur = first_target - 1
    mov = None
    for i in range(n_out):
        t = t0 + i / fps_out
        target = min(n_mov - 1, max(0, int(round((t + off) * fps_mov))))
        while cur < target:
            ok, nxt = cap.read()
            if not ok:
                break
            mov, cur = nxt, cur + 1
        if mov is None:
            break
        if per_frame_every and i % per_frame_every == 0 and i > 0:
            try:
                ref_i = get_ref(t)
                if mode == "pin":
                    _, maps, _ = pin_align(ref_i, mov, inset=inset, refine=refine,
                                           verbose=False)
                else:
                    _, H, maps = align_image(ref_i, mov, refine=refine, verbose=False)
            except RuntimeError:
                pass  # keep previous transform
        if mode == "pin":
            frame = cv2.remap(mov, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        else:
            frame = cv2.warpPerspective(mov, H, (w_out, h_out), flags=cv2.INTER_LINEAR)
            if refine and maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
        if verbose and i % 50 == 0:
            print(f"  frame {i}/{n_out}", end="\r")
    cap.release()
    writer.release()
    if verbose:
        print(f"\nWrote {out_path}")


# --------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original", help="source image/video shown on the CRT")
    ap.add_argument("filmed", help="photo/video of the CRT screen")
    ap.add_argument("-o", "--output", help="output path (default: <filmed>_aligned.<ext>)")
    ap.add_argument("--mode", choices=("pin", "exact"), default="pin",
                    help="pin: full-bleed corner pin + elastic interior match (default); "
                         "exact: strict geometric mapping, black where nothing was filmed")
    ap.add_argument("--inset", type=float, default=0.02,
                    help="pin: crop this fraction inside the detected screen edge (default 0.02)")
    ap.add_argument("--points", metavar="FILE",
                    help="pin: CSV of manual anchors, one per line: ref_x,ref_y,"
                         "filmed_x,filmed_y (pixel coords in the original / the filmed "
                         "photo). Skips automatic landmark search. Images only.")
    ap.add_argument("--stiffness", type=float, default=0.003,
                    help="pin: TPS regularization; higher = stiffer warp that averages "
                         "over anchors, lower = anchors pinned more exactly (default 0.003)")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip dense curvature/flow refinement")
    ap.add_argument("--overlay", action="store_true",
                    help="also write a *_overlay QC image (images only)")
    ap.add_argument("--per-frame", type=int, default=0, metavar="N",
                    help="video: re-estimate alignment every N frames (0 = once, fixed camera)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    kind_ref = media_kind(args.original)
    kind_mov = media_kind(args.filmed)
    refine = not args.no_refine

    stem, _ = os.path.splitext(args.filmed)
    if kind_mov == "video" or kind_ref == "video":
        out = args.output or stem + "_aligned.mp4"
        if kind_ref == "video" and kind_mov == "image":
            raise SystemExit("original=video with filmed=image is not supported")
        if args.points:
            raise SystemExit("--points is supported for image pairs only")
        align_video(args.original, args.filmed, out, args.mode, refine,
                    args.per_frame, args.inset, verbose)
    else:
        out = args.output or stem + "_aligned.png"
        ref = cv2.imread(args.original, cv2.IMREAD_COLOR)
        mov = cv2.imread(args.filmed, cv2.IMREAD_COLOR)
        if ref is None:
            raise SystemExit(f"Cannot read {args.original}")
        if mov is None:
            raise SystemExit(f"Cannot read {args.filmed}")
        if verbose:
            print(f"Aligning {os.path.basename(args.filmed)} -> "
                  f"{os.path.basename(args.original)} ({ref.shape[1]}x{ref.shape[0]}, "
                  f"mode={args.mode})")
        manual_points = None
        if args.points:
            manual_points = np.loadtxt(args.points, delimiter=",", ndmin=2)
            if manual_points.shape[1] != 4:
                raise SystemExit("--points file must have 4 columns: "
                                 "ref_x,ref_y,filmed_x,filmed_y")
        points_qc = None
        if args.mode == "pin":
            aligned, _, points_qc = pin_align(ref, mov, inset=args.inset, refine=refine,
                                              stiffness=args.stiffness,
                                              manual_points=manual_points,
                                              verbose=verbose)
        else:
            aligned, _, _ = align_image(ref, mov, refine=refine, verbose=verbose)
        cv2.imwrite(out, aligned)
        if verbose:
            print(f"Wrote {out}")
        if args.overlay:
            qc = make_overlay(ref, aligned)
            qc_path = os.path.splitext(out)[0] + "_overlay.jpg"
            cv2.imwrite(qc_path, qc, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if verbose:
                print(f"Wrote {qc_path}")
            if points_qc is not None:
                pts_path = os.path.splitext(out)[0] + "_points.jpg"
                draw_landmarks_qc(ref, *points_qc, pts_path)
                if verbose:
                    print(f"Wrote {pts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
