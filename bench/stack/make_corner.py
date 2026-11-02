"""A building corner (two textured vertical walls at 90 deg) + ground, seen by a
handheld camera that moves a few metres between shots, like walking round a corner.
Truth: projection of 3D points on both walls -> per-shot pixel positions."""
import cv2, numpy as np, json, os, sys
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
R0 = '/home/user/jasblack/crt-align/'
texA = cv2.imread(R0 + 'base_26.jpg'); texB = cv2.imread(R0 + 'base_49.jpg')
W, H, f = 1500, 2000, 1500.0  # portrait phone
rng = np.random.default_rng(3)
# walls: A spans x in [-12,0] at z=0 ... corner at origin; wall A: points (x,y,0) x<=0, wall B: (0,y,z) z>=0 ... height y in [0,16]
def look(cam, target):
    fwd = target - cam; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    return np.stack([right, -up, fwd])  # rows: camera axes (x right, y down, z forward)
def render(cam, tgt, roll):
    Rm = look(cam, tgt); c, s = np.cos(roll), np.sin(roll)
    Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ Rm
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    P = lambda X: K @ (Rm @ (np.asarray(X, float) - cam))
    img = np.full((H, W, 3), (235, 170, 90), np.uint8)  # sky (BGR)
    # ground plane y=0 grey
    def wall(tex, origin, u_axis, v_axis, uw, vh):
        th, tw = tex.shape[:2]
        # tex px (a,b) -> world origin + u*(a/tw*uw) + v*(b/th*vh); v points down from top
        M = np.column_stack([u_axis * uw / tw, v_axis * vh / th, origin])
        Hh = K @ Rm @ (M - np.outer(cam, [0, 0, 1]))
        return cv2.warpPerspective(tex, Hh / Hh[2, 2], (W, H)), cv2.warpPerspective(np.full((th, tw), 255, np.uint8), Hh / Hh[2, 2], (W, H))
    top = np.array([0, 16.0, 0])
    a, ma = wall(texA, np.array([-12, 16.0, 0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), 12, 16)
    b, mb = wall(texB, top, np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), 12, 16)
    for t, m in ((a, ma), (b, mb)): img[m > 127] = t[m > 127]
    return img, P
shots = []
for k in range(5):
    cam = np.array([-14 + rng.uniform(-2, 2), 1.6 + rng.uniform(-0.3, 0.3), 14 + rng.uniform(-2, 2)])
    tgt = np.array([0, 8, 0]) + rng.normal(0, 0.6, 3)
    img, P = render(cam, tgt, np.deg2rad(rng.uniform(-3, 3)))
    img = np.clip(img.astype(float) * rng.uniform(0.7, 1.3) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(f'{out}/corner_{k}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    pts = {}
    for name, X in {'corner_mid': [0, 8, 0], 'corner_top': [0, 14, 0], 'A_mid': [-6, 8, 0], 'B_mid': [0, 8, 6], 'A_far': [-11, 4, 0], 'B_far': [0, 4, 11]}.items():
        q = P(X); pts[name] = (q[:2] / q[2]).tolist()
    shots.append({'file': f'corner_{k}.jpg', 'pts': pts})
json.dump({'shots': shots}, open(f'{out}/truth.json', 'w'), indent=1)
