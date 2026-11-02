// Scale + rotation invariant binary features (ORB-style) in plain JS.
//   detect: Harris corners on a 1.25x pyramid, spread over a grid, sub-pixel peak
//   orientation: intensity centroid
//   descriptor: 256-bit rotated BRIEF on a smoothed patch
// Binary tests compare brightness only, so exposure / white balance changes between
// photos don't matter. Keypoints are returned in level-0 pixel coordinates.

import { downscale, blur } from './image.js';

const NBITS = 256, WORDS = NBITS / 32;
const PATCH = 15;          // half size of the BRIEF sampling area (level pixels)
const BORDER = 24;         // keep keypoints this far from the level's edge

// fixed test pattern: Gaussian pairs, deterministic
const PATTERN = (() => {
  let s = 0x9e3779b9;
  const r = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const g = () => {
    let u = 0;
    for (let k = 0; k < 6; k++) u += r();
    return (u - 3) * (PATCH / 2.2);
  };
  const c = v => Math.max(-PATCH, Math.min(PATCH, v));
  const p = new Float32Array(NBITS * 4);
  for (let i = 0; i < NBITS; i++) { p[4 * i] = c(g()); p[4 * i + 1] = c(g()); p[4 * i + 2] = c(g()); p[4 * i + 3] = c(g()); }
  return p;
})();

function bilinear(img, x, y) {
  const { data, w, h } = img;
  if (x < 0) x = 0; else if (x > w - 1.001) x = w - 1.001;
  if (y < 0) y = 0; else if (y > h - 1.001) y = h - 1.001;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * w + x0;
  return (data[i] * (1 - fx) + data[i + 1] * fx) * (1 - fy) + (data[i + w] * (1 - fx) + data[i + w + 1] * fx) * fy;
}

function harris(img) {
  const { w, h } = img;
  const S = blur(img, 1.0).data;
  const xx = new Float32Array(w * h), yy = new Float32Array(w * h), xy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x;
    const gx = 0.5 * (S[p + 1] - S[p - 1]), gy = 0.5 * (S[p + w] - S[p - w]);
    xx[p] = gx * gx; yy[p] = gy * gy; xy[p] = gx * gy;
  }
  const A = blur({ data: xx, w, h }, 1.5).data, B = blur({ data: yy, w, h }, 1.5).data, C = blur({ data: xy, w, h }, 1.5).data;
  const R = new Float32Array(w * h);
  for (let i = 0; i < R.length; i++) {
    const tr = A[i] + B[i];
    R[i] = A[i] * B[i] - C[i] * C[i] - 0.04 * tr * tr;
  }
  return R;
}

// mask: optional function (x0, y0) -> bool in level-0 coordinates
function detectLevel(img, scale, want, mask) {
  const { w, h } = img, R = harris(img);
  const cell = 24, gw = Math.ceil(w / cell), gh = Math.ceil(h / cell);
  const cand = [];
  let maxR = 0;
  for (let i = 0; i < R.length; i++) if (R[i] > maxR) maxR = R[i];
  const floor = maxR * 1e-4;
  for (let y = BORDER; y < h - BORDER; y++) for (let x = BORDER; x < w - BORDER; x++) {
    const p = y * w + x, v = R[p];
    if (v <= floor) continue;
    if (v < R[p - 1] || v < R[p + 1] || v < R[p - w] || v < R[p + w] ||
        v < R[p - w - 1] || v < R[p - w + 1] || v < R[p + w - 1] || v < R[p + w + 1]) continue;
    if (mask && !mask(x / scale, y / scale)) continue;
    // sub-pixel peak (separable parabola)
    const dx = (R[p + 1] - R[p - 1]) / (2 * (2 * v - R[p + 1] - R[p - 1]) || 1);
    const dy = (R[p + w] - R[p - w]) / (2 * (2 * v - R[p + w] - R[p - w]) || 1);
    cand.push({ x: x + Math.max(-0.5, Math.min(0.5, dx)), y: y + Math.max(-0.5, Math.min(0.5, dy)), r: v, c: ((y / cell) | 0) * gw + ((x / cell) | 0) });
  }
  // spread: strongest per grid cell first, then fill by response
  cand.sort((a, b) => b.r - a.r);
  const perCell = new Uint16Array(gw * gh), out = [];
  const cap = Math.max(1, Math.ceil((want / (gw * gh)) * 2));
  for (const c of cand) {
    if (out.length >= want) break;
    if (perCell[c.c] >= cap) continue;
    perCell[c.c]++; out.push(c);
  }
  return out;
}

function orientation(img, x, y) {
  let m10 = 0, m01 = 0;
  const r = 12;
  for (let dy = -r; dy <= r; dy++) {
    const span = Math.floor(Math.sqrt(r * r - dy * dy));
    for (let dx = -span; dx <= span; dx++) {
      const v = bilinear(img, x + dx, y + dy);
      m10 += dx * v; m01 += dy * v;
    }
  }
  return Math.atan2(m01, m10);
}

function describe(img, x, y, ang, out, off) {
  const c = Math.cos(ang), s = Math.sin(ang);
  for (let wi = 0; wi < WORDS; wi++) {
    let word = 0;
    for (let b = 0; b < 32; b++) {
      const i = (wi * 32 + b) * 4;
      const ax = PATTERN[i], ay = PATTERN[i + 1], bx = PATTERN[i + 2], by = PATTERN[i + 3];
      const va = bilinear(img, x + c * ax - s * ay, y + s * ax + c * ay);
      const vb = bilinear(img, x + c * bx - s * by, y + s * bx + c * by);
      if (va < vb) word |= 1 << b;
    }
    out[off + wi] = word;
  }
}

// img: { data, w, h } gray. Returns { n, x: Float32Array, y, scale, desc: Uint32Array }.
export function detectAndDescribe(img, { count = 1500, levels = 6, factor = 1.25, mask = null } = {}) {
  const pts = [];
  // features per level proportional to level area
  let areaSum = 0;
  for (let l = 0; l < levels; l++) areaSum += 1 / factor ** (2 * l);
  let L = img;
  for (let l = 0; l < levels; l++) {
    if (l > 0) {
      // each level from the previous one (cheap blur), not from level 0
      const w = Math.round(L.w / factor), h = Math.round(L.h / factor);
      if (Math.min(w, h) < 2 * BORDER + 16) break;
      L = downscale(L, w, h);
    }
    const sc = L.w / img.w;
    const want = Math.round(count * (1 / factor ** (2 * l)) / areaSum * 1.15);
    const kps = detectLevel(L, sc, want, mask);
    const D = blur(L, 1.6);
    for (const k of kps) {
      const ang = orientation(L, k.x, k.y);
      pts.push({ x: k.x / sc, y: k.y / sc, lx: k.x, ly: k.y, ang, lvl: l, D, sc });
    }
  }
  const n = pts.length;
  const x = new Float32Array(n), y = new Float32Array(n), scale = new Float32Array(n);
  const desc = new Uint32Array(n * WORDS);
  pts.forEach((p, i) => {
    x[i] = p.x; y[i] = p.y; scale[i] = 1 / p.sc;
    describe(p.D, p.lx, p.ly, p.ang, desc, i * WORDS);
  });
  return { n, x, y, scale, desc };
}

const pop = v => {
  v -= (v >>> 1) & 0x55555555;
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
};

function nearest(A, B) {
  const best = new Int32Array(A.n).fill(-1), d1 = new Uint16Array(A.n), d2 = new Uint16Array(A.n);
  for (let i = 0; i < A.n; i++) {
    let b1 = 999, b2 = 999, bi = -1;
    const ai = i * WORDS;
    for (let j = 0; j < B.n; j++) {
      const bj = j * WORDS;
      let d = 0;
      for (let k = 0; k < WORDS && d < b2; k++) d += pop(A.desc[ai + k] ^ B.desc[bj + k]);
      if (d < b1) { b2 = b1; b1 = d; bi = j; } else if (d < b2) b2 = d;
    }
    best[i] = bi; d1[i] = b1; d2[i] = b2;
  }
  return { best, d1, d2 };
}

// Mutual nearest neighbours with Lowe's ratio test.
// Returns { a: Int32Array, b: Int32Array, dist: Uint16Array } index pairs.
export function matchFeatures(A, B, { ratio = 0.85, maxDist = 90 } = {}) {
  if (!A.n || !B.n) return { a: [], b: [], dist: [] };
  const ab = nearest(A, B), ba = nearest(B, A);
  const a = [], b = [], dist = [];
  for (let i = 0; i < A.n; i++) {
    const j = ab.best[i];
    if (j < 0 || ba.best[j] !== i) continue;
    if (ab.d1[i] > maxDist || ab.d1[i] > ratio * ab.d2[i]) continue;
    a.push(i); b.push(j); dist.push(ab.d1[i]);
  }
  return { a, b, dist };
}
