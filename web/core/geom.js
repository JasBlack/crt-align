// 2D projective geometry for stack mode: 3x3 matrices (row-major arrays of 9),
// model fitting (translation / similarity / homography), convex polygon clipping
// and the crop solver. DOM-free; runs in Node for the bench.

export const I3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mul3(a, b) {
  const o = new Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  }
  return o;
}

export function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!det || !isFinite(det)) return null;
  const k = 1 / det;
  return [A * k, -(b * i - c * h) * k, (b * f - c * e) * k,
          B * k, (a * i - c * g) * k, -(a * f - c * d) * k,
          C * k, -(a * h - b * g) * k, (a * e - b * d) * k];
}

export function norm3(m) { const s = m[8] || 1; return m.map(v => v / s); }

export function apply(m, x, y) {
  const w = m[6] * x + m[7] * y + m[8];
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

export const translation = (tx, ty) => [1, 0, tx, 0, 1, ty, 0, 0, 1];
export const scaling = (sx, sy = sx) => [sx, 0, 0, 0, sy, 0, 0, 0, 1];

// Local linear scale of the map at (x, y): sqrt |det J|.
export function localScale(m, x, y) {
  const w = m[6] * x + m[7] * y + m[8];
  const u = (m[0] * x + m[1] * y + m[2]) / w, v = (m[3] * x + m[4] * y + m[5]) / w;
  const j00 = (m[0] - m[6] * u) / w, j01 = (m[1] - m[7] * u) / w;
  const j10 = (m[3] - m[6] * v) / w, j11 = (m[4] - m[7] * v) / w;
  return Math.sqrt(Math.abs(j00 * j11 - j01 * j10));
}

// ---------------------------------------------------------------- fitting
// Each fit takes src/dst as flat arrays [x0,y0,x1,y1,...] and returns H with dst ≈ H(src).

export function fitTranslation(src, dst, idx = null) {
  const ids = idx || [...Array(src.length / 2).keys()];
  let tx = 0, ty = 0;
  for (const i of ids) { tx += dst[2 * i] - src[2 * i]; ty += dst[2 * i + 1] - src[2 * i + 1]; }
  return translation(tx / ids.length, ty / ids.length);
}

// Least-squares similarity (Umeyama, no reflection).
export function fitSimilarity(src, dst, idx = null) {
  const ids = idx || [...Array(src.length / 2).keys()];
  const n = ids.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (const i of ids) { sx += src[2 * i]; sy += src[2 * i + 1]; dx += dst[2 * i]; dy += dst[2 * i + 1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let a = 0, b = 0, ss = 0;
  for (const i of ids) {
    const px = src[2 * i] - sx, py = src[2 * i + 1] - sy, qx = dst[2 * i] - dx, qy = dst[2 * i + 1] - dy;
    a += px * qx + py * qy; b += px * qy - py * qx; ss += px * px + py * py;
  }
  if (ss < 1e-9) return null;
  const c = a / ss, s = b / ss;
  return [c, -s, dx - c * sx + s * sy, s, c, dy - s * sx - c * sy, 0, 0, 1];
}

// Symmetric eigen decomposition (cyclic Jacobi); returns the eigenvector of the
// smallest eigenvalue of an n x n matrix A (flat, row-major).
function minEigvec(A, n) {
  const a = A.slice(), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = v[k * n + p], vkq = v[k * n + q];
        v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  let best = 0;
  for (let i = 1; i < n; i++) if (a[i * n + i] < a[best * n + best]) best = i;
  const out = new Array(n);
  for (let k = 0; k < n; k++) out[k] = v[k * n + best];
  return out;
}

function normalizer(pts, ids) {
  let mx = 0, my = 0;
  for (const i of ids) { mx += pts[2 * i]; my += pts[2 * i + 1]; }
  mx /= ids.length; my /= ids.length;
  let d = 0;
  for (const i of ids) d += Math.hypot(pts[2 * i] - mx, pts[2 * i + 1] - my);
  const s = Math.SQRT2 / Math.max(1e-9, d / ids.length);
  return [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1];
}

// Normalized DLT homography.
export function fitHomography(src, dst, idx = null) {
  const ids = idx || [...Array(src.length / 2).keys()];
  if (ids.length < 4) return null;
  const Ts = normalizer(src, ids), Td = normalizer(dst, ids);
  const M = new Float64Array(81);
  const row = new Float64Array(9);
  const add = () => { for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) M[r * 9 + c] += row[r] * row[c]; };
  for (const i of ids) {
    const [x, y] = apply(Ts, src[2 * i], src[2 * i + 1]);
    const [u, v] = apply(Td, dst[2 * i], dst[2 * i + 1]);
    row.set([-x, -y, -1, 0, 0, 0, u * x, u * y, u]); add();
    row.set([0, 0, 0, -x, -y, -1, v * x, v * y, v]); add();
  }
  const h = minEigvec(M, 9);
  const Tdi = inv3(Td);
  if (!Tdi) return null;
  const H = mul3(Tdi, mul3(h, Ts));
  if (!H.every(isFinite) || Math.abs(H[8]) < 1e-12) return null;
  return norm3(H);
}

export function fitModel(kind, src, dst, idx) {
  if (kind === 'translate') return fitTranslation(src, dst, idx);
  if (kind === 'similarity') return fitSimilarity(src, dst, idx);
  return fitHomography(src, dst, idx);
}
export const minSamples = kind => (kind === 'translate' ? 1 : kind === 'similarity' ? 2 : 4);

// A homography is plausible for photos of the same thing: orientation preserved,
// no extreme anisotropy or perspective inside the frame.
export function sane(H, w, h) {
  if (!H) return false;
  const pts = [[0, 0], [w, 0], [w, h], [0, h]];
  for (const [x, y] of pts) if (H[6] * x + H[7] * y + H[8] <= 0) return false;
  const q = pts.map(([x, y]) => apply(H, x, y));
  let area = 0, sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    const cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (sign === 0) sign = Math.sign(cr); else if (Math.sign(cr) !== sign) return false;
    area += a[0] * b[1] - b[0] * a[1];
  }
  if (sign <= 0) return false;
  const ratio = Math.abs(area / 2) / (w * h);
  return ratio > 0.04 && ratio < 25;
}

// ---------------------------------------------------------------- RANSAC

let seed = 12345;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

export function ransac(kind, src, dst, { thresh = 3, iters = 1500, weights = null, w = 1000, h = 1000 } = {}) {
  seed = 12345;
  const n = src.length / 2, k = minSamples(kind);
  if (n < k + (kind === 'homography' ? 2 : 1)) return null;
  const t2 = thresh * thresh;
  let best = null, bestScore = -1, bestIn = null;
  const ids = new Array(k);
  let maxIt = iters;
  for (let it = 0; it < maxIt; it++) {
    for (let j = 0; j < k; j++) {
      let r; do { r = Math.floor(rand() * n); } while (ids.slice(0, j).includes(r));
      ids[j] = r;
    }
    const H = fitModel(kind, src, dst, ids);
    if (!H || (kind === 'homography' && !sane(H, w, h))) continue;
    let score = 0;
    const inl = [];
    for (let i = 0; i < n; i++) {
      const [u, v] = apply(H, src[2 * i], src[2 * i + 1]);
      const e = (u - dst[2 * i]) ** 2 + (v - dst[2 * i + 1]) ** 2;
      if (e < t2) { inl.push(i); score += (weights ? weights[i] : 1) * (1 - e / t2); }
    }
    if (score > bestScore) {
      bestScore = score; best = H; bestIn = inl;
      const ratio = inl.length / n;
      const need = Math.log(0.002) / Math.log(1 - Math.max(1e-6, ratio ** k));
      maxIt = Math.min(iters, Math.max(60, Math.ceil(need)));
    }
  }
  if (!best || bestIn.length < k + 1) return null;
  // polish: refit on inliers, re-collect, twice
  let H = best, inl = bestIn;
  for (let r = 0; r < 3; r++) {
    const H2 = fitModel(kind, src, dst, inl);
    if (!H2 || (kind === 'homography' && !sane(H2, w, h))) break;
    const inl2 = [];
    for (let i = 0; i < n; i++) {
      const [u, v] = apply(H2, src[2 * i], src[2 * i + 1]);
      if ((u - dst[2 * i]) ** 2 + (v - dst[2 * i + 1]) ** 2 < t2) inl2.push(i);
    }
    if (inl2.length < inl.length * 0.9) break;
    H = H2; inl = inl2;
  }
  return { H, inliers: inl };
}

export function residuals(H, src, dst, idx) {
  return idx.map(i => {
    const [u, v] = apply(H, src[2 * i], src[2 * i + 1]);
    return Math.hypot(u - dst[2 * i], v - dst[2 * i + 1]);
  });
}

// ---------------------------------------------------------------- polygons
// Polygons are arrays of [x, y]. Half-plane: { n: [nx, ny], d } meaning n·p >= d.

export function quadOf(H, w, h) {
  return [[0, 0], [w, 0], [w, h], [0, h]].map(([x, y]) => apply(H, x, y));
}

export function polyArea(P) {
  let a = 0;
  for (let i = 0; i < P.length; i++) { const p = P[i], q = P[(i + 1) % P.length]; a += p[0] * q[1] - q[0] * p[1]; }
  return a / 2;
}

export function clipHalf(P, n, d) {
  const out = [];
  for (let i = 0; i < P.length; i++) {
    const p = P[i], q = P[(i + 1) % P.length];
    const fp = n[0] * p[0] + n[1] * p[1] - d, fq = n[0] * q[0] + n[1] * q[1] - d;
    if (fp >= 0) out.push(p);
    if ((fp >= 0) !== (fq >= 0)) {
      const t = fp / (fp - fq);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  }
  return out;
}

// Inward half-planes of a convex polygon (either winding).
export function halfPlanes(P) {
  const s = Math.sign(polyArea(P)) || 1, out = [];
  for (let i = 0; i < P.length; i++) {
    const p = P[i], q = P[(i + 1) % P.length];
    let nx = -(q[1] - p[1]) * s, ny = (q[0] - p[0]) * s;
    const L = Math.hypot(nx, ny);
    if (L < 1e-9) continue;
    nx /= L; ny /= L;
    out.push({ n: [nx, ny], d: nx * p[0] + ny * p[1] });
  }
  return out;
}

export function intersectAll(polys) {
  if (!polys.length) return [];
  let P = polys[0].slice();
  if (polyArea(P) < 0) P.reverse();
  for (let k = 1; k < polys.length && P.length; k++) {
    for (const hp of halfPlanes(polys[k])) { P = clipHalf(P, hp.n, hp.d); if (!P.length) break; }
  }
  return P;
}

export function inside(P, x, y) {
  return halfPlanes(P).every(hp => hp.n[0] * x + hp.n[1] * y >= hp.d - 1e-6);
}

function closestInPoly(P, x, y) {
  if (inside(P, x, y)) return [x, y];
  let best = null, bd = Infinity;
  for (let i = 0; i < P.length; i++) {
    const a = P[i], b = P[(i + 1) % P.length];
    const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
    const t = L ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / L)) : 0;
    const px = a[0] + t * dx, py = a[1] + t * dy, d = (px - x) ** 2 + (py - y) ** 2;
    if (d < bd) { bd = d; best = [px, py]; }
  }
  return best;
}

// Largest axis-aligned rectangle of aspect r (= width / height) inside the convex
// polygon V. place:
//   'centre'        anchor exactly in the middle
//   [fx, fy]        anchor at that fraction of the frame (thirds: [1/3, 1/3] ...)
//   'free'          largest anywhere that still contains the anchor (with a margin),
//                   as centred on the anchor as that size allows
// Returns { x, y, w, h } in V's coordinates, or null.
export function bestRect(V, r, anchor, place = 'centre') {
  if (!V || V.length < 3) return null;
  const hps = halfPlanes(V);
  const [ax, ay] = anchor;
  if (!inside(V, ax, ay)) return null;
  const kOf = hp => (r * Math.abs(hp.n[0]) + Math.abs(hp.n[1])) / 2;
  if (place !== 'free') {
    const [fx, fy] = place === 'centre' ? [0.5, 0.5] : place;
    const g = [(0.5 - fx) * r, 0.5 - fy];
    let s = Infinity;
    for (const hp of hps) {
      const den = kOf(hp) - (hp.n[0] * g[0] + hp.n[1] * g[1]);
      const num = hp.n[0] * ax + hp.n[1] * ay - hp.d;
      if (den > 1e-12) s = Math.min(s, num / den);
    }
    if (!isFinite(s) || s <= 0) return null;
    const cx = ax + s * g[0], cy = ay + s * g[1];
    return { x: cx - (r * s) / 2, y: cy - s / 2, w: r * s, h: s };
  }
  // free: binary search on height s; feasible centres = V shrunk by the rect, also
  // keeping the anchor at least 12% of the frame away from every edge.
  const m = 0.12;
  const feas = s => {
    let P = V.slice();
    if (polyArea(P) < 0) P.reverse();
    for (const hp of hps) {
      P = clipHalf(P, hp.n, hp.d + s * kOf(hp));
      if (!P.length) return null;
    }
    const hx = r * s * (0.5 - m), hy = s * (0.5 - m);
    for (const [n, d] of [[[1, 0], ax - hx], [[-1, 0], -(ax + hx)], [[0, 1], ay - hy], [[0, -1], -(ay + hy)]]) {
      P = clipHalf(P, n, d);
      if (!P.length) return null;
    }
    return P;
  };
  let lo = 0, hi = 0;
  for (const p of V) hi = Math.max(hi, Math.abs(p[1] - ay) * 2, Math.abs(p[0] - ax) * 2 / r);
  let bestP = null;
  for (let it = 0; it < 48; it++) {
    const mid = (lo + hi) / 2, P = feas(mid);
    if (P && P.length >= 1) { lo = mid; bestP = P; } else hi = mid;
  }
  if (!bestP || lo <= 0) return null;
  const s = lo * 0.9999;
  const P = feas(s) || bestP;
  const [cx, cy] = P.length >= 3 ? closestInPoly(P, ax, ay) : P[0];
  return { x: cx - (r * s) / 2, y: cy - s / 2, w: r * s, h: s };
}

// Scale a rect about the anchor (keeps the anchor at the same relative spot and
// stays inside any convex region that contains both the rect and the anchor).
export function zoomRect(rect, anchor, z) {
  if (!rect || z >= 1) return rect;
  const [ax, ay] = anchor;
  return { x: ax + (rect.x - ax) * z, y: ay + (rect.y - ay) * z, w: rect.w * z, h: rect.h * z };
}
