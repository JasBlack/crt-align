// Detect the lit CRT raster in a filmed photo: 4 corners + bulge of each edge.

import { resize, blur, otsu } from './image.js';
import { makeScreen, screenEval } from './model.js';

const WORK = 800;

function morph(bin, w, h, r, dilate) {
  // separable square max/min filter
  const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
  const want = dilate ? 1 : 0;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = dilate ? 0 : 1;
      for (let k = -r; k <= r && v !== want; k++) {
        const xx = x + k;
        if (xx < 0 || xx >= w) continue;
        if (bin[row + xx] === want) v = want;
      }
      tmp[row + x] = v;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let v = dilate ? 0 : 1;
      for (let k = -r; k <= r && v !== want; k++) {
        const yy = y + k;
        if (yy < 0 || yy >= h) continue;
        if (tmp[yy * w + x] === want) v = want;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function largestComponent(bin, w, h) {
  const label = new Int32Array(w * h).fill(-1);
  const q = new Int32Array(w * h);
  let best = -1, bestSize = 0, cur = 0;
  for (let i = 0; i < w * h; i++) {
    if (!bin[i] || label[i] >= 0) continue;
    let head = 0, tail = 0, size = 0;
    q[tail++] = i; label[i] = cur;
    while (head < tail) {
      const p = q[head++]; size++;
      const x = p % w, y = (p - x) / w;
      if (x > 0 && bin[p - 1] && label[p - 1] < 0) { label[p - 1] = cur; q[tail++] = p - 1; }
      if (x < w - 1 && bin[p + 1] && label[p + 1] < 0) { label[p + 1] = cur; q[tail++] = p + 1; }
      if (y > 0 && bin[p - w] && label[p - w] < 0) { label[p - w] = cur; q[tail++] = p - w; }
      if (y < h - 1 && bin[p + w] && label[p + w] < 0) { label[p + w] = cur; q[tail++] = p + w; }
    }
    if (size > bestSize) { bestSize = size; best = cur; }
    cur++;
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = label[i] === best ? 1 : 0;
  // fill holes: background reachable from the border stays background
  const reach = new Uint8Array(w * h);
  let head = 0, tail = 0;
  const push = p => { if (!out[p] && !reach[p]) { reach[p] = 1; q[tail++] = p; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (head < tail) {
    const p = q[head++], x = p % w, y = (p - x) / w;
    if (x > 0) push(p - 1);
    if (x < w - 1) push(p + 1);
    if (y > 0) push(p - w);
    if (y < h - 1) push(p + w);
  }
  for (let i = 0; i < w * h; i++) if (!reach[i]) out[i] = 1;
  return { mask: out, size: bestSize };
}

function convexHull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

function densify(poly, step) {
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step));
    for (let k = 0; k < n; k++) out.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  }
  return out;
}

// Least-squares parabola y = a x^2 + b x + c, with one trimming pass.
function fitParabola(xs, ys) {
  const solve = (idx) => {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0;
    for (const i of idx) {
      const x = xs[i], y = ys[i], x2 = x * x;
      s0++; s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2;
      t0 += y; t1 += x * y; t2 += x2 * y;
    }
    // [s4 s3 s2; s3 s2 s1; s2 s1 s0] [a b c] = [t2 t1 t0]
    const M = [[s4, s3, s2, t2], [s3, s2, s1, t1], [s2, s1, s0, t0]];
    for (let i = 0; i < 3; i++) {
      let piv = i;
      for (let r = i + 1; r < 3; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
      [M[i], M[piv]] = [M[piv], M[i]];
      if (Math.abs(M[i][i]) < 1e-12) return [0, 0, 0];
      for (let r = 0; r < 3; r++) {
        if (r === i) continue;
        const f = M[r][i] / M[i][i];
        for (let c = i; c < 4; c++) M[r][c] -= f * M[i][c];
      }
    }
    return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
  };
  let idx = xs.map((_, i) => i);
  let coef = solve(idx);
  const res = idx.map(i => ys[i] - (coef[0] * xs[i] * xs[i] + coef[1] * xs[i] + coef[2]));
  const rms = Math.sqrt(res.reduce((a, r) => a + r * r, 0) / Math.max(1, res.length));
  const keep = idx.filter((i, k) => Math.abs(res[k]) <= 2.5 * rms + 0.5);
  if (keep.length >= 6) coef = solve(keep);
  return coef;
}

function edgeCurve(ci, cj, pts) {
  const ex = cj[0] - ci[0], ey = cj[1] - ci[1], L = Math.hypot(ex, ey);
  const e = [ex / L, ey / L];
  return { origin: ci, e, L, coef: [0, 0, 0], pts };
}

function curvePoint(E, x) {
  const [a, b, c] = E.coef, y = a * x * x + b * x + c;
  return [E.origin[0] + x * E.e[0] + y * E.n[0], E.origin[1] + x * E.e[1] + y * E.n[1]];
}

function curveDeriv(E, x) {
  const [a, b] = E.coef, dy = 2 * a * x + b;
  return [E.e[0] + dy * E.n[0], E.e[1] + dy * E.n[1]];
}

function intersectCurves(A, xa, B, xb) {
  for (let it = 0; it < 20; it++) {
    const pa = curvePoint(A, xa), pb = curvePoint(B, xb);
    const da = curveDeriv(A, xa), db = curveDeriv(B, xb);
    const rx = pa[0] - pb[0], ry = pa[1] - pb[1];
    // [da, -db] [dxa, dxb]^T = -r
    const det = da[0] * -db[1] - -db[0] * da[1];
    if (Math.abs(det) < 1e-9) break;
    const dxa = (-rx * -db[1] - -db[0] * -ry) / det;
    const dxb = (da[0] * -ry - -rx * da[1]) / det;
    xa += dxa; xb += dxb;
    if (Math.abs(dxa) + Math.abs(dxb) < 1e-4) break;
  }
  return { p: curvePoint(A, xa), xa, xb };
}

export function detectScreen(film) {
  const scale = Math.min(1, WORK / Math.max(film.w, film.h));
  const w = Math.max(16, Math.round(film.w * scale)), h = Math.max(16, Math.round(film.h * scale));
  const k = film.w / w; // full px per small px
  const small = blur(resize(film, w, h), 2);
  const thr = otsu(small.data);
  let bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) bin[i] = small.data[i] > thr ? 1 : 0;
  const r = Math.max(2, Math.round(0.015 * Math.max(w, h)));
  bin = morph(morph(bin, w, h, r, true), w, h, r, false);          // close
  bin = morph(morph(bin, w, h, r >> 1, false), w, h, r >> 1, true); // open
  const { mask, size } = largestComponent(bin, w, h);

  const fallback = () => {
    const mx = film.w * 0.05, my = film.h * 0.05;
    return { corners: [[mx, my], [film.w - mx, my], [film.w - mx, film.h - my], [mx, film.h - my]],
             bulge: [0, 0, 0, 0], inset: 0.02, ok: false, mask: null };
  };
  if (size < 0.05 * w * h) return fallback();

  const boundary = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (!mask[p]) continue;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
        !mask[p - 1] || !mask[p + 1] || !mask[p - w] || !mask[p + w]) boundary.push([x + 0.5, y + 0.5]);
  }
  const hull = densify(convexHull(boundary), 2);
  if (hull.length < 8) return fallback();

  const pick = f => hull.reduce((best, q) => (f(q) > f(best) ? q : best), hull[0]);
  let C = [pick(q => -q[0] - q[1]), pick(q => q[0] - q[1]), pick(q => q[0] + q[1]), pick(q => q[1] - q[0])];
  const cx = C.reduce((a, q) => a + q[0], 0) / 4, cy = C.reduce((a, q) => a + q[1], 0) / 4;

  // edges: top TL->TR, right TR->BR, bottom BL->BR, left TL->BL
  const pairs = [[0, 1], [1, 2], [3, 2], [0, 3]];
  const edges = pairs.map(([i, j]) => edgeCurve(C[i], C[j], []));
  for (const E of edges) {
    let nx = -E.e[1], ny = E.e[0];
    const mx = E.origin[0] + E.e[0] * E.L / 2 - cx, my = E.origin[1] + E.e[1] * E.L / 2 - cy;
    if (nx * mx + ny * my < 0) { nx = -nx; ny = -ny; }
    E.n = [nx, ny];
  }
  for (const q of hull) {
    let bestE = null, bestD = Infinity, bestX = 0, bestY = 0;
    for (const E of edges) {
      const dx = q[0] - E.origin[0], dy = q[1] - E.origin[1];
      const x = dx * E.e[0] + dy * E.e[1], y = dx * E.n[0] + dy * E.n[1];
      if (x < 0.12 * E.L || x > 0.88 * E.L) continue;
      if (Math.abs(y) < bestD) { bestD = Math.abs(y); bestE = E; bestX = x; bestY = y; }
    }
    if (bestE && bestD < 0.12 * bestE.L) bestE.pts.push([bestX, bestY]);
  }
  for (const E of edges) {
    if (E.pts.length < 6) return fallback();
    E.coef = fitParabola(E.pts.map(p => p[0]), E.pts.map(p => p[1]));
  }
  const [T, R, B, Lf] = edges;
  const tl = intersectCurves(T, 0, Lf, 0);
  const tr = intersectCurves(T, T.L, R, 0);
  const br = intersectCurves(R, R.L, B, B.L);
  const bl = intersectCurves(B, 0, Lf, Lf.L);
  const corners = [tl.p, tr.p, br.p, bl.p];

  // bulge = curve offset at the chord midpoint, measured along the outward normal
  const ends = [[tl.xa, tr.xa], [tr.xb, br.xa], [bl.xa, br.xb], [tl.xb, bl.xb]];
  const bulge = edges.map((E, i) => {
    const [x0, x1] = ends[i];
    const xm = (x0 + x1) / 2;
    const [a, b, c] = E.coef;
    const y = v => a * v * v + b * v + c;
    return y(xm) - (y(x0) + y(x1)) / 2;
  });

  const full = corners.map(p => [p[0] * k, p[1] * k]);
  const fullBulge = bulge.map(b => b * k);
  const area = polyArea(full);
  if (area < 0.05 * film.w * film.h || area > 0.995 * film.w * film.h || !isConvex(full)) return fallback();

  const inset = autoInset({ corners: full, bulge: fullBulge }, mask, w, h, k);
  return { corners: full, bulge: fullBulge, inset, ok: true, mask: { data: mask, w, h, k } };
}

function polyArea(c) {
  let a = 0;
  for (let i = 0; i < c.length; i++) {
    const p = c[i], q = c[(i + 1) % c.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return Math.abs(a) / 2;
}

function isConvex(c) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = c[i], b = c[(i + 1) % 4], d = c[(i + 2) % 4];
    const z = (b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0]);
    if (z === 0) continue;
    if (sign && Math.sign(z) !== sign) return false;
    sign = Math.sign(z);
  }
  return true;
}

// Smallest inset whose boundary lies fully inside the lit raster (+ a small margin).
export function autoInset(scr, mask, w, h, k) {
  const S = makeScreen(scr.corners, scr.bulge);
  const out = new Float64Array(6);
  // erode the mask a little so we stay clear of the soft raster edge
  const er = morph(mask, w, h, 2, false);
  for (let inset = 0.005; inset <= 0.1; inset += 0.0025) {
    let inside = 0, total = 0;
    for (let i = 0; i <= 80; i++) {
      const f = inset + (1 - 2 * inset) * i / 80;
      for (const [s, t] of [[f, inset], [f, 1 - inset], [inset, f], [1 - inset, f]]) {
        screenEval(S, s, t, out);
        const x = Math.floor(out[0] / k), y = Math.floor(out[1] / k);
        total++;
        if (x >= 0 && y >= 0 && x < w && y < h && er[y * w + x]) inside++;
      }
    }
    if (inside >= 0.995 * total) return Math.min(0.1, inset + 0.005);
  }
  return 0.03;
}
