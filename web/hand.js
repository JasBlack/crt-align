// Hand warp: Photoshop-style puppet pins laid over the aligned layer.
//
// Each pin carries the content that was under it when it was dropped (src) to where
// it sits now (pos). A Gaussian radial-basis interpolation through every pin plus the
// four locked frame corners gives a smooth displacement D(uv) that is exact at each
// pin and fades to nothing beyond the pin's reach, so a drag pushes the neighbourhood
// and leaves the rest of the frame alone (puppet / liquify feel, not a global bend).
// The layer shows autoWarp(clamp(uv + D(uv))). Solving is a tiny linear system, so
// pins move live. Coordinates are canvas-normalized (u, v in [0,1]); distances are
// measured in aspect-corrected space so the falloff is round on screen.

export const MAX_PINS = 64;
const CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]];


export class HandWarp {
  constructor() {
    this.pins = [];      // { src: [u, v], pos: [u, v], grid?: [i, j] }
    this.aspect = 1;
    this.reach = 0.12;   // radius of a pin's influence, in frame heights
    this.sol = null;     // { n, c, w, k } — see solve()
  }

  setAspect(a) { this.aspect = a || 1; this.solve(); }
  setReach(r) { this.reach = r; this.solve(); }
  get count() { return this.pins.length; }
  get moved() { return this.pins.filter(p => Math.hypot(p.src[0] - p.pos[0], p.src[1] - p.pos[1]) > 1e-5).length; }
  get hasGrid() { return this.pins.some(p => p.grid); }

  // control points in aspect space and their displacement values (src - pos)
  controls() {
    const pts = this.pins.map(p => ({ pos: p.pos, d: [p.src[0] - p.pos[0], p.src[1] - p.pos[1]] }));
    for (const c of CORNERS) {
      if (!this.pins.some(p => Math.abs(p.pos[0] - c[0]) < 1e-6 && Math.abs(p.pos[1] - c[1]) < 1e-6)) pts.push({ pos: c, d: [0, 0] });
    }
    return pts;
  }

  solve() {
    if (!this.pins.length || !this.moved) { this.sol = null; return; }
    const pts = this.controls(), n = pts.length, A = this.aspect, k = 1 / (this.reach * this.reach);
    const cx = pts.map(p => p.pos[0] * A), cy = pts.map(p => p.pos[1]);
    // (K + lI) w = d, Gaussian kernel, two right-hand sides (du, dv)
    const M = new Float64Array(n * n), bx = new Float64Array(n), by = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const dx = cx[i] - cx[j], dy = cy[i] - cy[j];
        M[i * n + j] = Math.exp(-(dx * dx + dy * dy) * k);
      }
      M[i * n + i] += 1e-6;
      bx[i] = pts[i].d[0]; by[i] = pts[i].d[1];
    }
    const [wx, wy] = solve2(M, n, bx, by);
    const c = new Float32Array(n * 2), w = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { c[2 * i] = cx[i]; c[2 * i + 1] = cy[i]; w[2 * i] = wx[i]; w[2 * i + 1] = wy[i]; }
    this.sol = { n, c, w, k };
  }

  // displacement at a canvas point, same formula as the shader
  D(u, v) {
    const s = this.sol;
    if (!s) return [0, 0];
    const qx = u * this.aspect, qy = v;
    let dx = 0, dy = 0;
    for (let i = 0; i < s.n; i++) {
      const ex = qx - s.c[2 * i], ey = qy - s.c[2 * i + 1], f = Math.exp(-(ex * ex + ey * ey) * s.k);
      dx += s.w[2 * i] * f; dy += s.w[2 * i + 1] * f;
    }
    return [dx, dy];
  }

  // where the content shown at canvas point (u,v) comes from, before the hand warp
  source(u, v) {
    const [dx, dy] = this.D(u, v);
    return [Math.min(1, Math.max(0, u + dx)), Math.min(1, Math.max(0, v + dy))];
  }

  // Drop a pin on what's under (u,v) now: nothing moves until it's dragged.
  add(u, v) {
    if (this.pins.length >= MAX_PINS) return -1;
    this.pins.push({ src: this.source(u, v), pos: [u, v] });
    return this.pins.length - 1;
  }

  move(i, u, v) {
    const p = this.pins[i];
    if (!p) return;
    p.pos = [Math.min(1, Math.max(0, u)), Math.min(1, Math.max(0, v))];
    // a pin on the frame's edge stays on that edge
    if (p.edge) { if (p.edge[0] !== null) p.pos[0] = p.edge[0]; if (p.edge[1] !== null) p.pos[1] = p.edge[1]; }
    this.solve();
  }

  remove(i) { this.pins.splice(i, 1); this.solve(); }
  clear() { this.pins = []; this.sol = null; }

  // A lattice of pins over the frame, like Photoshop's Warp grid. Border pins slide
  // along their edge; the corners stay locked.
  addGrid(gx = 4, gy = 3) {
    this.pins = this.pins.filter(p => !p.grid);
    for (let j = 0; j <= gy; j++) for (let i = 0; i <= gx; i++) {
      const onX = i === 0 || i === gx, onY = j === 0 || j === gy;
      if (onX && onY) continue;
      if (this.pins.length >= MAX_PINS) break;
      const u = i / gx, v = j / gy;
      this.pins.push({ src: this.source(u, v), pos: [u, v], grid: [i, j],
                       edge: onX || onY ? [onX ? u : null, onY ? v : null] : null });
    }
    this.gridSize = [gx, gy];
    this.solve();
  }
  removeGrid() { this.pins = this.pins.filter(p => !p.grid); this.solve(); }

  nearest(u, v, maxDistUV) {
    let best = -1, bd = maxDistUV;
    this.pins.forEach((p, i) => {
      const d = Math.hypot((p.pos[0] - u) * this.aspect, p.pos[1] - v);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  toJSON() { return { pins: this.pins.map(p => ({ ...p })), gridSize: this.gridSize || null, reach: this.reach }; }
  fromJSON(j) {
    if (j && j.reach) this.reach = j.reach;
    this.pins = (j && j.pins ? j.pins : []).map(p => ({ ...p, src: p.src.slice(), pos: p.pos.slice() }));
    this.gridSize = j && j.gridSize ? j.gridSize : null;
    this.solve();
  }
}

// Gaussian elimination with partial pivoting, two right-hand sides.
function solve2(M, n, b1, b2) {
  const A = Float64Array.from(M), x = Float64Array.from(b1), y = Float64Array.from(b2);
  for (let k = 0; k < n; k++) {
    let p = k, mx = Math.abs(A[k * n + k]);
    for (let r = k + 1; r < n; r++) { const v = Math.abs(A[r * n + k]); if (v > mx) { mx = v; p = r; } }
    if (mx < 1e-14) continue;
    if (p !== k) {
      for (let c = 0; c < n; c++) { const t = A[k * n + c]; A[k * n + c] = A[p * n + c]; A[p * n + c] = t; }
      [x[k], x[p]] = [x[p], x[k]]; [y[k], y[p]] = [y[p], y[k]];
    }
    const piv = A[k * n + k];
    for (let r = k + 1; r < n; r++) {
      const f = A[r * n + k] / piv;
      if (!f) continue;
      for (let c = k; c < n; c++) A[r * n + c] -= f * A[k * n + c];
      x[r] -= f * x[k]; y[r] -= f * y[k];
    }
  }
  for (let k = n - 1; k >= 0; k--) {
    let sx = x[k], sy = y[k];
    for (let c = k + 1; c < n; c++) { sx -= A[k * n + c] * x[c]; sy -= A[k * n + c] * y[c]; }
    const piv = A[k * n + k] || 1e-14;
    x[k] = sx / piv; y[k] = sy / piv;
  }
  return [x, y];
}
