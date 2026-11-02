// Benchmark the browser engine (web/core) on the synthetic set, in Node.
// Usage: node bench/run_new.mjs SYNTH_DIR [quality] [lambda]
import fs from 'node:fs';
import path from 'node:path';
import { align } from '../web/core/optimize.js';
import { mapPoint } from '../web/core/model.js';

const dir = process.argv[2];
const quality = process.argv[3] || 'medium';
const lambda = process.argv[4] ? Number(process.argv[4]) : undefined;
const only = process.argv[5] || null;

const readF32 = p => { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

export function errorsFor(model, meta, truth) {
  const { W, H, grid, tw, th } = meta;
  const errs = [], errsAll = [], errsCore = [];
  const out = [0, 0];
  for (let gy = 0; gy < th; gy++) for (let gx = 0; gx < tw; gx++) {
    const x = gx * grid, y = gy * grid;
    const u = (x + 0.5) / W, v = (y + 0.5) / H;
    const k = (gy * tw + gx) * 2;
    // local filmed px per original px from neighbouring truth samples
    const k2 = (gy * tw + Math.min(tw - 1, gx + 1)) * 2;
    const sc = Math.hypot(truth[k2] - truth[k], truth[k2 + 1] - truth[k + 1]) / grid || 1;
    mapPoint(model, u, v, out);
    const e = Math.hypot(out[0] - truth[k], out[1] - truth[k + 1]) / sc;
    errsAll.push(e);
    if (u > 0.1 && u < 0.9 && v > 0.1 && v < 0.9) errs.push(e);
    if (u > 0.2 && u < 0.8 && v > 0.2 && v < 0.8) errsCore.push(e);
  }
  return { interior: stats(errs), core: stats(errsCore), all: stats(errsAll) };
}

export function stats(a) {
  const s = Float64Array.from(a).sort();
  const q = f => s[Math.min(s.length - 1, Math.floor(f * s.length))];
  return { median: q(0.5), p90: q(0.9), max: s[s.length - 1] };
}

const names = fs.readdirSync(dir).filter(f => f.endsWith('_meta.json')).map(f => f.slice(0, -10)).sort()
  .filter(n => !only || n.includes(only));
const rows = [];
for (const name of names) {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, `${name}_meta.json`)));
  const orig = { data: readF32(path.join(dir, `${name}_orig.f32`)), w: meta.ow, h: meta.oh };
  const film = { data: readF32(path.join(dir, `${name}_film.f32`)), w: meta.fw, h: meta.fh };
  const truth = readF32(path.join(dir, `${name}_truth.f32`));
  const t0 = performance.now();
  const res = await align({ orig, film, W: meta.W, H: meta.H, FW: meta.FW, FH: meta.FH, quality,
                            ...(lambda !== undefined ? { lambda } : {}) });
  const secs = (performance.now() - t0) / 1000;
  const e = errorsFor(res.model, meta, truth);
  rows.push({ name, ...e, secs, score: res.score });
  console.log(`${name.padEnd(18)} core med ${e.core.median.toFixed(2)} p90 ${e.core.p90.toFixed(2)} max ${e.core.max.toFixed(1)} | 10-90% med ${e.interior.median.toFixed(2)} p90 ${e.interior.p90.toFixed(2)} | score ${res.score.toFixed(3)} ${secs.toFixed(1)}s`);
  fs.writeFileSync(path.join(dir, `${name}_new_model.json`), JSON.stringify({
    affine: Array.from(res.model.affine), nx: res.model.nx, ny: res.model.ny,
    corners: res.model.screen.corners, bulge: res.model.screen.bulge, inset: res.model.inset,
  }));
}
const med = (reg, k) => stats(rows.map(r => r[reg][k])).median;
for (const reg of ['core', 'interior'])
  console.log(`NEW ${reg.padEnd(8)} (${quality}): median ${med(reg, 'median').toFixed(2)}px, median p90 ${med(reg, 'p90').toFixed(2)}px, worst p90 ${Math.max(...rows.map(r => r[reg].p90)).toFixed(2)}px, worst max ${Math.max(...rows.map(r => r[reg].max)).toFixed(1)}px`);
