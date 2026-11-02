// Parallax check: synthetic building corner (bench/stack/make_corner.py). Reports the
// error at known 3D points on both walls for each model, with and without a subject box.
import fs from 'node:fs'; import path from 'node:path';
import { prepare, alignStack } from '../../web/core/stack.js';
import { apply } from '../../web/core/geom.js';
const dir = process.argv[2];
const idx = JSON.parse(fs.readFileSync(path.join(dir, 'index.json')));
const truth = JSON.parse(fs.readFileSync(path.join(dir, 'truth.json')));
const items = idx.map(e => {
  const b = fs.readFileSync(path.join(dir, e.file + '.f32'));
  const lum = { data: new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4), w: e.w, h: e.h };
  return { lum, W: e.W, H: e.H, prep: prepare(lum, e.W, e.H), name: e.file };
});
const T = items.map(it => truth.shots.find(s => s.file === it.name).pts);
const names = Object.keys(T[0]);
const xs = names.map(n => T[0][n][0]), ys = names.map(n => T[0][n][1]);
const box = { x: Math.min(...xs) - 40, y: Math.min(...ys) - 40, w: Math.max(...xs) - Math.min(...xs) + 80, h: Math.max(...ys) - Math.min(...ys) + 80 };
for (const kind of ['homography', 'similarity', 'translate']) for (const roi of [null, box]) {
  const anchor = T[0].corner_mid;
  const { results } = await alignStack(items, { ref: 0, method: 'match', kind, roi, anchor });
  const per = {};
  for (const n of names) per[n] = [];
  results.forEach((r, i) => { if (i && r.ok) for (const n of names) { const q = apply(r.H, ...T[0][n]); per[n].push(Math.hypot(q[0] - T[i][n][0], q[1] - T[i][n][1])); } });
  const ok = results.filter(r => r.ok).length - 1;
  console.log(`${kind.padEnd(10)} ${roi ? 'box ' : 'full'} ok ${ok}/4  ` + names.map(n => `${n} ${Math.max(...per[n]).toFixed(1)}`).join('  '));
}
