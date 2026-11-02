// Stack-mode accuracy on synthetic bursts (bench/stack/make_burst.py).
// Usage: node bench/stack/run.mjs DIR... [--kind homography|similarity|translate] [--method match|bright]
import fs from 'node:fs'; import path from 'node:path';
import { prepare, alignStack, cropReport } from '../../web/core/stack.js';
import { mul3, inv3, apply } from '../../web/core/geom.js';
const args = process.argv.slice(2), opt = k => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : null; };
const kind = opt('--kind') || 'homography', methodArg = opt('--method');
for (const dir of args) {
  const idx = JSON.parse(fs.readFileSync(path.join(dir, 'index.json')));
  const truth = JSON.parse(fs.readFileSync(path.join(dir, 'truth.json')));
  const method = methodArg || (truth.shots[0].sun ? 'bright' : 'match');
  const t0 = performance.now();
  const items = idx.map(e => {
    const b = fs.readFileSync(path.join(dir, e.file + '.f32'));
    const lum = { data: new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4), w: e.w, h: e.h };
    return { lum, W: e.W, H: e.H, prep: method === 'match' ? prepare(lum, e.W, e.H) : null, name: e.file };
  });
  const tPrep = performance.now();
  const { results, anchor } = await alignStack(items, { ref: 0, method, kind });
  const tAlign = performance.now();
  const errs = [];
  results.forEach((r, i) => {
    if (i === 0) return;
    const sh = truth.shots.find(s => s.file === items[i].name);
    let e;
    if (sh.sun) {
      const [u, v] = apply(r.H, ...truth.shots[0].sun);
      e = { med: Math.hypot(u - sh.sun[0], v - sh.sun[1]) }; e.max = e.med;
    } else {
      const Ht = mul3(inv3(sh.H.flat()), truth.shots[0].H.flat());  // ref px -> shot i px
      const d = [];
      for (let y = 0.2; y <= 0.8; y += 0.05) for (let x = 0.2; x <= 0.8; x += 0.05) {
        const p = [x * items[0].W, y * items[0].H], a = apply(Ht, ...p), b = apply(r.H, ...p);
        if (a[0] < 0 || a[1] < 0 || a[0] > items[i].W || a[1] > items[i].H) continue;
        d.push(Math.hypot(a[0] - b[0], a[1] - b[1]));
      }
      d.sort((a, b) => a - b);
      e = { med: d[d.length >> 1], max: d[d.length - 1] };
    }
    errs.push(e.max);
    console.log(`  ${items[i].name.padEnd(12)} ${String(r.status).padEnd(18)} inl ${String(r.inliers ?? '-').padStart(4)}  err med ${e.med?.toFixed(2)} max ${e.max?.toFixed(2)} px`);
  });
  const rep = cropReport(items, results.map(r => r.ok ? r.H : null), results.map(r => r.ok), anchor);
  const crops = rep.rows.filter(r => ['16:9', '9:16', '1:1', '4:5'].includes(r.id)).map(r => `${r.id} ${r.outW}x${r.outH}`).join('  ');
  console.log(`${path.basename(dir).padEnd(14)} worst ${Math.max(...errs).toFixed(2)} px  prep ${((tPrep - t0) / 1000).toFixed(1)}s align ${((tAlign - tPrep) / 1000).toFixed(1)}s | ${crops}`);
}
