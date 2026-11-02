// Align real pairs with the browser engine; dump a map for real_render.py.
// Usage: node bench/real_align.mjs DIR [quality]
import fs from 'node:fs'; import path from 'node:path';
import { align } from '../web/core/optimize.js';
import { mapPoint, modelToJSON } from '../web/core/model.js';
const dir = process.argv[2], quality = process.argv[3] || 'fine';
const rd = p => { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('_meta.json'))) {
  const name = f.slice(0, -10), meta = JSON.parse(fs.readFileSync(path.join(dir, f)));
  const orig = { data: rd(path.join(dir, `${name}_orig.f32`)), w: meta.ow, h: meta.oh };
  const film = { data: rd(path.join(dir, `${name}_film.f32`)), w: meta.fw, h: meta.fh };
  const t0 = performance.now();
  const res = await align({ orig, film, W: meta.W, H: meta.H, FW: meta.FW, FH: meta.FH, quality });
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  // map at preview resolution (1600 wide) for rendering
  const pw = Math.min(1600, meta.W), ph = Math.round(pw * meta.H / meta.W);
  const map = new Float32Array(pw * ph * 2), o = [0, 0];
  for (let j = 0; j < ph; j++) for (let i = 0; i < pw; i++) {
    mapPoint(res.model, (i + 0.5) / pw, (j + 0.5) / ph, o);
    map[(j * pw + i) * 2] = o[0]; map[(j * pw + i) * 2 + 1] = o[1];
  }
  fs.writeFileSync(path.join(dir, `${name}_map.f32`), Buffer.from(map.buffer));
  fs.writeFileSync(path.join(dir, `${name}_model.json`), JSON.stringify({ ...modelToJSON(res.model), pw, ph }));
  console.log(`${name.padEnd(10)} score ${res.score.toFixed(3)}  screen ${res.screenDetected ? 'ok' : 'FALLBACK'}  ${secs}s`);
}
