// Precompute the alignment shipped with the app's sample pair.
// Usage: node bench/make_sample.mjs RAW_DIR
import fs from 'node:fs';
import { align } from '../web/core/optimize.js';
import { modelToJSON } from '../web/core/model.js';
const dir = process.argv[2];
const meta = JSON.parse(fs.readFileSync(`${dir}/meta.json`));
const rd = p => { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const res = await align({
  orig: { data: rd(`${dir}/o.f32`), w: meta.W, h: meta.H },
  film: { data: rd(`${dir}/f.f32`), w: meta.FW, h: meta.FH },
  W: meta.W, H: meta.H, FW: meta.FW, FH: meta.FH, quality: 'fine',
});
fs.writeFileSync('web/sample/alignment.json', JSON.stringify({
  app: 'crt-align', version: 1, edge: 0, score: res.score, model: modelToJSON(res.model),
}));
console.log('score', res.score.toFixed(3), 'margin', res.model.margin.map(v => v.toFixed(3)));
