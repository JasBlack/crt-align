// WebGL2 renderer: evaluates the warp model per pixel (same math as core/model.js)
// and composites the warped filmed layer against the original.

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uFilm;
uniform sampler2D uOrig;
uniform highp sampler2D uGrid;
uniform vec2 uViewport;       // drawing buffer size, px
uniform vec3 uView;           // canvas px = (frag px - xy) / z
uniform vec2 uCanvas;         // canvas size in canvas px (original W,H; or filmed FW,FH in raw mode)
uniform vec2 uFilmSize;       // model filmed size (FW, FH)
uniform float uH[8];
uniform vec4 uBulge;
uniform vec2 uN0, uN1, uN2, uN3;
uniform float uInset;
uniform vec2 uMargin;
uniform float uA[6];
uniform ivec2 uGridN;
uniform int uMode;            // 0 layer 1 blend 2 difference 3 checker 4 swipe 5 edges 6 original 7 raw filmed
uniform float uOpacity;
uniform float uTile;
uniform float uSwipe;
uniform float uHasOrig;
uniform float uHasModel;
uniform vec3 uBg;
uniform vec4 uDot;            // cutting-mat dots: rgb + strength
uniform float uDpr;
uniform vec3 uLine;           // overlay line colour (edges, swipe)
uniform int uEdge;            // 0 pin corners (fill frame), 1 exact (transparent beyond the raster)
uniform int uExport;
uniform int uHandN;           // hand warp: Gaussian RBF controls (see hand.js)
uniform vec4 uHand[68];       // xy = control point (aspect space), zw = weights
uniform float uHandK;         // 1 / reach^2
uniform float uAspect;          // 1 while rendering the deliverable: true transparency
out vec4 outColor;

float sm01(float t) { t = clamp(t, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float env(float x, float m) { return m <= 0.0 ? 1.0 : sm01(x / m) * sm01((1.0 - x) / m); }
vec4 bw(float f) {
  float f2 = f * f, f3 = f2 * f, g = 1.0 - f;
  return vec4(g * g * g, 3.0 * f3 - 6.0 * f2 + 4.0, -3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0, f3) / 6.0;
}

vec2 gP;  // screen parameter of the last mapUV call
vec2 mapUV(vec2 uv) {
  float su = uv.x * float(uGridN.x), sv = uv.y * float(uGridN.y);
  int kx = clamp(int(floor(su)), 0, uGridN.x - 1), ky = clamp(int(floor(sv)), 0, uGridN.y - 1);
  vec4 wu = bw(su - float(kx)), wv = bw(sv - float(ky));
  vec2 b = vec2(0.0);
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      b += wu[i] * wv[j] * texelFetch(uGrid, ivec2(kx + i, ky + j), 0).rg;
    }
  }
  vec2 d = uv - 0.5;
  vec2 st = vec2(0.5 + uA[0] * d.x + uA[1] * d.y + uA[2], 0.5 + uA[3] * d.x + uA[4] * d.y + uA[5]) + b;
  if (uEdge == 0) st = uv + vec2(env(uv.x, uMargin.x), env(uv.y, uMargin.y)) * (st - uv);
  vec2 p = uInset + st * (1.0 - 2.0 * uInset);
  gP = p;
  float s = p.x, t = p.y;
  float D = uH[6] * s + uH[7] * t + 1.0;
  vec2 X = vec2(uH[0] * s + uH[1] * t + uH[2], uH[3] * s + uH[4] * t + uH[5]) / D;
  float ss = 4.0 * s * (1.0 - s), tt = 4.0 * t * (1.0 - t);
  X += ss * (1.0 - t) * uBulge.x * uN0 + tt * s * uBulge.y * uN1 + ss * t * uBulge.z * uN2 + tt * (1.0 - s) * uBulge.w * uN3;
  return X;
}

vec3 film(vec2 X) {
  vec2 tc = (X + 0.5) / uFilmSize;
  if (tc.x < 0.0 || tc.y < 0.0 || tc.x > 1.0 || tc.y > 1.0) return vec3(0.0);
  return texture(uFilm, tc).rgb;
}

// smooth value noise, fixed seed: dot brightness drifts in soft patches
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  float a = hash(i), b = hash(i + vec2(1, 0)), c = hash(i + vec2(0, 1)), d = hash(i + vec2(1, 1));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
vec3 mat(vec2 fp) {
  float sp = 24.0 * uDpr;
  vec2 g = fp / sp, cell = floor(g);
  float dist = length((fract(g) - 0.5) * sp);
  float n = 0.65 * vnoise(cell / 9.0) + 0.35 * vnoise(cell / 3.5 + 40.0);
  float a = 0.02 + 0.14 * pow(clamp((n - 0.25) / 0.6, 0.0, 1.0), 1.6);
  float r = 1.6 * uDpr;
  float dotMask = 1.0 - smoothstep(r - 0.6, r + 0.6, dist);
  return mix(uBg, uDot.rgb, dotMask * a * uDot.a);
}

vec2 handD(vec2 uv) {
  vec2 q = vec2(uv.x * uAspect, uv.y);
  vec2 d = vec2(0.0);
  for (int i = 0; i < 68; i++) {
    if (i >= uHandN) break;
    vec2 e = q - uHand[i].xy;
    d += uHand[i].zw * exp(-dot(e, e) * uHandK);
  }
  return d;
}

float lumOrig(vec2 uv) { return dot(texture(uOrig, uv).rgb, vec3(0.299, 0.587, 0.114)); }

void main() {
  vec2 fp = vec2(gl_FragCoord.x, uViewport.y - gl_FragCoord.y);
  vec2 cp = (fp - uView.xy) / uView.z;
  vec2 uv = cp / uCanvas;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
    outColor = vec4(mat(fp), 1.0);
    return;
  }
  if (uMode == 7) { outColor = vec4(film(cp), 1.0); return; }
  vec3 o = uHasOrig > 0.5 ? texture(uOrig, uv).rgb : vec3(0.0);
  if (uMode == 6 || uHasModel < 0.5) { outColor = vec4(o, 1.0); return; }
  vec2 uvh = uHandN > 0 ? clamp(uv + handD(uv), 0.0, 1.0) : uv;
  vec3 f = film(mapUV(uvh));
  float a = 1.0;
  if (uEdge == 1 && (gP.x < 0.0 || gP.y < 0.0 || gP.x > 1.0 || gP.y > 1.0)) a = 0.0;
  if (uMode == 0) {
    if (a < 0.5) {
      // transparent in the export; a dim checker in the viewer
      vec2 c2 = floor(fp / 8.0);
      outColor = uExport == 1 ? vec4(0.0) : vec4(vec3(0.16 + 0.05 * mod(c2.x + c2.y, 2.0)), 1.0);
      return;
    }
    outColor = vec4(f, 1.0);
    return;
  }
  if (a < 0.5) f = o;
  vec3 c = f;
  if (uMode == 1) c = mix(o, f, uOpacity);
  else if (uMode == 2) c = clamp(abs(o - f) * 1.6, 0.0, 1.0);
  else if (uMode == 3) {
    vec2 q = floor(cp / uTile);
    c = mod(q.x + q.y, 2.0) < 0.5 ? o : f;
  } else if (uMode == 4) {
    float edge = abs(uv.x - uSwipe) * uCanvas.x * uView.z;
    c = uv.x < uSwipe ? o : f;
    if (edge < 1.0) c = uLine;
  } else if (uMode == 5) {
    float px = max(1.0, 1.25 / uView.z);
    vec2 e = vec2(px) / uCanvas;
    float gx = lumOrig(uv + vec2(e.x, 0.0)) - lumOrig(uv - vec2(e.x, 0.0));
    float gy = lumOrig(uv + vec2(0.0, e.y)) - lumOrig(uv - vec2(0.0, e.y));
    float g = smoothstep(0.06, 0.16, length(vec2(gx, gy)));
    c = mix(f * 0.85, uLine, g);
  }
  outColor = vec4(c, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export class Renderer {
  constructor(canvas, { preserve = false, alpha = false } = {}) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: preserve, alpha, premultipliedAlpha: false });
    if (!gl) throw new Error('This browser has no WebGL2, which the warp needs.');
    this.gl = gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.loc = {};
    const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(prog, i).name.replace(/\[0\]$/, '');
      this.loc[name] = gl.getUniformLocation(prog, name);
    }
    this.vao = gl.createVertexArray();
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.film = this.tex(); this.orig = this.tex(); this.grid = this.tex(true);
    this.hasOrig = false; this.hasModel = false;
    this.hand = null;
    this.filmSize = [1, 1];
  }

  tex(nearest = false) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, nearest ? gl.NEAREST : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, nearest ? gl.NEAREST : gl.LINEAR);
    return t;
  }

  // Too-large sources are scaled to the GPU limit; sampling uses normalized coords,
  // so the model keeps its full-resolution units.
  fit(source, w, h) {
    const k = Math.min(1, this.maxTex / Math.max(w, h));
    if (k >= 1) return source;
    const c = document.createElement('canvas');
    c.width = Math.floor(w * k); c.height = Math.floor(h * k);
    c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
    return c;
  }

  upload(t, source, w, h, mips) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.fit(source, w, h));
    if (mips) {
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    }
  }

  setOriginal(source, w, h) { this.upload(this.orig, source, w, h, true); this.hasOrig = true; this.origSize = [w, h]; }
  setFilm(source, w, h) { this.upload(this.film, source, w, h, true); this.filmSize = [w, h]; }

  setModel(m) {
    const gl = this.gl;
    if (!m) { this.hasModel = false; return; }
    this.model = m;
    this.hasModel = true;
    const gw = m.nx + 3, gh = m.ny + 3, data = new Float32Array(gw * gh * 2);
    for (let i = 0; i < gw * gh; i++) { data[2 * i] = m.cx[i]; data[2 * i + 1] = m.cy[i]; }
    gl.bindTexture(gl.TEXTURE_2D, this.grid);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, gw, gh, 0, gl.RG, gl.FLOAT, data);
  }

  // hand warp solution from HandWarp (or null), aspect = W / H of the canvas
  setHand(sol) { this.hand = sol; }

  // view: { x, y, z } maps canvas px -> buffer px as buf = canvas * z + (x, y)
  render({ mode = 1, opacity = 0.5, tile = 64, swipe = 0.5, view, canvasSize, bg = [0.08, 0.082, 0.09],
           dots = [1, 1, 1, 1], dpr = 1, line = [0.224, 1, 0.533],
           width = this.canvas.width, height = this.canvas.height, screenOverride = null, edge = 0,
           exportPass = false }) {
    const gl = this.gl, L = this.loc, m = this.model;
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.film); gl.uniform1i(L.uFilm, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.orig); gl.uniform1i(L.uOrig, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.grid); gl.uniform1i(L.uGrid, 2);
    gl.uniform2f(L.uViewport, width, height);
    gl.uniform3f(L.uView, view.x, view.y, view.z);
    gl.uniform2f(L.uCanvas, canvasSize[0], canvasSize[1]);
    gl.uniform1i(L.uMode, mode);
    gl.uniform1f(L.uOpacity, opacity);
    gl.uniform1f(L.uTile, tile);
    gl.uniform1f(L.uSwipe, swipe);
    gl.uniform1f(L.uHasOrig, this.hasOrig ? 1 : 0);
    gl.uniform1f(L.uHasModel, this.hasModel ? 1 : 0);
    gl.uniform3f(L.uBg, bg[0], bg[1], bg[2]);
    gl.uniform4f(L.uDot, dots[0], dots[1], dots[2], dots[3]);
    gl.uniform1f(L.uDpr, dpr);
    gl.uniform3f(L.uLine, line[0], line[1], line[2]);
    gl.uniform1i(L.uEdge, edge);
    gl.uniform1i(L.uExport, exportPass ? 1 : 0);
    const H = this.hand;
    gl.uniform1i(L.uHandN, H ? H.n : 0);
    gl.uniform1f(L.uAspect, canvasSize[0] / canvasSize[1]);
    if (H) {
      const packed = new Float32Array(68 * 4);
      for (let i = 0; i < H.n; i++) {
        packed[4 * i] = H.c[2 * i]; packed[4 * i + 1] = H.c[2 * i + 1];
        packed[4 * i + 2] = H.w[2 * i]; packed[4 * i + 3] = H.w[2 * i + 1];
      }
      gl.uniform4fv(L.uHand, packed);
      gl.uniform1f(L.uHandK, H.k);
    }
    if (m) {
      const S = screenOverride || m.screen;
      gl.uniform2f(L.uFilmSize, m.FW, m.FH);
      gl.uniform1fv(L.uH, S.H.slice(0, 8));
      gl.uniform4f(L.uBulge, S.bulge[0], S.bulge[1], S.bulge[2], S.bulge[3]);
      gl.uniform2f(L.uN0, S.N[0][0], S.N[0][1]); gl.uniform2f(L.uN1, S.N[1][0], S.N[1][1]);
      gl.uniform2f(L.uN2, S.N[2][0], S.N[2][1]); gl.uniform2f(L.uN3, S.N[3][0], S.N[3][1]);
      gl.uniform1f(L.uInset, m.inset);
      gl.uniform2f(L.uMargin, m.margin[0], m.margin[1]);
      gl.uniform1fv(L.uA, Array.from(m.affine));
      gl.uniform2i(L.uGridN, m.nx, m.ny);
    } else {
      gl.uniform2f(L.uFilmSize, this.filmSize[0], this.filmSize[1]);
      gl.uniform2i(L.uGridN, 1, 1);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Render only the warped layer at full original resolution into this canvas.
  renderLayer(W, H, { screenOverride = null, edge = 0 } = {}) {
    if (this.canvas.width !== W) this.canvas.width = W;
    if (this.canvas.height !== H) this.canvas.height = H;
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    this.render({ mode: 0, view: { x: 0, y: 0, z: 1 }, canvasSize: [W, H], width: W, height: H, screenOverride, edge,
                  exportPass: true });
  }
}
