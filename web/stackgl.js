// WebGL2 renderer for stack mode: draws photos through their reference -> image
// homographies. One full-screen triangle per layer; blending does the rest.
//   draw(layer)     one image (alpha for onion skin)
//   mean            additive blend of every included image with weight 1/n
//   diff            |a - b| of two images, single pass

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D uTex, uTex2;
uniform mat3 uH, uH2;          // reference px -> image px (full res)
uniform vec2 uSize, uSize2;    // image full-res size
uniform vec3 uView;            // reference px = (buffer px - xy) / z
uniform vec2 uViewport;
uniform int uMode;             // 0 image, 1 difference, 2 additive (mean)
uniform float uAlpha;
uniform float uGain;
out vec4 outColor;

vec4 img(sampler2D t, mat3 H, vec2 size, vec2 p) {
  vec3 q = H * vec3(p, 1.0);
  if (q.z <= 0.0) return vec4(0.0);
  vec2 uv = (q.xy / q.z) / size;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec4(0.0);
  return vec4(texture(t, uv).rgb, 1.0);
}

void main() {
  vec2 buf = vec2(gl_FragCoord.x, uViewport.y - gl_FragCoord.y);
  vec2 p = (buf - uView.xy) / uView.z;
  vec4 a = img(uTex, uH, uSize, p);
  if (uMode == 1) {
    vec4 b = img(uTex2, uH2, uSize2, p);
    float m = a.a * b.a;
    vec3 d = abs(a.rgb - b.rgb);
    outColor = vec4(d * m, m);
  } else if (uMode == 2) {
    outColor = vec4(a.rgb * a.a * uGain, a.a * uGain);
  } else {
    outColor = vec4(a.rgb, a.a * uAlpha);
  }
}`;

export class StackGL {
  constructor(canvas, { preserve = true } = {}) {
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: preserve, antialias: false, alpha: true });
    if (!gl) throw new Error('This browser has no WebGL2. Use a current Chrome, Edge, Firefox or Safari.');
    this.gl = gl; this.canvas = canvas;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    this.p = p;
    this.u = {};
    for (const n of ['uTex', 'uTex2', 'uH', 'uH2', 'uSize', 'uSize2', 'uView', 'uViewport', 'uMode', 'uAlpha', 'uGain']) this.u[n] = gl.getUniformLocation(p, n);
    this.vao = gl.createVertexArray();
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  }

  // source: ImageBitmap / canvas. Returns a texture handle.
  texture(source) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  free(t) { if (t) this.gl.deleteTexture(t); }

  begin(width, height) {
    const gl = this.gl;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.p);
    gl.bindVertexArray(this.vao);
    gl.uniform2f(this.u.uViewport, width, height);
    gl.uniform1i(this.u.uTex, 0); gl.uniform1i(this.u.uTex2, 1);
    gl.enable(gl.BLEND);
    this.w = width; this.h = height;
  }

  // layer: { tex, H (row-major 9), W, H: size }
  _bind(unit, L, hLoc, sLoc) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, L.tex);
    gl.uniformMatrix3fv(hLoc, true, new Float32Array(L.map));
    gl.uniform2f(sLoc, L.w, L.h);
  }

  draw(L, view, { alpha = 1 } = {}) {
    const gl = this.gl;
    this._bind(0, L, this.u.uH, this.u.uSize);
    gl.uniform3f(this.u.uView, view.x, view.y, view.z);
    gl.uniform1i(this.u.uMode, 0); gl.uniform1f(this.u.uAlpha, alpha);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  mean(layers, view) {
    const gl = this.gl;
    gl.uniform3f(this.u.uView, view.x, view.y, view.z);
    gl.uniform1i(this.u.uMode, 2); gl.uniform1f(this.u.uGain, 1 / Math.max(1, layers.length));
    gl.blendFunc(gl.ONE, gl.ONE);
    for (const L of layers) { this._bind(0, L, this.u.uH, this.u.uSize); gl.drawArrays(gl.TRIANGLES, 0, 3); }
  }

  diff(A, B, view) {
    const gl = this.gl;
    this._bind(0, A, this.u.uH, this.u.uSize);
    this._bind(1, B, this.u.uH2, this.u.uSize2);
    gl.uniform3f(this.u.uView, view.x, view.y, view.z);
    gl.uniform1i(this.u.uMode, 1);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Render one layer into an offscreen RGBA8 target of w x h and return the pixels
  // top row first (for export).
  renderToPixels(L, view, w, h) {
    const gl = this.gl;
    const fb = gl.createFramebuffer(), rb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
    try {
      this.begin(w, h);
      gl.disable(gl.BLEND);
      this.draw(L, view);
      const px = new Uint8ClampedArray(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
      // flip rows (GL reads bottom-up)
      const row = w * 4, tmp = new Uint8ClampedArray(row);
      for (let y = 0; y < h >> 1; y++) {
        const a = y * row, b = (h - 1 - y) * row;
        tmp.set(px.subarray(a, a + row)); px.copyWithin(a, b, b + row); px.set(tmp, b);
      }
      return px;
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteFramebuffer(fb); gl.deleteRenderbuffer(rb);
    }
  }
}
