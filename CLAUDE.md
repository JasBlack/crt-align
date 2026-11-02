# CLAUDE.md — CRT Align

## What this is

Aligns a **filmed CRT / analogue-video-synth capture** (photo or video) back onto the
**original image or video** it was made from, so the glitched capture can be laid over
the source as a pixel-registered glitch layer. Output always has the original's exact
pixel size. Handles stills and video, across different resolutions and frame rates.

Two implementations:

- **`web/`** — browser app, the current tool. Pure client-side ES modules, no build.
- **`crt_align.py`** — earlier Python/OpenCV CLI. Kept for batch use; its algorithm is
  older and less accurate (see Accuracy).

## Tech stack

- Browser app: vanilla JS (ES modules), WebGL2 (warp + composite shader), a module Web
  Worker for the solver, [mediabunny](https://mediabunny.dev) `1.61.4` from jsDelivr for
  WebCodecs video decode/encode (loaded only when a video is opened). Fonts from Google
  Fonts. No framework, no bundler, no npm dependencies.
- Bench / tooling: Node 18+ (runs `web/core` directly), Python 3.10+ with
  `opencv-python` + `numpy` (synthetic data, old-algorithm comparison).
- CLI: Python 3.12, OpenCV 5 (`opencv-python>=4.8`), NumPy.

## Folder structure

```
web/                       the browser app (deploy this folder as-is)
  index.html               page shell (full HTML document)
  app.css                  styles; single dark theme on purpose
  app.js                   UI controller: loading, view/pan/zoom, Pin + Raster tools,
                           video transport + sync, export, undo, alignment JSON
  engine.js                main-thread client for the solver worker (+ cancel)
  hand.js                  hand warp: puppet pins -> Gaussian RBF displacement (live)
  media.js                 image/video loading, luma extraction, file saving
                           (claude.ai `downloads` capability or a plain download)
  gl/renderer.js           WebGL2 shader: evaluates the warp per pixel, view modes
  core/                    the alignment engine — DOM-free, also runs in Node
    image.js               gray float images: resize, anti-aliased downscale, blur, pyramid
    model.js               warp model: screen homography + edge bulges, inset,
                           residual affine, B-spline mesh, edge bands; JSON I/O
    screen.js              CRT raster detection (corners, bulges, auto inset)
    optimize.js            NGF similarity, analytic gradients, L-BFGS, global search,
                           align() / refine() / scoreModel()
    sync.js                temporal sync candidates from per-frame luma signatures
    worker.js              runs align/refine/score off the main thread
  sample/                  sample pair + precomputed alignment shown on first open
bench/                     accuracy benchmark + helpers (see below)
crt_align.py               Python CLI (older algorithm)
requirements.txt           CLI deps
package.json               "type": "module" (lets Node import web/core) + scripts
.claude/launch.json        Claude Code preview config (serves web/ on :5140)
viking_points.csv          example manual-anchor file for the CLI's --points
base_*.jpg, new_00367_.png originals of the test set
IDG_*.JPG                  filmed CRT photos of them
*.mp4                      test videos (gitignored; large)
aligned/                   CLI outputs (gitignored)
```

Test pairs (original ↔ filmed): `new_00367_.png` ↔ `IDG_..._153548_887` (viking),
`base_26` ↔ `..._154202_498` (leopard), `base_29` ↔ `..._154207_784` (petronas),
`base_49` ↔ `..._154210_695` (chicago), `base_68` ↔ `..._154215_413` (cartoon; also the
web sample). Videos: `3253799-uhd_3840_2160_25fps.mp4` ↔ `crt-thing-3.mp4` (people,
offset ≈ −0.07 s), `14737157_3840_2160_60fps.mp4` ↔ `crt-thing-1.mp4` (traffic).
`13802596-*` duplicates `crt-thing-1`; `crt-thing-2` has no source here.

## Commands

```bash
# browser app
npx -y serve web -l 5140            # then http://localhost:5140

# accuracy benchmark (synthetic CRT captures with ground truth)
python bench/make_synth.py OUT       # 10 cases, ~1 min, ~150 MB
python bench/prep_raw.py OUT         # raw luma for Node
node bench/run_new.mjs OUT fine      # browser engine
python bench/run_old.py OUT          # Python CLI's pin_align

# real pairs through the browser engine, rendered previews + QC overlays
python bench/real_prep.py OUT name=orig.jpg,filmed.jpg ...
node bench/real_align.mjs OUT fine && python bench/real_render.py OUT

# regenerate web/sample/alignment.json (see bench/make_sample.mjs header)
node bench/make_sample.mjs RAW_DIR

# Python CLI
pip install -r requirements.txt
python crt_align.py ORIGINAL FILMED [-o OUT] [--overlay] [--points FILE]
```

No build step anywhere.

## Deployment

`web/` is a static site: GitHub Pages (deploy branch, folder `/web`), Netlify, Vercel,
or any static host. It was also published as a private claude.ai artifact; there it
saves exports through the `downloads` capability (declared as
`capabilities: {downloads: true}`), elsewhere through an ordinary `<a download>`.

## Environment variables / API keys

None. No network calls except jsDelivr (mediabunny) and Google Fonts. Nothing secret.

## Current status

Works (verified):
- Image↔image, image↔video, video↔video; output at the original's size.
- Auto alignment on all 5 real pairs, including the heavy viking glitch.
- View modes Layer / Blend / Diff / Checker / Swipe / Edges / Original; zoom + pan.
- Live hand warp (Pin tool): puppet pins, drag to warp in real time (solve + render
  ≈ 0.2 ms at 1600×900), mesh-grid pins, pin reach, alt-click/Delete to remove.
  Exported PNG/MP4 carry it; saved alignments and undo include it.
- Video key frame: Align / hand warp record the frame they were made on (purple mark
  on the timeline, "go to it"); the same fit applies to every frame.
- Raster tool (corner + bulge handles), Refine, Reset mesh, undo/redo,
  save/load alignment JSON.
- Video: auto sync (brightness correlation → spatial check of 3 candidates →
  frame-accurate refinement), scrub/play preview, 4K H.264 MP4 export
  (people pair: 13.7 s 3840×2160 in < 30 s on this machine).
- PNG export; Exact edge mode exports real transparency.
- Phone layout (no horizontal scroll).

Limits / not done:
- **Follow camera** (handheld video) is experimental: re-detects the screen per frame
  and moves the pinned corners by the detection's drift; detection noise can jitter.
  A proper tracker (feature tracking on the TV body) would be the next step.
- Alignment takes ~10–15 s (Standard/Best), sync ~45 s for two 14 s clips.
  Possible speed-ups: fewer search seeds, WASM, or GPU evaluation of the objective.
- The synthetic residual error (~3 px mild, ~6 px heavy glitch at ~3500 px width) is
  mostly the glitch itself moving edges; more iterations don't reduce it.
- The CRT often crops the original (3:2 image on a 4:3 screen loses ~5–8% per side).
  That strip doesn't exist in the capture: Pin corners eases it in, Exact leaves it
  transparent. There is no way to "match" it.
- Not tested in Safari / Firefox.

## Important decisions / gotchas

- **Hand warp is a render-time layer on top of the auto warp** (`hand.js`, shader
  `handD`): the layer shows `autoWarp(clamp(uv + D(uv)))`, where D is a Gaussian RBF
  through the pins (exact at each pin, ~0 beyond the reach) plus zero-pins at the four
  corners. A thin-plate spline was tried first: one pin bent the whole frame. The hand
  warp never touches the solver; full Align clears it, Refine keeps it. The solver's
  own pin constraints (`model.pins`, `pinEnergy`) remain in the engine but the UI no
  longer uses them.
- **UI belongs to the JAS.BLACK tool family** (paint.jas.black = Code Painter,
  snap.jas.black = SnipSnap). Tokens are Code Painter's exactly (`app.css`, on
  `body[data-ui]`, dark default + its light theme): keys with a 2 px lip that press down,
  5 px LEDs, green `#39ff88` + purple `#b36bff` accents, green-on-black LCD status line,
  IBM Plex Mono everywhere, Archivo 800 for the badge, numbered step panels with
  right-aligned hints. SnipSnap contributes the cutting-mat dot grid (drawn in the stage
  shader, same 24 px spacing and noise), the round play key + inset timeline, the
  tracked-uppercase progress overlay and left-edged toasts. Purple = pins/markers,
  green = active state, edges overlay and screen handles. Keep new UI inside this kit.

- **NGF similarity** (squared dot of normalized gradients) is the core choice: it is
  blind to contrast inversion, false colour and emboss, and flat glitch noise carries
  no weight. Intensity-based metrics fail on synth footage.
- **Anti-alias before every downscale** (`image.js downscale`): CRT scanlines sit near
  Nyquist; a box filter folds them into fake horizontal moire bands that the matcher
  locks onto, producing a systematic vertical stretch (was a 5–15 px error).
- **The optimizer fits the band-free content warp; edge pinning is render-only**
  (`model.js renderST` / `fitMargins`). Applying the edge band during optimization makes
  the mesh fight the band and the distortion leaks into the interior. The band width
  comes from the measured overscan (≈1.8×) and is widened until the map is fold-free;
  a too-narrow band caused mirrored "kaleidoscope" strips at the frame edges.
- **Global search needs diversity**: the 40 px brute-force landscape is flat for busy
  textures (leopard spots); seeds are polished at 64 and 128 px with shear frozen, the
  best candidate per distinct zoom goes to a tournament through the first mesh levels.
- Dense optical flow as the final warp was tried and rejected: it melts the glitch
  texture. The mesh is a coarse B-spline (≤16 cells across) with bending regularization.
- `optimize.js` `tick()` yields to the UI only on the main thread; in the worker it is a
  no-op (background-tab timer throttling made solves crawl).
- The shader in `gl/renderer.js` duplicates `model.js mapPoint`; change both together.
- `web/index.html` is a full document (needed for GitHub Pages). The artifact publisher
  wraps pages in its own skeleton; the nested tags are tolerated by browsers.
- WebGL renderer uses `preserveDrawingBuffer: true` so screenshots/thumbnails capture it.
- mediabunny `CanvasSink` pools canvases: upload a frame to GL before asking for more;
  always `return()` the async frame iterator (export does, in `finally`).
- The project was briefly called "CRT Match": saved alignment files with
  `"app": "crt-match"` still load, and the theme preference falls back to the old
  `crt-match-ui` localStorage key.
- Python CLI on OpenCV 5: `cv2.AKAZE_create` is gone (SIFT/ORB used), RANSAC via
  `cv2.USAC_MAGSAC`.
