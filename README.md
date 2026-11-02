# CRT Align

Align a filmed CRT capture (photo or video) back onto the original image or video it
was made from, so the glitched capture drops on top of the source as a
pixel-registered glitch layer.

Two tools live here:

- **`web/` — the browser app (recommended).** Everything runs locally in the browser:
  your files never leave the machine. GPU warp, WebCodecs video, live overlay views,
  hand-placed pins, PNG / MP4 export at the original's exact resolution.
- **`crt_align.py` — the original Python CLI.** Older algorithm, kept for batch use.

## Run the browser app

Any static file server works; it needs to be served over http (not opened as a file).

```bash
npx -y serve web -l 5140
```

Then open http://localhost:5140. It opens with a sample pair already aligned.

To put it online, publish the `web/` folder as-is: GitHub Pages (Settings → Pages →
deploy from branch, folder `/web`), Netlify, Vercel or any static host. No build step.

Browser support: Chrome or Edge recommended (WebGL2 + WebCodecs). Safari 17+ and
Firefox 130+ work for images; video export depends on their encoders.

### Using it

1. Drop the **original** and the **filmed CRT** shot on the two slots (image or video each).
2. For two videos, press **Auto sync** (Video panel). It finds the time offset, even
   across different frame rates.
3. Press **Align**. Check with the view modes: *Edges* draws the original's outlines
   over the glitch layer; *Checker* and *Swipe* compare directly.
4. Fix anything off by hand, live, like Photoshop's Puppet Warp: with the **Pin** tool,
   click the glitch layer to drop a pin and drag it; the layer warps under the cursor
   while the other pins hold their place. Alt-click (or Delete) removes a pin.
   **Mesh grid** lays a 4×3 grid of pins over the frame for a Warp-style grid, and
   **pin reach** sets how far each pin's pull spreads. The **Screen** tool drags the
   screen corners and edge bulges on the filmed photo.
   For video, align and warp on one frame: that frame becomes the **key frame**
   (purple mark on the timeline) and its fit holds for every frame of the clip.
5. **Export**: PNG for stills, MP4 for video, always at the original's pixel size.
   *Save alignment* stores the fit as JSON to reuse on other shots from the same setup.

Frame edges: **Pin corners** fills the frame edge to edge. If the CRT cropped the
original (common: a 3:2 image shown on a 4:3 screen loses its sides), that missing
strip is eased in near the edges. **Exact** matches everywhere and leaves the strip
the screen never showed transparent (black in video).

## How the alignment works

1. **Screen detection** — the lit raster is segmented, each edge fitted with a parabola
   (curved CRT glass), corners taken where the curves meet.
2. **Global search** — zoom / offset of the content inside the screen, from a
   brute-force search plus a grid of seeds, polished at two resolutions; the best few
   candidates compete through the first mesh levels.
3. **Mesh fit** — a cubic B-spline mesh (the automatic version of a Photoshop mesh warp),
   optimized coarse to fine with L-BFGS against an edge-orientation similarity (NGF).
   It compares edge *directions* only, so false colour, inversion, emboss and brightness
   changes from the synth don't mislead it.
4. **Pinning** — the frame edges are pinned only at render time, with an edge band sized
   from the measured overscan so the warp never folds.

## Accuracy

`bench/` builds synthetic CRT captures with known ground truth (overscan zoom, raster
tilt, barrel glass, camera perspective, false colour, emboss, line wobble, scanlines,
noise) from the five test originals, in a mild and a heavy glitch variant.

Error in original-image pixels, central 60% of the frame (originals 1200–4272 px wide):

| | median | median p90 | worst p90 | worst max | failures |
|---|---|---|---|---|---|
| Python CLI (`pin_align`) | 5.5 px | 13.2 px | 38.7 px | 103 px | 1 of 10 (OpenCV exception) |
| Browser app (`fine`) | 5.1 px | 7.8 px | 15.0 px | 18 px | 0 of 10 |

Typical accuracy is similar; the difference is in the bad spots, where the old
landmark warp could be off by 100 px and the new mesh stays within ~18 px. On the real
test pairs the browser engine also locks the heavily glitched viking shot
automatically, which the Python version could not.

```bash
python bench/make_synth.py OUT && python bench/prep_raw.py OUT
node bench/run_new.mjs OUT fine     # browser engine
python bench/run_old.py OUT         # Python CLI
```

## Python CLI

```bash
pip install -r requirements.txt
python crt_align.py ORIGINAL FILMED [-o OUTPUT] [--overlay] [--points anchors.csv]
```

See `python crt_align.py --help` for all flags (`--mode pin|exact`, `--inset`,
`--stiffness`, `--per-frame N`).
