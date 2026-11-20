# ReAil

**Live demo:** https://joramvanloenen.github.io/ReAil/

ReAil is a realtime img2img experiment: a sliceable grey Three.js sphere drives a local SDXL-Turbo server. The square source and output viewports always keep the same aspect ratio.

## Selectable Three.js sources

Use **SOURCE GEOMETRY** to swap the scene driving SDXL-Turbo:

| Source | Behavior |
|---|---|
| Grey sphere | Original solid, continuously rotating and sliceable sphere |
| Rough human head | Procedurally sculpted grey clay head with nose, brows, sockets, lips, chin, ears and a closed triangulated surface that can be sliced |
| Editable metaball blob | Connected implicit iso-surface generated from movable field lobes using marching tetrahedra |
| Zero-G particles | Individual 3D grey spheres, emitted slowly, floating without gravity, attracted to each other with short-range separation and soft cloud containment |

The **rough head and metaball blob** use the existing mesh-slicing interaction. To sculpt the blob, select **EDIT**, drag the white handles in the Three.js viewport, adjust the selected lobe's radius and use **+ LOBE / − LOBE** to change the shape. Select **ORBIT** to change the viewing angle before moving lobes in depth. During dragging, the blob is extracted at reduced resolution for responsiveness; releasing the handle rebuilds a smoother mesh.

The particle source defaults to **ORBIT**. Set **EMIT / SEC**, **ATTRACTION**, and **SEPARATION** to control its physical behavior. The emitter starts with a small visible cluster and adds particles gradually, to a bounded count. There is no downward gravity and the floor is hidden.

Each source updates the suggested SDXL prompt unless you've entered your own prompt. The existing AI texture feedback applies to the head, metaball and individual particles. Blob edit handles are hidden from captured AI input frames, and stale AI frames from the previous shape or an earlier mesh edit are ignored.

## Performance improvements

**AI resolution now means actual inference resolution**, not merely how large the generated image is displayed. Both the browser's JPEG capture and the Python diffusion input switch together:

| Resolution | Pixels | Notes |
|---|---:|---|
| 256×256 | 65,536 | Fastest; SDXL-Turbo detail/identity may suffer |
| 384×384 | 147,456 | Default speed/quality compromise |
| 512×512 | 262,144 | Native SDXL-Turbo resolution; highest quality of these options |

The site also includes:

- **Prompt embedding cache:** SDXL's two text encoders run only when a prompt first appears (up to eight cached prompts).
- **Fast VAE:** optional Tiny SDXL Autoencoder (`madebyollin/taesdxl`) for faster encode/decode, with some quality loss. First enable may download additional weights.
- **Skip unchanged frames:** compares a low-resolution preview of the Three.js scene and avoids running diffusion on nearly identical frames. A forced refresh occurs every ~2.5 seconds. This reduces total GPU work when the scene hardly changes; it does **not** make an individual generated frame faster.
- **Backpressure:** the browser keeps only one request in flight, sending the most recent current scene after each result rather than queuing obsolete frames.
- **Accurate timing:** server timing synchronizes CUDA and separates model compute latency from full server processing; the browser separately reports end-to-end latency.
- **Effective steps:** UI displays actual denoising steps / requested scheduler steps. For example, 2 configured steps at strength 0.5 usually means only one denoising step.

Tuning controls for strength, steps (1–4), temporal stability, motion response, seed, and FPS are retained. The current Three.js frame remains the image-to-image input on **every** generation, and locked noise can reduce flickering.

## AI texture feedback

The generated SDXL-Turbo image can now be fed back onto the Three.js sphere and its sliced fragments.

- **FEEDBACK ON:** enable/disable the visual loop (enabled by default).
- **AI TEXTURE FEEDBACK:** adjust how strongly the AI image replaces the grey shaded surface (default 70%). Zero leaves the source grey and breaks the loop.
- ReAil records the source camera and each fragment's transform **when the image is sent**, then creates screen-projected UV coordinates when that response returns. This keeps the result attached to the actual geometry as it moves rather than sliding across the screen.
- Projection is restricted to front-facing and in-frame vertices. New fragments only receive an AI projection after a subsequent inference result.
- An AI result is applied to the 3D mesh *before* the next inference request is allowed, so the next source frame includes the previous result.
- **RESET AI MEMORY** also clears the projected AI texture influence, giving the loop a fresh start.
- Feedback can increase visual drift and reinforcement. Reduce its strength or turn it off if the results spiral.

The feedback shader operates only on the sphere/fragments, not on the floor or background. It uses the same output canvas as the AI preview, without a second image download or server-side changes. The existing 256/384/512 GPU inference resolution control still works.

## Geometry-aware closed-loop feedback

ReAil now has an adaptive **closed-loop feedback controller**, inspired by the
geometry-guided generation and recursive feedback research discussed in the chat.
It adds no second AI model and does not change the CUDA server.

**Feedback signal:**

1. The current Three.js frame is captured and sent to SDXL-Turbo.
2. A separate tiny **96×96 geometry depth/coverage pass** records which fragments
   are visible at the exact source camera pose. Floor, ring and sculpt handles
   are excluded.
3. The returned AI image is compared with that silhouette. A lightweight sensor
   estimates how much generated material spills outside the shape or disappears
   from within it. **This is a coarse RGB silhouette signal, not a semantic model.**
4. A bounded, smoothed PI(D)-style controller adjusts how much AI texture returns
   to the Three.js model. An increase in measured drift reduces the gain.
5. Per-vertex projection confidence combines front-facing normals, depth
   consistency, silhouette proximity and motion. Hidden or uncertain surfaces
   reject most of the previous image.
6. Appearance memory blends a fraction of the captured current scene (which
   already contains geometry-reprojected AI details) into the new feedback
   texture, instead of blending two unaligned previous screen images.
7. A seeded, **object-space variation pattern** enters the outgoing Three.js
   render and consequently the next SDXL input. The pattern moves with each mesh,
   including individual particles and sliced pieces. This is **not latent noise
   warping** inside the diffusion sampler.

The generated AI output panel remains the unmodified AI image. Only the
geometry's **input-feedback texture** is blended and corrected.

### Controls

| Control | Effect |
|---|---|
| Behavior | Presets: Stable / Balanced / Mutate; changing sliders selects Custom |
| Surface adhesion | Keeps generated details attached when the 3D surface moves |
| Mutation | Introduces object-stable variation and permits more visual drift |
| Geometry lock | Rejects uncertain silhouettes and occluded regions more strongly |
| Appearance memory | Preserves previous reprojected appearance in the next texture |
| Auto correction | Uses measured silhouette drift to adjust feedback gain |
| AI Texture Feedback | Existing overall feedback amount and ON/OFF toggle |

Live measurements show **DRIFT** (estimated silhouette deviation),
**GAIN** (actual adaptive gain) and **VISIBLE** (source geometry coverage in the
depth sensor). These are diagnostics, not guarantees of identity accuracy.

If the browser cannot read back the depth render target, ReAil falls back to the
existing front-facing projection. Turning feedback off also skips the extra depth
readback. No change to the Python server or additional CUDA inference is required.

### Regression tests

Requires Node.js:

```powershell
node test-feedback.mjs
```

This tests depth visibility, silhouette error detection and gain reduction.
The actual shader, WebGL depth pass and live GPU pipeline still require a browser
and GPU for end-to-end verification.

## Run locally

Start the frontend from the actual ReAil folder:

```powershell
cd C:\Users\Joram\ReAil
python -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000.

In a second terminal:

```powershell
cd C:\Users\Joram\ReAil
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8765
```

The browser auto-connects to `ws://127.0.0.1:8765/ws`. Health: http://127.0.0.1:8765/health.

## Local GPU benchmark

Use the supplied script to compare actual single-frame inference latency on your GPU, excluding warm-up:

```powershell
python benchmark.py
python benchmark.py --vae both --samples 7
```

It benchmarks 256, 384, and 512 pixels with the same SDXL-Turbo pipeline and reports median processing time. Running the Tiny VAE variant may download its weights the first time. Results depend on your GPU, installed PyTorch/Diffusers versions, background GPU applications, and first-run warm-up.

## Deployment

GitHub Pages hosts only the static frontend. `server.py` requires a local or separately hosted CUDA GPU and is not deployed to Pages. This backend uses ordinary Diffusers; TensorRT, compilation and StreamDiffusion are **not** enabled by default. No specific FPS is guaranteed.

Dependencies: Three.js, SDXL-Turbo, PyTorch CUDA, Diffusers, FastAPI, and optional TAESDXL.
