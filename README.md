# ReAil

**Live demo:** https://joramvanloenen.github.io/ReAil/

ReAil is a realtime img2img experiment: a sliceable grey Three.js sphere drives a local SDXL-Turbo server. The square source and output viewports always keep the same aspect ratio.

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
