# ReAil

**Live demo:** https://joramvanloenen.github.io/ReAil/

ReAil is a focused realtime image-to-image experiment. A neutral grey Three.js sphere can be rotated and sliced, then each 512×512 source frame is sent over WebSocket to a local **SDXL-Turbo** GPU backend.

## Current pipeline

```
Three.js sphere
      ↓
512 × 512 source frame
      ↓
WebSocket
      ↓
SDXL-Turbo img2img
      ↓
512 × 512 AI frame
```

The source and AI viewports use the same square aspect ratio.

## Controls

The interface exposes only the controls used by the remote diffusion experiment:

- Slice / Orbit / Reset
- WebSocket server address
- Prompt
- Maximum inference rate
- Sphere rotation
- AI transformation strength
- Inference steps
- Temporal stability
- Motion response
- Seed locking / randomization
- Temporal-memory reset

The default backend tuning is 2 inference steps, strength 0.5, guidance scale 0, fixed seed 1337, and 42% temporal stability.

## Run locally

Frontend:

```powershell
cd C:\Users\Joram\ReAil
python -m http.server 8000 --bind 127.0.0.1
```

Open:

```
http://127.0.0.1:8000
```

GPU backend:

```powershell
cd C:\Users\Joram\ReAil
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8765
```

The frontend connects to:

```
ws://127.0.0.1:8765/ws
```

Health check:

```
http://127.0.0.1:8765/health
```

The first SDXL-Turbo request may trigger a model download.

## Backend

The backend uses:

- `stabilityai/sdxl-turbo`
- Diffusers
- PyTorch CUDA
- FastAPI / WebSocket
- FP16 inference
- 512×512 input/output
- fixed-seed temporal noise
- adjustable output temporal blending

This is still a proof of concept rather than StreamDiffusion/TensorRT. The actual AI frame rate depends on GPU and tuning.

## Deployment

GitHub Pages hosts only the static frontend. `server.py` must run separately on a CUDA-capable machine.
