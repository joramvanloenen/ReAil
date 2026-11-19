# ReAil — reality, re-rendered

**Live demo (GitHub Pages):** https://joramvanloenen.github.io/ReAil/

Interactive proof of concept: a rotating, cuttable neutral-grey Three.js sphere is fed to an AI image-to-image generator and displayed in a deliberately low-resolution output viewport. Slicing breaks the actual triangle mesh, adds lighter-grey solid cut caps, and moves and drops the fragments. Browser ONNX mode uses 224×224 input; the remote SDXL-Turbo path captures the Three.js scene at 512×512.

## What works

- Drag across the sphere to cut its real 3D triangle geometry with a plane derived from the gesture; cuts can be repeated and affect existing fragments.
- Slice / Orbit toggle; one-click cuts; reset; rotation speed and pause; mouse zoom.
- Original source and AI output in separate responsive live viewports.
- **Genuine browser AI:** pretrained ONNX neural style transfer (Candy, Mosaic, Rain Princess, Udnie) runs in a Web Worker via ONNX Runtime Web. No API key or server. Each model downloads ~6.7 MB on first use.
- **Optional generative diffusion:** a Python CUDA SDXL-Turbo img2img backend with WebSocket support, prompt input, and frame transport. This is a different model from the browser neural-style option.
- Select 64×64, 96×96, 128×128, or 224×224 display output. Adjust requested inference rate, pixelated scaling, and compare against raw preview.
- Real scene-FPS, measured AI-result-FPS, and end-to-end inference latency indicators.
- Live remote-diffusion tuning for transformation strength, inference steps, temporal stability, motion response, seed locking/randomization, and temporal-memory reset. Changes apply on the next generated frame without restarting Uvicorn.
- Remote telemetry shows the effective strength, steps, source motion, seed and GPU compute time returned by the server.
- Clearly labeled **raw preview / NOT AI** fallback if inference cannot initialize. No pretend image generation.

## How the pipeline works

    Three.js sphere → WebGL framebuffer → 224×224 RGB
                                            |
                         +------------------+-------------------+
                         |                                      |
            Browser ONNX feed-forward AI                   Optional WebSocket
              224×224 neural style                         CUDA SDXL-Turbo img2img
                         |                                      |
                         +------------------+-------------------+
                                            |
                        Downsample / show as pixelated output

The free browser option is **actual learned neural image-to-image style transfer**, not diffusion or a photorealistic generator. It makes painterly interpretations of the scene with pretrained weights. It operates at a fixed 224×224 tensor resolution even when displayed at 64×64, so a smaller displayed resolution alone **does not** improve ONNX inference speed. Expect hardware-dependent rates: browser WASM can take hundreds of milliseconds per AI frame. The 3D renderer remains responsive independently.

For a *photorealistic* result, switch to the optional SDXL-Turbo WebSocket. ReAil sends a 512×512 source frame and defaults to 2 inference steps at strength 0.5 with guidance disabled. It costs more GPU compute than the browser ONNX path, and **30 FPS is not guaranteed**.

## How to run

Open the published Pages URL, or start any static server locally:

    python -m http.server 8000

Then open http://localhost:8000 . The CDN-hosted Three.js and ONNX dependencies and Hugging Face-hosted model files need an internet connection for initial load. A secure browser is required for some WebGPU features, but this proof of concept uses ONNX WebAssembly for compatibility. You do not need to install Python for the browser-only mode.

## Controls

- Drag a line through the sphere in Slice mode to cut along that screen-space direction.
- Click + MAKE A CUT to split it with a preset cutting plane.
- Choose Orbit mode to rotate the camera (or scroll to zoom).
- RESET reconstructs the whole unsliced sphere.
- The right side updates whenever a model has completed processing a new frame; its AI FPS will generally be lower than the left viewport FPS.
- Choose RAW PREVIEW to compare against source-downsampling without inference.

## Optional photorealistic diffusion server

Requires an NVIDIA CUDA GPU and Python 3.10+ with a suitable installed PyTorch CUDA build. A GPU with modern tensor cores is recommended. Tested performance should be measured on your own card; a GTX 1080 is unlikely to deliver 30 FPS. The initial model download can be several GB.

    pip install -r requirements.txt
    uvicorn server:app --host 127.0.0.1 --port 8765

In the UI select **REMOTE DIFFUSION (WS)**, enter ws://127.0.0.1:8765, press CONNECT. Requests are JPEG data URIs; responses contain generated JPEG data URIs with the same JSON transport:

    {"type":"frame","image":"data:image/jpeg;base64,...","prompt":"a realistic red sphere","frameId":1}

Response:

    {"type":"frame","image":"data:image/jpeg;base64,...","frameId":1,"computeMs":123.4}

For use from an HTTPS-hosted GitHub Pages site, your browser may block insecure ws:// requests, except where browser rules permit trusted localhost loopback. If blocked, run the UI locally over HTTP or expose the server securely via **wss://** and proper authentication. Do not expose the example backend openly: it has no authentication and allows expensive GPU operations.

The optional backend is a proof of concept, **not StreamDiffusion**. It uses SDXL-Turbo with a locked/repeatable diffusion seed plus adjustable output temporal blending to reduce frame-to-frame re-interpretation while keeping the current Three.js frame as the actual img2img input. The tuning controls are sent with every WebSocket frame, so strength, steps, stability, motion response and seed can be changed live. For sustained high-frame-rate results, a StreamDiffusion / TensorRT inference server would still be the next optimization step.

## Deployment

The GitHub Actions workflow in .github/workflows/deploy.yml publishes the static site. In repository Settings → Pages, ensure **Source: GitHub Actions** is selected. GitHub Pages serves the browser frontend only, not server.py.

## Models & dependencies

- Three.js (MIT), https://threejs.org/
- ONNX Runtime Web (MIT), https://onnxruntime.ai/
- ONNX Model Zoo Fast Neural Style Transfer / trained models, https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style
- Optional Stability AI SDXL-Turbo model, https://huggingface.co/stabilityai/sdxl-turbo
- Optional Diffusers, https://github.com/huggingface/diffusers

ReAil is an experimental research playground. No AI performance guarantee, no fabricated generation speed, and no hidden paid API calls.
