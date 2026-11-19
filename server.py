"""Optional real generative img2img backend for ReAil.

This service is NOT hosted on GitHub Pages. It needs a CUDA GPU and your own
machine/server. It consumes source JPEG frames from the browser over WebSocket
and produces SD-Turbo image-to-image frames. No promise of 30 FPS.
Run: uvicorn server:app --host 127.0.0.1 --port 8765
"""
import asyncio
import base64
from io import BytesIO
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="ReAil optional diffusion backend")
pipeline = None
pipeline_lock = asyncio.Lock()
MAX_FRAME_BYTES = 2_000_000
DEFAULT_PROMPT = (
    "A photorealistic red apple, cut open, ivory flesh, "
    "cinematic studio lighting, fine natural texture"
)


async def get_pipeline():
    global pipeline
    if pipeline is None:
        async with pipeline_lock:
            if pipeline is None:
                import torch
                from diffusers import AutoPipelineForImage2Image

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA GPU required for this optional SD-Turbo backend. "
                        "Use the browser ONNX engine if you have no CUDA device."
                    )
                model_name = os.getenv("REAIL_MODEL", "stabilityai/sd-turbo")
                pipeline = await asyncio.to_thread(
                    AutoPipelineForImage2Image.from_pretrained,
                    model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                pipeline = pipeline.to("cuda")
                pipeline.set_progress_bar_config(disable=True)
    return pipeline


def run_diffusion(pipe, data_uri: str, prompt: str):
    import torch
    from PIL import Image

    if not data_uri.startswith("data:image/jpeg;base64,"):
        raise ValueError("Expected a base64 JPEG data URI")
    encoded = data_uri.split(",", 1)[1]
    if len(encoded) > (MAX_FRAME_BYTES * 4 // 3 + 128):
        raise ValueError("Input frame too large")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("Input frame too large")
    source = Image.open(BytesIO(raw)).convert("RGB").resize((256, 256))
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt[:400],
            image=source,
            strength=0.6,
            num_inference_steps=2,
            guidance_scale=0.0,
        ).images[0]
    ms = (time.perf_counter() - start) * 1000
    output = BytesIO()
    result.save(output, format="JPEG", quality=80)
    uri = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return uri, round(ms, 1)


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("REAIL_MODEL", "stabilityai/sd-turbo")}


@app.websocket("/")
async def live(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            request = await socket.receive_json()
            if request.get("type") != "frame":
                continue
            try:
                pipe = await get_pipeline()
                prompt = request.get("prompt", DEFAULT_PROMPT) or DEFAULT_PROMPT
                image, ms = await asyncio.to_thread(
                    run_diffusion, pipe, request.get("image", ""), str(prompt)
                )
                await socket.send_json({
                    "type": "frame", "frameId": request.get("frameId"),
                    "image": image, "computeMs": ms
                })
            except Exception as exc:
                await socket.send_json({"type": "error", "message": str(exc)[:500]})
    except WebSocketDisconnect:
        return
