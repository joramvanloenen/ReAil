"""ReAil realtime SD-Turbo backend with live browser tuning.

The current Three.js frame is always the diffusion input. Temporal coherence is
created with a repeatable noise seed plus optional output blending. The browser
can change strength, step count, temporal stability, motion response and seed on
every frame without restarting Uvicorn.

Run:
    python -m uvicorn server:app --host 127.0.0.1 --port 8765
"""

import asyncio
import base64
from io import BytesIO
import os
import secrets
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image, ImageChops, ImageStat

app = FastAPI(title="ReAil realtime diffusion backend")

pipeline = None
pipeline_lock = asyncio.Lock()
inference_lock = asyncio.Lock()

MAX_FRAME_BYTES = 2_000_000
MODEL_SIZE = (256, 256)

DEFAULT_PROMPT = (
    "photorealistic red apple, preserve the exact silhouette, orientation, camera "
    "position and composition of the input, realistic apple skin, ivory flesh where "
    "cut, natural studio lighting, same apple identity between consecutive frames"
)

DEFAULT_STRENGTH = float(os.getenv("REAIL_STRENGTH", "0.34"))
DEFAULT_STEPS = int(os.getenv("REAIL_STEPS", "4"))
DEFAULT_STABILITY = float(os.getenv("REAIL_STABILITY", "0.35"))
DEFAULT_MOTION_RESPONSE = float(os.getenv("REAIL_MOTION_RESPONSE", "1.0"))
DEFAULT_SEED = int(os.getenv("REAIL_SEED", "1337"))
GUIDANCE_SCALE = float(os.getenv("REAIL_GUIDANCE", "0.0"))
MOTION_SENSITIVITY = float(os.getenv("REAIL_MOTION_SENSITIVITY", "10.0"))


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_settings(raw):
    raw = raw if isinstance(raw, dict) else {}

    steps = int(clamp(as_int(raw.get("steps"), DEFAULT_STEPS), 1, 8))
    requested_strength = clamp(
        as_float(raw.get("strength"), DEFAULT_STRENGTH),
        0.05,
        1.0,
    )

    # Diffusers img2img must get at least one actual denoising timestep.
    minimum_strength = min(1.0, (1.0 / steps) + 0.001)
    strength = clamp(requested_strength, minimum_strength, 1.0)

    stability = clamp(
        as_float(raw.get("temporalStability"), DEFAULT_STABILITY),
        0.0,
        0.90,
    )
    motion_response = clamp(
        as_float(raw.get("motionResponse"), DEFAULT_MOTION_RESPONSE),
        0.0,
        2.0,
    )
    seed = int(clamp(as_int(raw.get("seed"), DEFAULT_SEED), 0, 2_147_483_647))
    lock_seed = bool(raw.get("lockSeed", True))

    return {
        "steps": steps,
        "requestedStrength": requested_strength,
        "strength": strength,
        "temporalStability": stability,
        "motionResponse": motion_response,
        "seed": seed,
        "lockSeed": lock_seed,
    }


async def get_pipeline():
    global pipeline

    if pipeline is not None:
        return pipeline

    async with pipeline_lock:
        if pipeline is not None:
            return pipeline

        import torch
        from diffusers import AutoPipelineForImage2Image

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the ReAil diffusion backend.")

        model_name = os.getenv("REAIL_MODEL", "stabilityai/sd-turbo")

        print()
        print("=" * 54)
        print("ReAil realtime diffusion")
        print("GPU:  ", torch.cuda.get_device_name(0))
        print("Model:", model_name)
        print("=" * 54)

        pipeline = await asyncio.to_thread(
            AutoPipelineForImage2Image.from_pretrained,
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        pipeline = pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=True)

        try:
            pipeline.safety_checker = None
        except Exception:
            pass

        try:
            pipeline.unet.to(memory_format=torch.channels_last)
        except Exception:
            pass

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        print("Model ready.")
        print()
        return pipeline


def decode_data_uri(data_uri):
    if not isinstance(data_uri, str) or not data_uri.startswith("data:image/jpeg;base64,"):
        raise ValueError("Expected a base64 JPEG data URI")

    encoded = data_uri.split(",", 1)[1]
    if len(encoded) > (MAX_FRAME_BYTES * 4 // 3 + 128):
        raise ValueError("Input frame too large")

    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("Input frame too large")

    return (
        Image.open(BytesIO(raw))
        .convert("RGB")
        .resize(MODEL_SIZE, Image.Resampling.LANCZOS)
    )


def encode_image(image):
    output = BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=False)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def measure_scene_change(previous_source, current_source):
    if previous_source is None:
        return 1.0

    previous_small = previous_source.resize((64, 64), Image.Resampling.BILINEAR)
    current_small = current_source.resize((64, 64), Image.Resampling.BILINEAR)
    difference = ImageChops.difference(previous_small, current_small)
    stats = ImageStat.Stat(difference)
    mean_difference = (sum(stats.mean) / len(stats.mean)) / 255.0

    return clamp(mean_difference * MOTION_SENSITIVITY, 0.0, 1.0)


def choose_seed(settings, frame_number):
    if settings["lockSeed"]:
        return settings["seed"]

    # Different, high-quality noise every frame when explicitly unlocked.
    return (settings["seed"] + secrets.randbelow(2_000_000_000) + frame_number) % 2_147_483_647


def run_diffusion(pipe, source, prompt, settings, frame_number):
    import torch

    frame_seed = choose_seed(settings, frame_number)
    generator = torch.Generator(device="cuda").manual_seed(frame_seed)

    start = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt=str(prompt)[:500],
            image=source,
            strength=settings["strength"],
            num_inference_steps=settings["steps"],
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0].convert("RGB")

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, round(elapsed_ms, 1), frame_seed


def temporal_filter(previous_result, current_result, scene_change, settings):
    if previous_result is None or settings["temporalStability"] <= 0.0:
        return current_result, 1.0

    # Stability is the maximum fraction of the PREVIOUS result retained.
    # Motion response rapidly removes that history when the Three.js scene changes.
    responsive_motion = clamp(scene_change * settings["motionResponse"], 0.0, 1.0)
    previous_amount = settings["temporalStability"] * (1.0 - responsive_motion)
    previous_amount = clamp(previous_amount, 0.0, 0.90)
    new_amount = 1.0 - previous_amount

    previous_result = previous_result.resize(MODEL_SIZE, Image.Resampling.LANCZOS)
    filtered = Image.blend(previous_result, current_result, new_amount)
    return filtered, new_amount


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.getenv("REAIL_MODEL", "stabilityai/sd-turbo"),
        "resolution": MODEL_SIZE,
        "defaults": {
            "strength": DEFAULT_STRENGTH,
            "steps": DEFAULT_STEPS,
            "temporalStability": DEFAULT_STABILITY,
            "motionResponse": DEFAULT_MOTION_RESPONSE,
            "seed": DEFAULT_SEED,
            "lockSeed": True,
        },
        "liveTuning": True,
    }


@app.websocket("/")
async def live(socket: WebSocket):
    await socket.accept()
    print("ReAil client connected.")

    previous_source = None
    previous_result = None
    frame_number = 0

    try:
        while True:
            request = await socket.receive_json()
            message_type = request.get("type")

            if message_type == "reset":
                previous_source = None
                previous_result = None
                frame_number = 0
                await socket.send_json({"type": "reset"})
                continue

            if message_type != "frame":
                continue

            try:
                frame_number += 1
                pipe = await get_pipeline()
                source = decode_data_uri(request.get("image", ""))
                prompt = request.get("prompt", DEFAULT_PROMPT) or DEFAULT_PROMPT
                settings = normalize_settings(request.get("settings"))

                scene_change = measure_scene_change(previous_source, source)

                async with inference_lock:
                    generated, compute_ms, frame_seed = await asyncio.to_thread(
                        run_diffusion,
                        pipe,
                        source,
                        prompt,
                        settings,
                        frame_number,
                    )

                stabilized, new_amount = temporal_filter(
                    previous_result,
                    generated,
                    scene_change,
                    settings,
                )

                previous_source = source.copy()
                previous_result = stabilized.copy()

                await socket.send_json({
                    "type": "frame",
                    "frameId": request.get("frameId"),
                    "image": encode_image(stabilized),
                    "computeMs": compute_ms,
                    "settings": {
                        "requestedStrength": round(settings["requestedStrength"], 4),
                        "strength": round(settings["strength"], 4),
                        "steps": settings["steps"],
                        "temporalStability": round(settings["temporalStability"], 4),
                        "motionResponse": round(settings["motionResponse"], 4),
                        "seed": settings["seed"],
                        "frameSeed": frame_seed,
                        "lockSeed": settings["lockSeed"],
                        "sceneChange": round(scene_change, 4),
                        "newFrameAmount": round(new_amount, 4),
                    },
                })

            except Exception as exc:
                print("Diffusion error:", repr(exc))
                await socket.send_json({
                    "type": "error",
                    "message": str(exc)[:500],
                })

    except WebSocketDisconnect:
        print("ReAil client disconnected.")
