"""ReAil SDXL-Turbo real-time image-to-image backend.

Optimizations:
- The UI controls the *actual* GPU inference resolution: 256, 384 or 512.
- Prompt embeddings are cached instead of running both SDXL text encoders per frame.
- Optional tiny SDXL VAE (madebyollin/taesdxl) for lower-latency encode/decode.
- Single GPU inference at a time, latest-frame backpressure remains in the client.
- Measures actual pipeline and end-to-end server processing separately.
- Every frame still uses the CURRENT Three.js source, not the previous AI output.

Run: python -m uvicorn server:app --host 127.0.0.1 --port 8765
"""
import asyncio
import base64
from collections import OrderedDict
from io import BytesIO
import os
import secrets
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image, ImageChops, ImageStat

app = FastAPI(title="ReAil optimized SDXL-Turbo backend")

pipeline = None
standard_vae = None
tiny_vae = None
pipeline_lock = asyncio.Lock()
inference_lock = asyncio.Lock()
prompt_cache = OrderedDict()
PROMPT_CACHE_LIMIT = 8

MAX_FRAME_BYTES = 2_000_000
SUPPORTED_RESOLUTIONS = (256, 384, 512)
DEFAULT_RESOLUTION = 384

DEFAULT_PROMPT = (
    "Photorealistic smooth matte grey sphere, preserve exact silhouette and "
    "orientation, lighter grey solid interior where sliced, natural studio "
    "lighting, stable object identity"
)

DEFAULT_STRENGTH = float(os.getenv("REAIL_STRENGTH", "0.50"))
DEFAULT_STEPS = int(os.getenv("REAIL_STEPS", "2"))
DEFAULT_STABILITY = float(os.getenv("REAIL_STABILITY", "0.42"))
DEFAULT_MOTION_RESPONSE = float(os.getenv("REAIL_MOTION_RESPONSE", "1.0"))
DEFAULT_SEED = int(os.getenv("REAIL_SEED", "1337"))
GUIDANCE_SCALE = 0.0  # Required for SDXL-Turbo.
MOTION_SENSITIVITY = float(os.getenv("REAIL_MOTION_SENSITIVITY", "10.0"))


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def as_float(value, default):
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else default
    except (TypeError, ValueError):
        return default


def as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def normalize_settings(raw):
    raw = raw if isinstance(raw, dict) else {}
    steps = int(clamp(as_int(raw.get("steps"), DEFAULT_STEPS), 1, 4))
    requested_strength = clamp(as_float(raw.get("strength"), DEFAULT_STRENGTH), 0.05, 1.0)

    # Diffusers img2img must have at least one denoising step.
    strength = max(requested_strength, 1.0 / steps)

    requested_resolution = as_int(raw.get("resolution"), DEFAULT_RESOLUTION)
    resolution = requested_resolution if requested_resolution in SUPPORTED_RESOLUTIONS else DEFAULT_RESOLUTION

    return {
        "steps": steps,
        "effectiveSteps": max(1, int(steps * strength)),
        "strength": strength,
        "requestedStrength": requested_strength,
        "resolution": resolution,
        "fastVae": raw.get("fastVae") is True,
        "temporalStability": clamp(
            as_float(raw.get("temporalStability"), DEFAULT_STABILITY), 0.0, 0.90
        ),
        "motionResponse": clamp(
            as_float(raw.get("motionResponse"), DEFAULT_MOTION_RESPONSE), 0.0, 2.0
        ),
        "seed": int(clamp(as_int(raw.get("seed"), DEFAULT_SEED), 0, 2_147_483_647)),
        "lockSeed": raw.get("lockSeed", True) is not False,
    }


async def get_pipeline():
    global pipeline, standard_vae

    if pipeline is not None:
        return pipeline

    async with pipeline_lock:
        if pipeline is not None:
            return pipeline

        import torch
        from diffusers import AutoPipelineForImage2Image

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for ReAil.")

        model = os.getenv("REAIL_MODEL", "stabilityai/sdxl-turbo")
        print(f"Loading {model} on {torch.cuda.get_device_name(0)}")

        loaded = await asyncio.to_thread(
            AutoPipelineForImage2Image.from_pretrained,
            model,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        loaded = loaded.to("cuda")
        loaded.set_progress_bar_config(disable=True)

        try:
            loaded.unet.to(memory_format=torch.channels_last)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        standard_vae = loaded.vae
        pipeline = loaded
        print("SDXL-Turbo ready: variable resolution, cached prompt embeddings, optional Tiny VAE")
        return pipeline


def choose_vae(pipe, use_tiny):
    global tiny_vae

    if not use_tiny:
        pipe.vae = standard_vae
        return "standard"

    if tiny_vae is None:
        import torch
        from diffusers import AutoencoderTiny

        print("Loading optional madebyollin/taesdxl (first use only)")
        tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=torch.float16
        ).to("cuda")
        tiny_vae.eval()

    pipe.vae = tiny_vae
    return "tiny"


def get_prompt_embeddings(pipe, prompt):
    import torch

    cached = prompt_cache.get(prompt)
    if cached is not None:
        prompt_cache.move_to_end(prompt)
        return cached, True

    with torch.inference_mode():
        embeddings = pipe.encode_prompt(
            prompt=prompt,
            device=torch.device("cuda"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    # SDXL returns: prompt, negative, pooled prompt, negative pooled.
    cached = (embeddings[0], embeddings[2])
    prompt_cache[prompt] = cached

    while len(prompt_cache) > PROMPT_CACHE_LIMIT:
        prompt_cache.popitem(last=False)

    return cached, False


def decode_image(data_uri, resolution):
    if not isinstance(data_uri, str) or not data_uri.startswith("data:image/jpeg;base64,"):
        raise ValueError("Expected JPEG base64 data URI")

    encoded = data_uri.split(",", 1)[1]
    if len(encoded) > MAX_FRAME_BYTES * 4 // 3 + 128:
        raise ValueError("Input too large")

    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("Input too large")

    with Image.open(BytesIO(raw)) as image:
        rgb = image.convert("RGB")

    if rgb.size != (resolution, resolution):
        rgb = rgb.resize((resolution, resolution), Image.Resampling.BILINEAR)

    return rgb


def encode_image(image):
    output = BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=False)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def measure_scene_change(previous, current):
    if previous is None:
        return 1.0
    a = previous.resize((64, 64), Image.Resampling.BILINEAR)
    b = current.resize((64, 64), Image.Resampling.BILINEAR)
    stats = ImageStat.Stat(ImageChops.difference(a, b))
    difference = sum(stats.mean) / len(stats.mean) / 255.0
    return clamp(difference * MOTION_SENSITIVITY, 0.0, 1.0)


def temporal_filter(previous, current, scene_change, settings):
    if previous is None or settings["temporalStability"] <= 0:
        return current, 1.0

    previous_amount = settings["temporalStability"] * (
        1.0 - clamp(scene_change * settings["motionResponse"], 0.0, 1.0)
    )
    previous_amount = clamp(previous_amount, 0.0, 0.90)
    if previous.size != current.size:
        previous = previous.resize(current.size, Image.Resampling.BILINEAR)

    new_amount = 1.0 - previous_amount
    return Image.blend(previous, current, new_amount), new_amount


def run_diffusion(pipe, source, prompt, settings, frame_number):
    import torch

    # This entire call is serialized by inference_lock.
    vae_name = choose_vae(pipe, settings["fastVae"])
    embeddings, prompt_cache_hit = get_prompt_embeddings(pipe, prompt)

    seed = settings["seed"]
    if not settings["lockSeed"]:
        seed = (seed + secrets.randbelow(2_000_000_000) + frame_number) % 2_147_483_647
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # GPU kernels are async. Synchronize to report meaningful wall latency,
    # not just how fast Python queued CUDA work.
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt_embeds=embeddings[0],
            pooled_prompt_embeds=embeddings[1],
            image=source,
            strength=settings["strength"],
            num_inference_steps=settings["steps"],
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            original_size=(settings["resolution"], settings["resolution"]),
            target_size=(settings["resolution"], settings["resolution"]),
        ).images[0].convert("RGB")

    torch.cuda.synchronize()
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    return result, elapsed, seed, prompt_cache_hit, vae_name


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.getenv("REAIL_MODEL", "stabilityai/sdxl-turbo"),
        "supportedResolutions": list(SUPPORTED_RESOLUTIONS),
        "defaultResolution": DEFAULT_RESOLUTION,
        "promptCache": True,
        "tinyVae": True,
        "liveTuning": True,
    }


async def handle_live_socket(socket: WebSocket):
    print(f"ReAil connection: {socket.client} origin={socket.headers.get('origin', 'unknown')}")
    await socket.accept()

    previous_source = None
    previous_result = None
    previous_resolution = None
    frame_number = 0

    try:
        while True:
            request = await socket.receive_json()
            message_type = request.get("type")

            if message_type == "reset":
                previous_source = None
                previous_result = None
                previous_resolution = None
                frame_number = 0
                await socket.send_json({"type": "reset"})
                continue

            if message_type != "frame":
                continue

            request_started = time.perf_counter()

            try:
                settings = normalize_settings(request.get("settings"))
                resolution = settings["resolution"]

                # Keep temporal history only while it matches the new resolution.
                if previous_resolution != resolution:
                    previous_source = None
                    previous_result = None
                    previous_resolution = resolution

                pipe = await get_pipeline()
                source = await asyncio.to_thread(decode_image, request.get("image", ""), resolution)
                prompt = str(request.get("prompt") or DEFAULT_PROMPT)[:500]
                scene_change = measure_scene_change(previous_source, source)
                frame_number += 1

                async with inference_lock:
                    generated, compute_ms, frame_seed, cache_hit, vae_name = (
                        await asyncio.to_thread(
                            run_diffusion, pipe, source, prompt, settings, frame_number
                        )
                    )

                stabilized, new_amount = temporal_filter(
                    previous_result, generated, scene_change, settings
                )
                previous_source = source
                previous_result = stabilized

                image_uri = await asyncio.to_thread(encode_image, stabilized)
                total_ms = round((time.perf_counter() - request_started) * 1000, 1)

                await socket.send_json({
                    "type": "frame",
                    "frameId": request.get("frameId"),
                    "image": image_uri,
                    "computeMs": compute_ms,
                    "serverMs": total_ms,
                    "settings": {
                        "resolution": resolution,
                        "requestedStrength": round(settings["requestedStrength"], 4),
                        "strength": round(settings["strength"], 4),
                        "steps": settings["steps"],
                        "effectiveSteps": settings["effectiveSteps"],
                        "temporalStability": settings["temporalStability"],
                        "motionResponse": settings["motionResponse"],
                        "seed": settings["seed"],
                        "frameSeed": frame_seed,
                        "lockSeed": settings["lockSeed"],
                        "fastVae": settings["fastVae"],
                        "vae": vae_name,
                        "promptCacheHit": cache_hit,
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


@app.websocket("/ws")
async def live_ws(socket: WebSocket):
    await handle_live_socket(socket)


@app.websocket("/")
async def live_root(socket: WebSocket):
    await handle_live_socket(socket)
