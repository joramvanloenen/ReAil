"""Benchmark actual SDXL inference work at 256/384/512.

Uses the same model and run_diffusion() function as the WebSocket server.
Timing excludes initial model download, first-run warmup and browser transport.

Run:
    python benchmark.py
    python benchmark.py --vae both --samples 7
"""
import argparse
import asyncio
import statistics

from PIL import Image, ImageDraw

from server import (
    DEFAULT_PROMPT,
    get_pipeline,
    normalize_settings,
    run_diffusion,
)


def make_source(size):
    image = Image.new("RGB", (size, size), "#111111")
    draw = ImageDraw.Draw(image)
    pad = int(size * 0.18)
    draw.ellipse(
        (pad, pad, size - pad, size - pad),
        fill="#969696",
        outline="#b2b2b2",
        width=max(1, size // 80),
    )
    return image


async def main():
    parser = argparse.ArgumentParser(description="ReAil SDXL latency benchmark")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--vae", choices=("standard", "tiny", "both"), default="standard")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.5)
    args = parser.parse_args()

    pipe = await get_pipeline()
    modes = (False, True) if args.vae == "both" else (args.vae == "tiny",)
    samples = max(2, args.samples)
    warmup = max(1, args.warmup)

    print()
    print("ReAil local inference benchmark (warmup excluded)")
    print("Resolution | VAE      | Steps | Median GPU pipeline | Range")
    print("-----------|----------|-------|---------------------|----------------")

    for fast in modes:
        for resolution in (256, 384, 512):
            settings = normalize_settings({
                "resolution": resolution,
                "steps": args.steps,
                "strength": args.strength,
                "fastVae": fast,
                "seed": 1337,
                "lockSeed": True,
            })
            source = make_source(resolution)
            latencies = []

            for i in range(warmup + samples):
                _, ms, _, _, _ = run_diffusion(
                    pipe, source, DEFAULT_PROMPT, settings, i
                )
                if i >= warmup:
                    latencies.append(ms)

            median = statistics.median(latencies)
            print(
                f"{resolution:>4}²      | "
                f"{'Tiny' if fast else 'Standard':<8} | "
                f"{settings['effectiveSteps']}/{settings['steps']:<5} | "
                f"{median:>9.1f} ms          | "
                f"{min(latencies):.1f}–{max(latencies):.1f} ms"
            )

    print()
    print("These are local pipeline timings, not network/browser frame latency.")
    print("At 256/384, SDXL-Turbo may lose detail versus its native 512 resolution.")


if __name__ == "__main__":
    asyncio.run(main())
