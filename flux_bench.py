#!/usr/bin/env python3
"""
FLUX.1 [schnell] text-to-image benchmark.

Measures image generation performance through the Stoney Cloud API:
  - seconds per image (latency)
  - images per minute (throughput)
  - p99 latency (worst case)

Usage:
  python flux_bench.py \
      --model black-forest-labs/FLUX.1-schnell \
      --num-images 20 \
      --size 1024x1024 \
      --steps 4

  # or set the key as an environment variable:
  export STONEY_KEY=sk-your-key-here
  python flux_bench.py --model black-forest-labs/FLUX.1-schnell
"""

import argparse
import asyncio
import os
import statistics
import time

import aiohttp


async def one_request(session, url, headers, payload, idx, results):
    """Send a single image-generation request and record its latency."""
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            await resp.read()
            elapsed = time.perf_counter() - start
            ok = resp.status == 200
            results.append({"idx": idx, "latency": elapsed, "ok": ok,
                            "status": resp.status})
            tag = "ok" if ok else f"HTTP {resp.status}"
            print(f"  image {idx:>3}: {elapsed:6.2f}s  ({tag})")
    except Exception as e:
        elapsed = time.perf_counter() - start
        results.append({"idx": idx, "latency": elapsed, "ok": False,
                        "status": "exception"})
        print(f"  image {idx:>3}: {elapsed:6.2f}s  (FAILED: {e})")


async def run(args):
    url = f"{args.base_url.rstrip('/')}/v1/images/generations"

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload_base = {
        "model": args.model,
        "prompt": args.prompt,
        "n": 1,
        "size": args.size,
        "num_inference_steps": args.steps,
    }

    results = []
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(idx):
        async with sem:
            await one_request(session, url, headers,
                              dict(payload_base), idx, results)

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    print(f"\nBenchmarking FLUX at {args.size}, {args.steps} steps, "
          f"{args.num_images} images\n")
    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(guarded(i) for i in range(1, args.num_images + 1)))
    wall_total = time.perf_counter() - wall_start

    # ---- Resolution info --------------------------------------------------
    w, h = (int(x) for x in args.size.split("x"))
    megapixels = (w * h) / 1_000_000

    # ---- Report -----------------------------------------------------------
    oks = [r["latency"] for r in results if r["ok"]]
    n_ok = len(oks)
    n_fail = len(results) - n_ok

    print("\n" + "=" * 52)
    print(" FLUX.1 [schnell] benchmark results")
    print("=" * 52)
    print(f" config:          {args.size}, {args.steps} steps")
    print(f" megapixels:      {megapixels:.3f} MP")
    print(f" images ok:       {n_ok}")
    print(f" images failed:   {n_fail}")
    print(f" wall-clock:      {wall_total:6.2f}s")
    if n_ok:
        oks_sorted = sorted(oks)
        def pct(p):
            k = max(0, min(len(oks_sorted) - 1,
                           int(round(p / 100 * (len(oks_sorted) - 1)))))
            return oks_sorted[k]
        imgs_per_min = n_ok / wall_total * 60
        print(f" mean latency:    {statistics.mean(oks):6.2f}s / image")
        print(f" p99:             {pct(99):6.2f}s")
        print(f" min / max:       {min(oks):.2f}s / {max(oks):.2f}s")
        print(f" throughput:      {imgs_per_min:6.1f} images / minute")
        print(f"                  {n_ok / wall_total:6.2f} images / second")
    print("=" * 52 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="FLUX text-to-image benchmark")
    p.add_argument(
        "--api-key",
        default=os.environ.get("STONEY_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="API key (or set STONEY_KEY env variable)",
    )
    p.add_argument(
        "--base-url",
        default="https://llm.stoney-cloud.com",
        help="API base URL (default: https://llm.stoney-cloud.com)",
    )
    p.add_argument(
        "--model",
        required=True,
        help="model id as shown by /v1/models",
    )
    p.add_argument(
        "--prompt",
        default="a photorealistic mountain landscape at golden hour",
    )
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="simultaneous requests (default: 1)",
    )
    p.add_argument("--size", default="1024x1024",
                   help="image resolution e.g. 1024x1024 (default: 1024x1024)")
    p.add_argument(
        "--steps",
        type=int,
        default=4,
        help="denoising steps (default: 4)",
    )
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))