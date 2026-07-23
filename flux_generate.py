#!/usr/bin/env python3
"""
Generate a FLUX image via the Stoney Cloud API and save it to disk.
 
Usage:
  python flux_generate.py \
      --api-key sk-your-key-here \
      --prompt "a red panda astronaut floating in space, photorealistic"
 
  # or set the key as an environment variable:
  export STONEY_KEY=sk-your-key-here
  python flux_generate.py --prompt "a Swiss mountain at golden hour"
 
  # custom resolution and steps:
  python flux_generate.py \
      --api-key sk-your-key-here \
      --prompt "a minimalist logo on white background" \
      --size 1024x1024 \
      --steps 4 \
      --out my_image.png 
"""
 
import argparse
import base64
import json
import os
import urllib.request
import urllib.error
 
 
def main():
    p = argparse.ArgumentParser(
        description="Generate a FLUX image via the Stoney Cloud API"
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("STONEY_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="your Stoney Cloud API key (or set STONEY_KEY env variable)",
    )
    p.add_argument(
        "--base-url",
        default="https://llm.stoney-cloud.com",
        help="API base URL (default: https://llm.stoney-cloud.com)",
    )
    p.add_argument(
        "--model",
        default="black-forest-labs/FLUX.1-schnell",
        help="model name as served by the API",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="image description (if omitted you will be asked interactively)",
    )
    p.add_argument(
        "--size",
        default="1024x1024",
        help="image resolution (default: 1024x1024)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=4,
        help="denoising steps 1-16 — more steps = higher quality but slower (default: 4)",
    )
    p.add_argument(
        "--out",
        default="flux_image.png",
        help="output filename (default: flux_image.png)",
    )
    args = p.parse_args()
 
    # ── API key ────────────────────────────────────────────────────────────────
    if not args.api_key:
        print(
            "Error: no API key found.\n"
            "Pass it with --api-key or set the STONEY_KEY environment variable:\n"
            "  export STONEY_KEY=sk-your-key-here"
        )
        return
 
    # ── Prompt ─────────────────────────────────────────────────────────────────
    if not args.prompt:
        args.prompt = input("Enter your image prompt: ").strip()
        if not args.prompt:
            print("No prompt given, exiting.")
            return
 
    # ── Request ────────────────────────────────────────────────────────────────
    url = f"{args.base_url.rstrip('/')}/v1/images/generations"
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "n": 1,
        "size": args.size,
        "num_inference_steps": args.steps,
    }
 
    print(f"\nGenerating: \"{args.prompt}\"")
    print(f"Config:     {args.size}, {args.steps} steps")
    print(f"Endpoint:   {url}\n")
 
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
    )
 
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        print(f"Error {e.code}: {e.reason}")
        if e.code == 401:
            print("Check your API key — authentication failed.")
        elif e.code == 404:
            print("Model not found — check the --model name.")
        else:
            print(f"Response: {body[:500]}")
        return
    except urllib.error.URLError as e:
        print(f"Connection failed: {e.reason}")
        print("Check your internet connection and the --base-url.")
        return
 
    # ── Save image ─────────────────────────────────────────────────────────────
    item = data["data"][0]
    if "b64_json" in item and item["b64_json"]:
        img_bytes = base64.b64decode(item["b64_json"])
        with open(args.out, "wb") as f:
            f.write(img_bytes)
        print(f"Saved → {args.out}")
    elif "url" in item:
        print(f"Server returned a URL instead of base64: {item['url']}")
        print("Download it directly from that URL.")
    else:
        print("Unexpected response shape. Raw keys:", list(item.keys()))
        print(json.dumps(data, indent=2)[:1000])
 
 
if __name__ == "__main__":
    main()