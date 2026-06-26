import argparse
import base64
import concurrent.futures
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

CHARS_PER_TOKEN = 4


def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p / 100)]


def request_one(
    image: Path,
    endpoint: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + encoded},
                    }
                ],
            }
        ],
        "max_tokens": max_tokens,
        # "temperature": 0.2,
        # "top_p": 0.9,
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    t0 = time.perf_counter()
    first_token = None
    chars = 0
    completion_tokens = 0
    prompt_tokens = 0

    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            event = line[6:]
            if event == "[DONE]":
                break
            try:
                obj = json.loads(event)
            except json.JSONDecodeError:
                continue
            # capture text deltas for timing + char fallback
            try:
                text = obj["choices"][0].get("delta", {}).get("content") or ""
            except (KeyError, IndexError):
                text = ""
            if text:
                first_token = first_token or time.perf_counter()
                chars += len(text)
            # capture real token counts from the server's usage field
            # (usually present in the final chunk when streaming)
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

    t1 = time.perf_counter()
    return {
        "image": str(image),
        "latency_s": t1 - t0,
        "ttft_s": (first_token - t0) if first_token else (t1 - t0),
        "chars": chars,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:40000/v1/chat/completions"
    )
    parser.add_argument("--data", type=Path, default=Path("cv_bench_data"))
    parser.add_argument("--model", default="lightonai/LightOnOCR-2-1B")
    parser.add_argument("--api-key", default=os.environ.get("STONEY_API_KEY"))
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="skip the readable table on stderr, CSV only",
    )
    args = parser.parse_args()

    images = sorted(args.data.glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no PNG files found in {args.data}")

    start = time.perf_counter()
    ok: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_one,
                image,
                args.endpoint,
                args.model,
                args.api_key,
                args.max_tokens,
                args.timeout,
            )
            for image in images
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                ok.append(future.result())
            except Exception as exc:
                errors.append(repr(exc))

    wall = time.perf_counter() - start
    latencies = [r["latency_s"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok]
    total_chars = sum(r["chars"] for r in ok)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    pages_s = len(ok) / wall if wall else 0.0
    requests_s = pages_s  # one request == one page for OCR
    # prefer the server-reported token count; fall back to char estimate
    if total_completion_tokens > 0:
        out_tok_s = total_completion_tokens / wall if wall else 0.0
        tok_source = "server"
    else:
        out_tok_s = (total_chars / CHARS_PER_TOKEN) / wall if wall else 0.0
        tok_source = "estimated"

    def fmt(value: float | None) -> float | str:
        return round(value, 3) if value is not None else ""

    # ---- CSV row on stdout (for piping / appending to a results file) ----
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "endpoint",
            "requested",
            "ok",
            "failed",
            "concurrency",
            "wall_s",
            "pages_s",
            "requests_s",
            "out_tok_s",
            "tok_source",
            "latency_p50_s",
            "latency_p95_s",
            "latency_p99_s",
            "latency_max_s",
            "ttft_p50_s",
            "ttft_p95_s",
            "ttft_p99_s",
            "ttft_max_s",
        ]
    )
    writer.writerow(
        [
            args.endpoint,
            len(images),
            len(ok),
            len(errors),
            args.concurrency,
            round(wall, 3),
            round(pages_s, 3),
            round(requests_s, 3),
            round(out_tok_s, 3),
            tok_source,
            fmt(percentile(latencies, 50)),
            fmt(percentile(latencies, 95)),
            fmt(percentile(latencies, 99)),
            fmt(max(latencies) if latencies else None),
            fmt(percentile(ttfts, 50)),
            fmt(percentile(ttfts, 95)),
            fmt(percentile(ttfts, 99)),
            fmt(max(ttfts) if ttfts else None),
        ]
    )

    # ---- readable table on stderr (for watching live in the terminal) ----
    if not args.no_pretty:
        rows = [
            ("concurrency", args.concurrency),
            ("requested", len(images)),
            ("ok", len(ok)),
            ("failed", len(errors)),
            ("duration_s", round(wall, 3)),
            ("pages_s", round(pages_s, 3)),
            ("pages_min", round(pages_s * 60, 1)),
            (f"out_tok_s", round(out_tok_s, 1)),
            ("latency_p50_s", fmt(percentile(latencies, 50))),
            ("latency_p99_s", fmt(percentile(latencies, 99))),
            
        ]
        label_width = max(len(label) for label, _ in rows)
        print(f"\n  --- benchmark result (concurrency {args.concurrency}) ---", file=sys.stderr)
        for label, value in rows:
            print(f"  {label:<{label_width}} : {value}", file=sys.stderr)
        print(file=sys.stderr)

    if errors:
        print(f"{len(errors)} request(s) failed; first: {errors[0]}", file=sys.stderr)


if __name__ == "__main__":
    main()
