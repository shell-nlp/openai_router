from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import httpx


BASE_URL = os.getenv("OPENAI_ROUTER_BASE_URL", "http://127.0.0.1:28000")
API_KEY = os.getenv("OPENAI_ROUTER_API_KEY", "")

PAYLOAD = {
    "model": "qwen3.5",
    "stream": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    # "enable_thinking": True,
    # "tool_choice": "auto",
    # "stream_options": {"include_usage": True},
    "messages": [
        {
            "role": "user",
            "content": "你好",
        }
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5, help="重复请求次数，默认 5")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=-1,
        help="并发数，默认 -1（等于 repeat，全部并行）",
    )
    return parser.parse_args()


def run_once(request_index: int) -> tuple[int, float, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    url = f"{BASE_URL.rstrip('/')}/v1/chat/completions"
    started_at = perf_counter()
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, headers=headers, json=PAYLOAD) as response:
            response.raise_for_status()
            parts: list[str] = []
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or delta.get("reasoning") or delta.get(
                    "thinking"
                )
                if text:
                    parts.append(text)

    elapsed = perf_counter() - started_at
    return request_index, elapsed, "".join(parts)


def main() -> None:
    args = parse_args()
    repeat = max(1, args.repeat)
    concurrency = repeat if args.concurrency < 0 else max(1, args.concurrency)

    batch_started_at = perf_counter()
    results: dict[int, tuple[float, str]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_once, index) for index in range(repeat)]
        for future in as_completed(futures):
            request_index, elapsed, content = future.result()
            results[request_index] = (elapsed, content)

    batch_elapsed = perf_counter() - batch_started_at
    total_request_elapsed = 0.0
    max_request_elapsed = 0.0

    for index in range(repeat):
        elapsed, content = results.get(index, (0.0, ""))
        total_request_elapsed += elapsed
        max_request_elapsed = max(max_request_elapsed, elapsed)
        print(f"\n--- request {index + 1}/{repeat} ---")
        print(f"time: {elapsed:.3f}s")
        print(content)

    print("\n=== summary ===")
    print(f"batch_time: {batch_elapsed:.3f}s")
    print(f"sum_request_time: {total_request_elapsed:.3f}s")
    print(f"max_request_time: {max_request_elapsed:.3f}s")
    if max_request_elapsed > 0:
        print(f"batch_vs_max: {batch_elapsed / max_request_elapsed:.2f}x")


if __name__ == "__main__":
    main()
