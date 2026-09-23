#!/usr/bin/env python3
"""Measure native profile-lane throughput and output isolation.

Example:
  python3 tools/friend-bench/lane_benchmark.py --model ~/models/model.gguf --lanes 1 2 4

This is intentionally a small live-server benchmark rather than a synthetic
microbenchmark: it exercises request admission, lane routing, batching, KV
ownership, and result collection together.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(base, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/api/v1/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())


def run(model, lanes, count):
    port = free_port()
    slots = max(4, lanes)
    command = [sys.executable, str(ROOT / "koboldcpp.py"), "--model", str(model),
               "--port", str(port), "--contextsize", "2048", "--gpulayers", "99",
               "--skiplauncher", "--quiet", "--parallelrequests", str(slots),
               "--profile-lanes", str(lanes), "--prefill-tokens", "32",
               "--schedule-tokens", "64", "--noshift"]
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                with urllib.request.urlopen(base + "/api/extra/requests", timeout=2):
                    break
            except (OSError, ValueError):
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"lane benchmark server failed to start (lanes={lanes})")
                time.sleep(.1)
        payloads = [
            {"prompt": f"Continue this short deterministic answer number {i}: ",
             "max_length": 32, "temperature": 0, "top_k": 0, "top_p": 1,
             "seed": 1000 + i, "cache_salt": f"bench-{i}"}
            for i in range(count)
        ]
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            results = list(pool.map(lambda p: request(base, p), payloads))
        elapsed = time.monotonic() - started
        texts = [r["results"][0]["text"] for r in results]
        if not all(texts):
            raise AssertionError("empty benchmark result")
        return {"lanes": lanes, "requests": count, "seconds": elapsed,
                "requests_per_second": count / elapsed, "text_lengths": [len(t) for t in texts]}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--requests", type=int, default=4)
    args = parser.parse_args()
    if any(lane < 1 or lane > 32 for lane in args.lanes) or args.requests < 1:
        parser.error("lanes must be 1..32 and requests must be positive")
    for lane in args.lanes:
        print(json.dumps(run(args.model.expanduser(), lane, args.requests), sort_keys=True), flush=True)
