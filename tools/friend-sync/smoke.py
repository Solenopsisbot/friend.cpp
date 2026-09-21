#!/usr/bin/env python3
"""
friend.cpp smoke tests: start a real koboldcpp server per model, poke the features
friend.cpp adds on top of koboldcpp, kill it, print a table.

Meant to run after every upstream sync (tools/friend-sync/sync.sh runs it for you),
but it works on any built tree:

    python3 tools/friend-sync/smoke.py                      # all default models found
    python3 tools/friend-sync/smoke.py --models ~/models/Bonsai-1.7B-Q1_0.gguf
    python3 tools/friend-sync/smoke.py --only coherent,cache --keep-logs /tmp/smoke

Stdlib only. Needs a built koboldcpp library next to koboldcpp.py. The head check
also needs `uv` (to run tools/friend-heads/extract_head.py with numpy); without it
that check is skipped, not failed.

Per model it launches up to four servers (ports from 5090-5095, whichever are free):

  main   GPU, not --quiet (the cache check greps the log), with --head-pool self=<the
         model's own head> and --cvec-dir <tmp>:
           coherent     sampled text is non-empty and mostly printable
           determinism  same seed twice gives the same text
           blue_noise   blue-noise sampling generates, and is itself seed-deterministic
           head         "head":"self" (a copy of the model's own head) reproduces base
                        greedy output bit-for-bit; an unknown head fails the request
           cache        persona A, persona B, A again: the log says
                        "[Prompt cache: reusing" and A's output matches a --cache-ram 0 run
           steer        /api/extra/steer/build returns ok; "steer" changes greedy
                        output and dropping it returns to the base output
  nocache  GPU, --cache-ram 0: the reference output for the cache check
  cpu      --gpulayers 0: greedy output agrees with the GPU for the first 16 tokens
  batch    --parallelrequests 4 --noshift, same head pool and the steering vector the
           main server saved: mixed-profile requests fired concurrently match the same
           requests run one at a time

Exit code 0 only if nothing FAILed (SKIPs are fine). Models are configurable with
--models or FRIEND_SMOKE_MODELS (comma or colon separated); missing files are skipped.
Extra server flags (e.g. --usecuda on a CUDA build) go in --server-flags or
FRIEND_SMOKE_FLAGS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = [
    "~/models/Bonsai-1.7B-Q1_0.gguf",          # Q1_0: upstream llama.cpp type
    "~/models/Ternary-Bonsai-1.7B-PQ2_0.gguf",  # PQ2_0: Prism-only type
    "~/models/Qwen3.5-0.8B-Q4_0.gguf",          # hybrid (recurrent + attention)
]
PORTS = list(range(5090, 5096))
ALL_CHECKS = ["coherent", "determinism", "blue_noise", "head", "cache", "steer", "gpu_cpu", "batch"]
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "friend-smoke"


# ---------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------

def chat(system: str, user: str) -> str:
    """ChatML prompt ending inside the assistant turn. Every default model (Bonsai is
    Qwen3-based, Qwen3.5) speaks ChatML; the turn boundaries also matter for the hybrid
    model, whose prompt cache checkpoints right before `<|im_start|>`. The empty think
    block is Qwen3's non-thinking mode: without it the first few generated tokens are
    template noise, which makes "the first 16 tokens agree" a much weaker check."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


SIMPLE = chat("You are a helpful assistant.", "Write two sentences about the ocean.")
GREEDY = chat("You are a helpful assistant.", "What is the capital of France? Answer in one sentence.")

# Persona cards need to be comfortably above --cache-min-tokens (64) so they get cached.
_ROOK = ("You are Rook, a cheerful and slightly chaotic fox spirit who lives in a small "
         "bookshop at the edge of a rainy harbour town. You love puns, warm tea, old maps, "
         "and the smell of paper. You speak casually, tease your friends gently, and always "
         "end up rambling about some obscure bit of folklore. You never break character, "
         "you never mention being an AI, and you keep replies short and lively. ")
_MIRA = ("You are Mira, a calm, precise astronomer who works the night shift at a "
           "mountain observatory. You are dry, kind, and quietly funny, you like exact "
           "numbers, strong coffee, and the silence before dawn. You explain things "
           "patiently, you correct misconceptions without condescension, and you always "
           "relate conversations back to the sky somehow. Keep replies short. ")
PERSONA_A = chat(_ROOK * 3, "hey rook, what are you reading today?")
PERSONA_B = chat(_MIRA * 3, "mira, what can you see tonight?")

STEER_POS = [chat("You are wildly excited and overjoyed about absolutely everything!", q) + "Oh"
             for q in ("Tell me about my cat.", "How was your day?", "What do you think of rain?",
                       "Describe a sandwich.")]
STEER_NEG = [chat("You are bored, flat and utterly indifferent about everything.", q) + "Oh"
             for q in ("Tell me about my cat.", "How was your day?", "What do you think of rain?",
                       "Describe a sandwich.")]


# ---------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------

@dataclass
class Result:
    model: str
    check: str
    status: str  # PASS / FAIL / SKIP
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Ctx:
    args: argparse.Namespace
    results: list[Result] = field(default_factory=list)

    def add(self, model: str, check: str, status: str, detail: str = "", seconds: float = 0.0):
        r = Result(model, check, status, detail, seconds)
        self.results.append(r)
        colour = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}.get(status, "")
        reset = "\033[0m" if colour and sys.stdout.isatty() else ""
        colour = colour if reset else ""
        print(f"  {colour}{status:<4}{reset} {check:<12} {detail}", flush=True)

    def wants(self, check: str) -> bool:
        return check in self.args.only_set


# ---------------------------------------------------------------------------------
# server process management
# ---------------------------------------------------------------------------------

_live_servers: list["Server"] = []


def port_free(port: int) -> bool:
    # koboldcpp binds every interface; a bind to 127.0.0.1 can succeed on macOS even
    # while something holds the wildcard address, so try connecting first
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


_port_rr = 0


def pick_port() -> int:
    """Round-robin over 5090-5095 so a just-killed server's socket never gets in the
    way, skipping ports something else (another agent's server?) is using."""
    global _port_rr
    for _ in range(len(PORTS)):
        p = PORTS[_port_rr % len(PORTS)]
        _port_rr += 1
        if port_free(p):
            return p
    raise RuntimeError(f"no free port in {PORTS[0]}-{PORTS[-1]}")


class Server:
    def __init__(self, ctx: Ctx, name: str, model: Path, flags: list[str], quiet: bool = True):
        self.ctx, self.name, self.model = ctx, name, model
        self.port = pick_port()
        self.log_path = Path(ctx.args.log_dir) / f"{model.stem}.{name}.log"
        cmd = [ctx.args.python, str(REPO / "koboldcpp.py"), "--model", str(model),
               "--port", str(self.port), "--contextsize", "4096", "--gpulayers", "99",
               "--skiplauncher"]
        if quiet:
            cmd.append("--quiet")
        cmd += ctx.args.server_flag_list + flags
        # a later --gpulayers in flags overrides the default 99 (argparse keeps the last)
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "Server":
        self.log = open(self.log_path, "w")
        self.log.write("$ " + shlex.join(self.cmd) + "\n\n")
        self.log.flush()
        # own process group so we can kill koboldcpp and anything it spawned in one go
        self.proc = subprocess.Popen(self.cmd, cwd=str(REPO), stdout=self.log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        _live_servers.append(self)
        deadline = time.time() + self.ctx.args.start_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} server exited with {self.proc.returncode} during startup "
                                   f"(log: {self.log_path})\n{self.tail(15)}")
            try:
                with urllib.request.urlopen(self.base + "/api/v1/model", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"{self.name} server not ready after {self.ctx.args.start_timeout}s (log: {self.log_path})")

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        if self in _live_servers:
            _live_servers.remove(self)
        if hasattr(self, "log") and not self.log.closed:
            self.log.close()

    # -- HTTP helpers ----------------------------------------------------------------

    def post(self, path: str, body: dict, timeout: float = 180) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return {"_http_error": e.code, "_body": e.read().decode("utf-8", "replace")[:500]}

    def generate(self, prompt: str, max_length: int = 32, greedy: bool = True, **extra) -> dict:
        """POST /api/v1/generate and return results[0] ({"text", "finish_reason", ...})."""
        body = {"prompt": prompt, "max_length": max_length, "rep_pen": 1.0,
                "trim_stop": False, "stop_sequence": []}
        if greedy:
            body.update(temperature=0.0, top_k=1)
        else:
            body.update(temperature=0.8, top_k=40, top_p=0.95, min_p=0.02)
        body.update(extra)
        out = self.post("/api/v1/generate", body)
        if "results" not in out:
            raise RuntimeError(f"generate failed: {json.dumps(out)[:300]}")
        return out["results"][0]

    def text(self, prompt: str, max_length: int = 32, greedy: bool = True, **extra) -> str:
        return self.generate(prompt, max_length, greedy, **extra)["text"]

    def log_text(self) -> str:
        self.log.flush()
        return self.log_path.read_text(errors="replace")

    def tail(self, n: int) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""


def cleanup(*_):
    for s in list(_live_servers):
        s.stop()


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------

def printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for c in s if c.isprintable() or c in "\n\t")
    return ok / len(s)


def short(s: str, n: int = 50) -> str:
    s = s.replace("\n", "\\n")
    return repr(s[:n] + ("..." if len(s) > n else ""))


def first_diff(a: str, b: str) -> str:
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return f"differ at char {i}: {short(a[i:], 30)} vs {short(b[i:], 30)}"


def extract_head(ctx: Ctx, model: Path) -> tuple[Path | None, str]:
    """Extract the model's own LM head with tools/friend-heads/extract_head.py, cached in
    ~/.cache/friend-smoke keyed by path + size + mtime (the Qwen3.5 head is ~140 MB)."""
    if not shutil.which("uv"):
        return None, "uv not found"
    st = model.stat()
    key = hashlib.sha1(f"{model.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{model.stem}.{key}.head.gguf"
    if out.exists() and out.stat().st_size > 0:
        return out, "cached"
    tmp = out.with_suffix(".tmp")
    cmd = ["uv", "run", "--quiet", "--with", "numpy", "--with", "pyyaml", "--with", "tqdm",
           "python", str(REPO / "tools/friend-heads/extract_head.py"), str(model), str(tmp)]
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if p.returncode != 0 or not tmp.exists():
        return None, "extract_head.py failed: " + (p.stderr.strip().splitlines() or ["?"])[-1]
    tmp.rename(out)
    return out, "extracted"


# ---------------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------------

def run_model(ctx: Ctx, model: Path):
    name = model.stem
    print(f"\n=== {name} ===", flush=True)
    work = Path(tempfile.mkdtemp(prefix=f"friend-smoke-{name}-"))
    cvec_dir = work / "cvecs"
    cvec_dir.mkdir()

    head, head_note = (None, "not requested")
    if ctx.wants("head") or ctx.wants("batch"):
        t = time.time()
        head, head_note = extract_head(ctx, model)
        if head is None and ctx.wants("head"):
            ctx.add(name, "head", "SKIP", head_note)
        elif head is not None:
            print(f"  head: {head.name} ({head_note}, {time.time() - t:.1f}s)", flush=True)

    shared: dict = {}  # outputs handed from one server to the next
    main_flags = ["--cvec-dir", str(cvec_dir)]
    if head is not None:
        main_flags += ["--head-pool", f"self={head}"]

    # -- main server -----------------------------------------------------------------
    main_checks = [c for c in ("coherent", "determinism", "blue_noise", "head", "cache", "steer", "gpu_cpu")
                   if ctx.wants(c)]
    if main_checks:
        try:
            with Server(ctx, "main", model, main_flags, quiet=False) as s:
                if ctx.wants("coherent"):
                    timed(ctx, name, "coherent", check_coherent, s)
                if ctx.wants("determinism"):
                    timed(ctx, name, "determinism", check_determinism, s)
                if ctx.wants("blue_noise"):
                    timed(ctx, name, "blue_noise", check_blue_noise, s)
                shared["greedy"] = s.text(GREEDY, 32)
                shared["greedy16"] = s.text(GREEDY, 16)
                if ctx.wants("head") and head is not None:
                    timed(ctx, name, "head", check_head, s, shared)
                if ctx.wants("cache"):
                    timed(ctx, name, "cache", check_cache_main, s, shared)
                if ctx.wants("steer"):
                    timed(ctx, name, "steer", check_steer, s, shared)
        except Exception as e:
            ctx.add(name, "main-server", "FAIL", str(e).splitlines()[0])
            for line in str(e).splitlines()[1:]:
                print("       | " + line)
            shutil.rmtree(work, ignore_errors=True)
            return

    # -- reference run with the prompt cache off -------------------------------------
    if ctx.wants("cache") and "cache_a2" in shared:
        t = time.time()
        try:
            with Server(ctx, "nocache", model, ["--cache-ram", "0"]) as s:
                ref = s.text(PERSONA_A, 24)
            if ref == shared["cache_a2"]:
                ctx.add(name, "cache", "PASS", shared["cache_note"] + ", output == --cache-ram 0",
                        shared["cache_secs"] + time.time() - t)
            else:
                ctx.add(name, "cache", "FAIL", "reused output != --cache-ram 0 output: "
                        + first_diff(shared["cache_a2"], ref), shared["cache_secs"] + time.time() - t)
        except Exception as e:
            ctx.add(name, "cache", "FAIL", f"nocache server: {str(e).splitlines()[0]}")

    # -- CPU vs GPU greedy -------------------------------------------------------------
    if ctx.wants("gpu_cpu") and "greedy16" in shared:
        t = time.time()
        try:
            with Server(ctx, "cpu", model, ["--gpulayers", "0"]) as s:
                cpu = s.text(GREEDY, 16)
                gpu = shared["greedy16"]
                if cpu == gpu and cpu.strip():
                    ctx.add(name, "gpu_cpu", "PASS", f"16 greedy tokens agree: {short(cpu, 40)}", time.time() - t)
                else:
                    # say how many tokens they share, which is the number that matters
                    ids_c = s.post("/api/extra/tokencount", {"prompt": cpu, "special": False}).get("ids", [])
                    ids_g = s.post("/api/extra/tokencount", {"prompt": gpu, "special": False}).get("ids", [])
                    same = next((k for k in range(min(len(ids_c), len(ids_g))) if ids_c[k] != ids_g[k]),
                                min(len(ids_c), len(ids_g)))
                    ctx.add(name, "gpu_cpu", "FAIL", f"agree on {same}/16 tokens: GPU {short(gpu, 40)} "
                            f"CPU {short(cpu, 40)}", time.time() - t)
        except Exception as e:
            ctx.add(name, "gpu_cpu", "FAIL", f"cpu server: {str(e).splitlines()[0]}")

    # -- continuous batching -------------------------------------------------------------
    if ctx.wants("batch"):
        t = time.time()
        flags = ["--parallelrequests", "4", "--noshift", "--cvec-dir", str(cvec_dir)]
        if head is not None:
            flags += ["--head-pool", f"self={head}"]
        try:
            with Server(ctx, "batch", model, flags) as s:
                check_batch(ctx, name, s, head is not None, shared.get("steer_ok", False), t)
        except Exception as e:
            ctx.add(name, "batch", "FAIL", f"batch server: {str(e).splitlines()[0]}")

    shutil.rmtree(work, ignore_errors=True)


def timed(ctx: Ctx, model: str, check: str, fn, *a):
    t = time.time()
    try:
        r = fn(*a)
    except Exception as e:
        ctx.add(model, check, "FAIL", f"exception: {e}", time.time() - t)
        return
    if r is None:  # the check reports itself later (e.g. cache, finished by another server)
        return
    status, detail = r
    ctx.add(model, check, status, detail, time.time() - t)


def check_coherent(s: Server):
    out = s.text(SIMPLE, 48, greedy=False, sampler_seed=7)
    ratio = printable_ratio(out)
    letters = sum(c.isalpha() for c in out)
    if not out.strip():
        return "FAIL", "empty output"
    if ratio < 0.9 or letters < len(out) * 0.4:
        return "FAIL", f"garbled (printable {ratio:.0%}, letters {letters}/{len(out)}): {short(out)}"
    if len(set(out.split())) < 3:
        return "FAIL", f"degenerate: {short(out)}"
    return "PASS", short(out)


def check_determinism(s: Server):
    a = s.text(SIMPLE, 40, greedy=False, sampler_seed=1234)
    b = s.text(SIMPLE, 40, greedy=False, sampler_seed=1234)
    if a != b:
        return "FAIL", "seed 1234 twice: " + first_diff(a, b)
    c = s.text(SIMPLE, 40, greedy=False, sampler_seed=4321)
    note = "" if c != a else " (warning: seed 4321 gave the same text)"
    return "PASS", f"seed 1234 reproducible ({len(a)} chars){note}"


def check_blue_noise(s: Server):
    a = s.text(SIMPLE, 40, greedy=False, sampler_seed=99, blue_noise=True, temperature=1.0)
    b = s.text(SIMPLE, 40, greedy=False, sampler_seed=99, blue_noise=True, temperature=1.0)
    c = s.text(SIMPLE, 40, greedy=False, sampler_seed=99, blue_noise=True, temperature=1.0, rng_type="lowbias32")
    if not a.strip() or printable_ratio(a) < 0.9:
        return "FAIL", f"bad output: {short(a)}"
    if a != b:
        return "FAIL", "not seed-deterministic: " + first_diff(a, b)
    if not c.strip():
        return "FAIL", "lowbias32 rng gave empty output"
    return "PASS", f"deterministic, lowbias32 ok: {short(a, 40)}"


def check_head(s: Server, shared: dict):
    base = shared["greedy"]
    swapped = s.text(GREEDY, 32, head="self")
    back = s.text(GREEDY, 32)
    bad = s.generate(GREEDY, 4, head="no-such-head")
    if swapped != base:
        return "FAIL", "head=self != base: " + first_diff(swapped, base)
    if back != base:
        return "FAIL", "base after head swap changed: " + first_diff(back, base)
    if bad.get("finish_reason") != "error":
        return "FAIL", f"unknown head did not error (finish_reason={bad.get('finish_reason')!r})"
    return "PASS", "head=self reproduces base greedy; unknown head errors"


def check_cache_main(s: Server, shared: dict):
    """Persona A, persona B, A again. Only the reuse half is judged here; the output is
    compared against a fresh --cache-ram 0 server afterwards (see run_model)."""
    t = time.time()
    s.text(PERSONA_A, 24)
    s.text(PERSONA_B, 24)
    mark = len(s.log_text())
    a2 = s.text(PERSONA_A, 24)
    log = s.log_text()[mark:]
    reuse = [l for l in log.splitlines() if "[Prompt cache: reusing" in l]
    if not reuse:
        return "FAIL", "no '[Prompt cache: reusing' line after returning to persona A"
    shared["cache_a2"] = a2
    shared["cache_note"] = reuse[-1].strip("[] ").replace("Prompt cache: ", "")
    shared["cache_secs"] = time.time() - t
    return None


def check_steer(s: Server, shared: dict):
    r = s.post("/api/extra/steer/build", {"name": "smoke", "positive": STEER_POS, "negative": STEER_NEG})
    if not r.get("ok"):
        return "FAIL", f"steer/build: {r.get('error') or json.dumps(r)[:200]}"
    layers = r.get("layers")
    base = shared["greedy"]
    steered = s.text(GREEDY, 32, steer={"smoke": 4.0})
    back = s.text(GREEDY, 32)
    if steered == base:
        return "FAIL", f"steer smoke=4 did not change greedy output (layers {layers})"
    if back != base:
        return "FAIL", "output without steer != base after steering: " + first_diff(back, base)
    shared["steer_ok"] = True
    saved = " saved" if r.get("path") else ""
    return "PASS", f"built{saved} (layers {layers}), changes output: {short(steered, 36)}"


def check_batch(ctx: Ctx, name: str, s: Server, have_head: bool, have_steer: bool, t0: float):
    """Mixed-profile requests (base / head / steer / sampled / different prompts), first one
    at a time, then all at once. Continuous batching must not change any of them."""
    reqs: list[tuple[str, dict]] = [
        ("base", dict(prompt=GREEDY, max_length=32)),
        ("persona", dict(prompt=PERSONA_A, max_length=32)),
        ("sampled", dict(prompt=SIMPLE, max_length=32, greedy=False, sampler_seed=5)),
        ("persona-b", dict(prompt=PERSONA_B, max_length=32)),
    ]
    if have_head:
        reqs.append(("head", dict(prompt=GREEDY, max_length=32, head="self")))
    if have_steer:
        reqs.append(("steer", dict(prompt=SIMPLE, max_length=32, steer={"smoke": 3.0})))

    def run(kw):
        kw = dict(kw)
        return s.text(kw.pop("prompt"), kw.pop("max_length"), kw.pop("greedy", True), **kw)

    serial = {label: run(kw) for label, kw in reqs}
    concurrent: dict[str, str] = {}
    errors: list[str] = []

    def worker(label, kw):
        try:
            concurrent[label] = run(kw)
        except Exception as e:  # noqa: BLE001 - report every failure, not just the first
            errors.append(f"{label}: {e}")

    mark = len(s.log_text())
    threads = [threading.Thread(target=worker, args=(label, kw)) for label, kw in reqs]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=300)
    # koboldcpp prints "BatchRequest:<id>, ..." per request served by the continuous-batching
    # worker, even under --quiet. Zero of them means everything fell back to the
    # one-at-a-time path, and a match would prove nothing.
    n_batched = s.log_text()[mark:].count("BatchRequest:")
    mism = [l for l, _ in reqs if concurrent.get(l) != serial[l]]
    profiles = "+".join(l for l, _ in reqs)
    if errors:
        ctx.add(name, "batch", "FAIL", "; ".join(errors)[:200], time.time() - t0)
    elif any(not v.strip() for v in serial.values()):
        empty = [l for l, v in serial.items() if not v.strip()]
        ctx.add(name, "batch", "FAIL", f"empty output for {empty}", time.time() - t0)
    elif mism:
        l = mism[0]
        ctx.add(name, "batch", "FAIL", f"{len(mism)}/{len(reqs)} differ, e.g. {l}: "
                + first_diff(concurrent.get(l, ""), serial[l]), time.time() - t0)
    elif n_batched == 0:
        ctx.add(name, "batch", "FAIL", "outputs match but no request went through the batching worker "
                "(no 'BatchRequest:' in the log)", time.time() - t0)
    else:
        ctx.add(name, "batch", "PASS", f"{len(reqs)} concurrent == serial, {n_batched} via batch worker "
                f"({profiles})", time.time() - t0)


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", help="model GGUFs (default: FRIEND_SMOKE_MODELS or the three ~/models defaults)")
    ap.add_argument("--only", help=f"comma list of checks to run ({','.join(ALL_CHECKS)})")
    ap.add_argument("--skip", help="comma list of checks to skip")
    ap.add_argument("--server-flags", default=os.environ.get("FRIEND_SMOKE_FLAGS", ""),
                    help="extra koboldcpp flags for every server, e.g. '--usecuda' (env FRIEND_SMOKE_FLAGS)")
    ap.add_argument("--python", default=sys.executable, help="python used to run koboldcpp.py")
    ap.add_argument("--keep-logs", metavar="DIR", help="write server logs here and keep them (default: temp dir, "
                    "kept only on failure)")
    ap.add_argument("--start-timeout", type=float, default=240, help="seconds to wait for a server to come up")
    args = ap.parse_args()

    models = args.models
    if not models:
        env = os.environ.get("FRIEND_SMOKE_MODELS", "")
        models = [m for m in env.replace(":", ",").split(",") if m.strip()] if env else DEFAULT_MODELS
    only = set(ALL_CHECKS if not args.only else [c.strip() for c in args.only.split(",")])
    only -= set(c.strip() for c in (args.skip or "").split(",") if c.strip())
    unknown = only - set(ALL_CHECKS)
    if unknown:
        ap.error(f"unknown checks: {', '.join(sorted(unknown))}")
    args.only_set = only
    args.server_flag_list = shlex.split(args.server_flags)
    args.log_dir = args.keep_logs or tempfile.mkdtemp(prefix="friend-smoke-logs-")
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if not any((REPO / f).exists() for f in ("koboldcpp_default.so", "koboldcpp_default.dll",
                                              "koboldcpp_cublas.so", "koboldcpp_vulkan.so")):
        print(f"no built koboldcpp library in {REPO}; build first (e.g. make -j4 LLAMA_METAL=1 koboldcpp_default)")
        return 2

    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(143)))
    ctx = Ctx(args)
    t0 = time.time()
    try:
        for m in models:
            path = Path(m).expanduser()
            if not path.exists():
                ctx.add(path.stem, "(model)", "SKIP", f"not found: {path}")
                continue
            run_model(ctx, path)
    except KeyboardInterrupt:
        print("\ninterrupted")
        cleanup()
        return 130
    finally:
        cleanup()

    # summary table
    print(f"\n{'model':<30} {'check':<12} {'result':<6} {'time':>6}  detail")
    print("-" * 110)
    for r in ctx.results:
        t = f"{r.seconds:5.1f}s" if r.seconds else ""
        print(f"{r.model[:30]:<30} {r.check:<12} {r.status:<6} {t:>6}  {r.detail[:120]}")
    n = {k: sum(r.status == k for r in ctx.results) for k in ("PASS", "FAIL", "SKIP")}
    print("-" * 110)
    print(f"{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped in {time.time() - t0:.0f}s")
    tested = [r for r in ctx.results if r.status != "SKIP"]
    if n["FAIL"] or args.keep_logs:
        print(f"server logs: {args.log_dir}")
    if not tested:
        print("nothing was tested (no models found?)")
        return 1
    if not n["FAIL"] and not args.keep_logs:
        shutil.rmtree(args.log_dir, ignore_errors=True)
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
