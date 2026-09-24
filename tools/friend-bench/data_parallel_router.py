#!/usr/bin/env python3
"""Run several friend.cpp workers behind one OpenAI-compatible endpoint.

Each worker owns a complete model context, so this provides real data parallelism
when one process is saturating a device or when several devices are available.
Requests with the same ``cache_salt`` and adapter profile are sticky to one
replica; unlabelled requests use the least-busy replica. Profile stickiness is
what makes LoRA/steering lanes safe: those adapters are context-local state.
"""

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]


class Pool:
    def __init__(self, ports):
        self.ports = list(ports)
        self.active = [0] * len(self.ports)
        self.failure_until = [0.0] * len(self.ports)
        self.failure_count = [0] * len(self.ports)
        self.observed_running = [0] * len(self.ports)
        self.observed_waiting = [0] * len(self.ports)
        self.observed_kv_blocks = [0] * len(self.ports)
        self.observed_at = [0.0] * len(self.ports)
        self.observed_latency_ms = [0.0] * len(self.ports)
        self.draining = [False] * len(self.ports)
        self.lock = threading.Lock()
        self.epoch = uuid.uuid4().hex[:12]
        self._health_stop = threading.Event()
        self._health_thread = None

    def start_health_monitor(self, interval=0.5):
        if self._health_thread and self._health_thread.is_alive():
            return
        self._health_stop.clear()
        def run():
            while not self._health_stop.wait(interval):
                for index, port in enumerate(self.ports):
                    self.refresh_worker(index, port)
        self._health_thread = threading.Thread(target=run, name="friend-router-health", daemon=True)
        self._health_thread.start()

    def stop_health_monitor(self):
        self._health_stop.set()
        if self._health_thread:
            self._health_thread.join(timeout=2)

    def refresh_worker(self, index, port=None):
        """Refresh bounded scheduler state used by the next routing decision."""
        port = self.ports[index] if port is None else port
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
        started = time.monotonic()
        try:
            conn.request('GET', '/api/extra/requests')
            response = conn.getresponse()
            if response.status != 200:
                raise ValueError(f'worker state probe returned {response.status}')
            entries = json.loads(response.read())
            running = sum(item.get('state') in ('prefill', 'generating') for item in entries)
            waiting = sum(item.get('state') == 'waiting' for item in entries)
            kv_blocks = sum(int(item.get('kv_blocks', 0) or 0) for item in entries)
            latency_ms = (time.monotonic() - started) * 1000.0
            with self.lock:
                self.observed_running[index] = running
                self.observed_waiting[index] = waiting
                self.observed_kv_blocks[index] = kv_blocks
                self.observed_at[index] = time.monotonic()
                previous = self.observed_latency_ms[index]
                self.observed_latency_ms[index] = latency_ms if previous <= 0 else (0.25 * latency_ms + 0.75 * previous)
            self.mark_success(index)
            return True
        except (OSError, http.client.HTTPException, ValueError, TypeError, KeyError):
            self.mark_failure(index)
            return False
        finally:
            conn.close()

    def set_observed(self, index, running=0, waiting=0, kv_blocks=0, latency_ms=0.0):
        """Inject a state sample for deterministic tests and embedded callers."""
        with self.lock:
            self.observed_running[index] = max(0, int(running))
            self.observed_waiting[index] = max(0, int(waiting))
            self.observed_kv_blocks[index] = max(0, int(kv_blocks))
            self.observed_at[index] = time.monotonic()
            self.observed_latency_ms[index] = max(0.0, float(latency_ms))

    def set_draining(self, index, draining=True):
        """Stop assigning new requests while preserving existing ownership."""
        with self.lock:
            self.draining[index] = bool(draining)

    def choose(self, body, exclude=()):
        excluded = set(exclude)
        sticky = None
        try:
            value = json.loads(body.decode()) if body else {}
            sticky = value.get("cache_salt") or value.get("session_id")
            # LoRA, steering vectors and heads mutate context-local state. Keep
            # each execution profile on one replica so its KV and adapter state
            # never cross a lane boundary. The canonical JSON also makes routing
            # stable across requests that use different key ordering.
            profile = {key: value[key] for key in ("lora", "steer", "cvec", "head") if key in value}
            if profile:
                profile_key = json.dumps(profile, sort_keys=True, separators=(",", ":"))
                sticky = f"{sticky or ''}|{profile_key}"
        except Exception:
            pass
        with self.lock:
            # Draining is an operator decision, not a transient health failure:
            # never bring a drained worker back via the cooldown probe fallback.
            available = [i for i in range(len(self.ports))
                         if i not in excluded and not self.draining[i]]
            if not available:
                raise ValueError("no accepting untried replicas")
            now = time.monotonic()
            healthy = [i for i in available if self.failure_until[i] <= now]
            # Keep probing when every replica is cooling down. This prevents a
            # transient outage from becoming a permanent routing blackout.
            candidates = healthy or available
            def score(i):
                observed = 0.0
                if now - self.observed_at[i] <= 2.0:
                    observed = (self.observed_running[i] +
                                0.25 * self.observed_waiting[i] +
                                0.001 * self.observed_kv_blocks[i] +
                                0.01 * self.observed_latency_ms[i])
                return self.active[i] + observed
            if sticky is not None:
                digest = hashlib.sha256(str(sticky).encode()).digest()
                index = int.from_bytes(digest[:8], "big") % len(self.ports)
                if index not in candidates:
                    index = min(candidates, key=score)
            else:
                index = min(candidates, key=score)
            self.active[index] += 1
            return index, self.ports[index]

    def done(self, index):
        with self.lock:
            self.active[index] = max(0, self.active[index] - 1)

    def mark_failure(self, index):
        """Temporarily stop sending new work to a failed replica."""
        with self.lock:
            self.failure_count[index] += 1
            delay = min(30.0, 0.25 * (2 ** min(self.failure_count[index] - 1, 7)))
            self.failure_until[index] = time.monotonic() + delay

    def mark_success(self, index):
        with self.lock:
            self.failure_count[index] = 0
            self.failure_until[index] = 0.0

    def request_id(self, index, local_id):
        return f"{self.epoch}:{index}:{local_id}"

    def owner(self, request_id):
        epoch, index, local_id = str(request_id).split(":")
        index, local_id = int(index), int(local_id)
        if epoch != self.epoch or not 0 <= index < len(self.ports) or local_id < 1:
            raise ValueError("unknown request owner")
        return index, local_id


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pool = None
    timeout = 300

    def do_GET(self):
        if self.path == '/api/extra/requests':
            self.list_requests()
            return
        self.forward(b"")

    def do_POST(self):
        # Reject ambiguous framing instead of leaving unread bytes on a reused
        # client connection. This proxy accepts length-delimited JSON requests.
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get('Transfer-Encoding') or not 0 <= length <= 32 * 1024 * 1024:
                raise ValueError()
        except ValueError:
            self.close_connection = True
            self.send_error(400, 'invalid request framing')
            return
        body = self.rfile.read(length)
        owner = None
        if self.path in ('/api/extra/requests/pause', '/api/extra/requests/resume', '/api/extra/requests/cancel'):
            try:
                value = json.loads(body)
                owner, local_id = self.pool.owner(value['id'])
                value['id'] = local_id
                body = json.dumps(value).encode()
            except (ValueError, KeyError, TypeError):
                self.send_error(400, 'use a request id from the router request list')
                return
        self.forward(body, owner)

    def upstream_headers(self):
        hop = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length'}
        hop.update(x.strip().lower() for x in self.headers.get('Connection', '').split(','))
        return {k: v for k, v in self.headers.items() if k.lower() not in hop}

    def list_requests(self):
        result = []
        alive = 0
        for index, port in enumerate(self.pool.ports):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=self.timeout)
            try:
                conn.request('GET', self.path, headers=self.upstream_headers())
                response = conn.getresponse()
                body = response.read()
                self.pool.mark_success(index)
                entries = json.loads(body) if response.status == 200 else []
                self.pool.set_observed(index,
                                       running=sum(item.get('state') in ('prefill', 'generating') for item in entries),
                                       waiting=sum(item.get('state') == 'waiting' for item in entries),
                                       kv_blocks=sum(int(item.get('kv_blocks', 0) or 0) for item in entries))
                if response.status != 200:
                    continue
                alive += 1
                for req in json.loads(body):
                    req['id'] = self.pool.request_id(index, req['id'])
                    req['worker'] = index
                    result.append(req)
            except (OSError, http.client.HTTPException, ValueError, TypeError, KeyError):
                # A dead replica should not hide the live workers' request
                # state. Controls still fail explicitly when their owner is
                # unavailable; the aggregate view remains useful for health.
                self.pool.mark_failure(index)
                continue
            finally:
                conn.close()
        body = json.dumps(result).encode()
        self.send_response(200 if alive else 503)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward(self, body, owner=None):
        route_started = time.monotonic()
        attempted = set()
        for _ in range(len(self.pool.ports)):
            if len(attempted) >= len(self.pool.ports):
                break
            if owner is None:
                try:
                    index, port = self.pool.choose(body, attempted)
                except ValueError:
                    self.close_connection = True
                    self.send_error(503, 'no accepting replicas')
                    return
            else:
                index, port = owner, self.pool.ports[owner]
                with self.pool.lock:
                    self.pool.active[index] += 1
            attempted.add(index)
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=self.timeout)
            connected = False
            headers_sent = False
            try:
                # Fail over only before sending any request bytes. A timeout
                # after request() may mean generation has already started.
                conn.connect()
                connected = True
                upstream_started = time.monotonic()
                headers = self.upstream_headers()
                if body:
                    headers["Content-Length"] = str(len(body))
                conn.request(self.command, self.path, body=body or None, headers=headers)
                response = conn.getresponse()
                self.send_response(response.status, response.reason)
                hop = {'connection', 'keep-alive', 'transfer-encoding', 'trailer', 'upgrade'}
                hop.update(x.strip().lower() for x in response.getheader('Connection', '').split(','))
                for key, value in response.getheaders():
                    if key.lower() not in hop:
                        self.send_header(key, value)
                # These headers make the worker boundary visible to clients
                # collecting per-request timings. The epoch changes whenever
                # the router starts, so a route ID cannot be mistaken for a
                # request owner after a restart. Header latency ends when the
                # upstream sends response headers; streaming body time remains
                # represented by the response's own timing fields.
                self.send_header('X-Friend-Router-Epoch', self.pool.epoch)
                self.send_header('X-Friend-Router-Worker', str(index))
                self.send_header('X-Friend-Router-Queue-Ms',
                                 f'{max(0.0, (upstream_started - route_started) * 1000.0):.3f}')
                self.send_header('X-Friend-Router-Upstream-Header-Ms',
                                 f'{max(0.0, (time.monotonic() - upstream_started) * 1000.0):.3f}')
                self.send_header('Connection', 'close')
                self.end_headers()
                headers_sent = True
                self.close_connection = True
                # read(n) waits to fill n bytes on non-chunked SSE responses;
                # read1 returns the available data so each event flushes promptly.
                while chunk := response.read1(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                # A return in the try suite skips its else suite. Clear the
                # cooldown here, after the upstream response has been consumed.
                if response.status < 500:
                    self.pool.mark_success(index)
                else:
                    self.pool.mark_failure(index)
                return
            except (OSError, http.client.HTTPException) as exc:
                if not connected and owner is None and len(attempted) < len(self.pool.ports):
                    self.pool.mark_failure(index)
                    continue
                if not connected:
                    self.pool.mark_failure(index)
                self.close_connection = True
                if not headers_sent:
                    self.send_error(502, f'replica {port} unavailable: {exc}')
                return
            finally:
                conn.close()
                self.pool.done(index)
        self.close_connection = True
        self.send_error(502, 'all replicas unavailable')

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--first-replica-port", type=int, default=5101)
    parser.add_argument("--replica-arg", action="append", default=[])
    args = parser.parse_args()
    if args.replicas < 1:
        parser.error("--replicas must be positive")
    ports = [args.first_replica_port + i for i in range(args.replicas)]
    children = []
    common = [sys.executable, str(ROOT / "koboldcpp.py"), "--model", args.model,
              "--skiplauncher", "--quiet"]
    for port in ports:
        children.append(subprocess.Popen(common + ["--port", str(port)] + args.replica_arg, cwd=ROOT))
    pool = Pool(ports)
    Handler.pool = pool
    pool.start_health_monitor()
    server = ThreadingHTTPServer(("", args.port), Handler)
    print(f"friend.cpp data-parallel router on :{args.port}; replicas: {', '.join(map(str, ports))}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pool.stop_health_monitor()
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
