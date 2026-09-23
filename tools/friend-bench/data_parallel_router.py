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
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]


class Pool:
    def __init__(self, ports):
        self.ports = list(ports)
        self.active = [0] * len(self.ports)
        self.lock = threading.Lock()
        self.epoch = uuid.uuid4().hex[:12]

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
            available = [i for i in range(len(self.ports)) if i not in excluded]
            if not available:
                raise ValueError("no untried replicas")
            if sticky is not None:
                digest = hashlib.sha256(str(sticky).encode()).digest()
                index = int.from_bytes(digest[:8], "big") % len(self.ports)
                if index in excluded:
                    index = min(available, key=lambda i: self.active[i])
            else:
                index = min(available, key=lambda i: self.active[i])
            self.active[index] += 1
            return index, self.ports[index]

    def done(self, index):
        with self.lock:
            self.active[index] = max(0, self.active[index] - 1)

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
        if self.path in ('/api/extra/requests/pause', '/api/extra/requests/resume'):
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
        attempted = set()
        for _ in range(len(self.pool.ports)):
            if len(attempted) >= len(self.pool.ports):
                break
            if owner is None:
                index, port = self.pool.choose(body, attempted)
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
                self.send_header('Connection', 'close')
                self.end_headers()
                headers_sent = True
                self.close_connection = True
                # read(n) waits to fill n bytes on non-chunked SSE responses;
                # read1 returns the available data so each event flushes promptly.
                while chunk := response.read1(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            except (OSError, http.client.HTTPException) as exc:
                if not connected and owner is None and len(attempted) < len(self.pool.ports):
                    continue
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
    server = ThreadingHTTPServer(("", args.port), Handler)
    print(f"friend.cpp data-parallel router on :{args.port}; replicas: {', '.join(map(str, ports))}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
