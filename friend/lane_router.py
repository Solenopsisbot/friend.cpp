#!/usr/bin/env python3
"""Route requests to head-affine KoboldCpp workers.

Head-bearing requests stay on one worker so its KV cache remains hot. Requests
without a head use lane zero, preserving ordinary continuous batching.
"""
import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKENDS = [int(p) for p in os.environ.get("FRIEND_LANE_PORTS", "9201,9202,9203,9204").split(",")]

class Router(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self): self.forward()
    def do_POST(self): self.forward()
    def do_PUT(self): self.forward()
    def do_DELETE(self): self.forward()

    def forward(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        lane = 0
        if self.path.startswith("/v1/chat/completions") and body:
            try:
                head = json.loads(body).get("head")
                if head:
                    lane = 1 + (sum(ord(c) for c in str(head)) % max(1, len(BACKENDS) - 1))
            except (ValueError, TypeError):
                pass
        lane = min(lane, len(BACKENDS) - 1)
        conn = http.client.HTTPConnection("127.0.0.1", BACKENDS[lane], timeout=600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}}
        headers["Content-Length"] = str(len(body))
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            payload = response.read() if response.length is None else None
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in {"connection", "transfer-encoding", "content-length"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", len(payload) if payload is not None else response.length)
            self.end_headers()
            if payload is not None:
                self.wfile.write(payload)
            else:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as exc:
            self.send_error(502, str(exc))
        finally:
            conn.close()

    def log_message(self, fmt, *args):
        return

if __name__ == "__main__":
    port = int(os.environ.get("FRIEND_ROUTER_PORT", "9190"))
    ThreadingHTTPServer(("0.0.0.0", port), Router).serve_forever()
