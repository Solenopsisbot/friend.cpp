import json
import socket
import sys
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "friend-bench"))
from data_parallel_router import Pool
from data_parallel_router import Handler


def pick(pool, payload):
    index, _ = pool.choose(json.dumps(payload).encode())
    pool.done(index)
    return index


class FakeWorker(BaseHTTPRequestHandler):
    """Small upstream used to test the proxy without loading a model."""

    worker_id = 0
    fail_connect = False
    pause_ids = []
    generation_ids = []

    def do_GET(self):
        if self.path == "/api/extra/requests":
            body = json.dumps([{"id": self.worker_id + 1, "state": "generating"}]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in (b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"):
            self.wfile.write(chunk)
            self.wfile.flush()
            time.sleep(0.01)
        self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get('drop_after_submit'):
            # The worker has consumed the request, but its response is lost.
            # Retrying here could execute a generation twice.
            self.generation_ids.append(self.worker_id)
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True
            return
        self.pause_ids.append((self.worker_id, body.get("id")))
        response = json.dumps({"accepted": True, "worker": self.worker_id,
                               "id": body.get("id")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        return


def start_fake(worker_id):
    class Worker(FakeWorker):
        pass
    Worker.worker_id = worker_id
    Worker.pause_ids = FakeWorker.pause_ids
    server = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def proxy_request(server, method, path, body=b""):
    conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Content-Length": str(len(body))} if method == "POST" else {}
    conn.request(method, path, body=body or None, headers=headers)
    return conn, conn.getresponse()


if __name__ == "__main__":
    pool = Pool([5101, 5102, 5103, 5104])
    profile = {"lora": {"rook": 0.8}, "head": "rook"}
    assert pick(pool, dict(profile, cache_salt="tenant-a")) == pick(pool, dict(profile, cache_salt="tenant-a"))
    assert pick(pool, {"head": "mira", "cache_salt": "tenant-a"}) in range(4)
    failed, _ = pool.choose(json.dumps(dict(profile, cache_salt="tenant-a")).encode())
    pool.done(failed)
    retry, _ = pool.choose(json.dumps(dict(profile, cache_salt="tenant-a")).encode(), {failed})
    assert retry != failed
    pool.done(retry)
    pool.mark_failure(0)
    cooled, _ = pool.choose(b"{}")
    assert cooled != 0
    pool.done(cooled)
    pool.mark_success(0)
    pool.set_observed(0, running=4, waiting=2, kv_blocks=512)
    pool.set_observed(1, running=0, waiting=0, kv_blocks=0)
    state_pick, _ = pool.choose(b"{}")
    assert state_pick == 1
    pool.done(state_pick)
    latency_pool = Pool([5101, 5102])
    latency_pool.set_observed(0, running=0, latency_ms=1000)
    latency_pool.set_observed(1, running=0, latency_ms=1)
    latency_pick, _ = latency_pool.choose(b"{}")
    assert latency_pick == 1
    latency_pool.done(latency_pick)
    latency_pool.set_draining(1, True)
    draining_pick, _ = latency_pool.choose(b"{}")
    assert draining_pick != 1
    latency_pool.done(draining_pick)
    latency_pool.set_draining(1, False)
    drained_pool = Pool([5101, 5102])
    drained_pool.set_draining(0)
    drained_pool.mark_failure(1)
    assert pick(drained_pool, {}) == 1, 'cooldown fallback resurrected a drained replica'
    drained_pool.set_draining(1)
    try:
        drained_pool.choose(b'{}')
        raise AssertionError('all-draining pool accepted new work')
    except ValueError:
        pass
    assert drained_pool.active == [0, 0]
    namespaced = pool.request_id(2, 41)
    assert pool.owner(namespaced) == (2, 41)

    # Exercise the actual HTTP proxy: streaming must flush each upstream
    # event, a dead replica may be skipped before request bytes are sent, and
    # pause controls must be routed back to the worker that owns the ID.
    worker_a, thread_a = start_fake(10)
    worker_b, thread_b = start_fake(20)
    router = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Handler.pool = Pool([worker_a.server_port, worker_b.server_port])
    Handler.pool.start_health_monitor(interval=0.01)
    router_thread = threading.Thread(target=router.serve_forever, daemon=True)
    router_thread.start()
    try:
        conn, response = proxy_request(router, "GET", "/stream")
        assert response.status == 200
        assert response.read(11) == b"data: one\n\n"
        assert response.read(11) == b"data: two\n\n"
        assert response.read(len(b"data: [DONE]\n\n")) == b"data: [DONE]\n\n"
        conn.close()

        # A connection failure before request bytes are sent is safe to retry
        # on another replica. Port 1 is deliberately closed in the test
        # environment; the healthy worker must receive the request.
        Handler.pool.stop_health_monitor()
        Handler.pool = Pool([1, worker_b.server_port])
        conn, response = proxy_request(router, "GET", "/stream")
        assert response.status == 200
        assert response.read().endswith(b"[DONE]\n\n")
        conn.close()

        owner_id = Handler.pool.request_id(1, 77)
        conn, response = proxy_request(router, "POST", "/api/extra/requests/pause",
                                       json.dumps({"id": owner_id}).encode())
        assert response.status == 200
        assert json.loads(response.read())["worker"] == 20
        conn.close()
        assert FakeWorker.pause_ids[-1] == (20, 77)
        conn, response = proxy_request(router, "POST", "/api/extra/requests/cancel",
                                       json.dumps({"id": owner_id}).encode())
        assert response.status == 200
        assert json.loads(response.read())["worker"] == 20
        conn.close()
        assert FakeWorker.pause_ids[-1] == (20, 77)
        assert all(value == 0 for value in Handler.pool.active)

        # Owner controls keep working through drain while new requests receive
        # a retryable response. Never reroute a control to a different owner.
        Handler.pool.set_draining(0)
        Handler.pool.set_draining(1)
        conn, response = proxy_request(router, 'POST', '/api/extra/requests/cancel',
                                       json.dumps({'id': owner_id}).encode())
        assert response.status == 200
        assert json.loads(response.read())['worker'] == 20
        conn.close()
        conn, response = proxy_request(router, 'POST', '/v1/completions', b'{}')
        assert response.status == 503
        response.read()
        conn.close()

        Handler.pool = Pool([worker_b.server_port])
        Handler.pool.mark_failure(0)
        conn, response = proxy_request(router, 'GET', '/stream')
        assert response.read().endswith(b'[DONE]\n\n')
        conn.close()
        # Response delivery and the handler's finally block can race the client.
        deadline = time.monotonic() + 2
        while Handler.pool.active[0] and time.monotonic() < deadline:
            time.sleep(.001)
        assert Handler.pool.failure_count == [0], 'successful response did not clear cooldown'
        assert Handler.pool.failure_until == [0.0]

        Handler.pool = Pool([worker_a.server_port, worker_b.server_port])
        FakeWorker.generation_ids.clear()
        conn, response = proxy_request(router, 'POST', '/v1/completions',
                                       b'{"drop_after_submit": true}')
        assert response.status == 502
        response.read()
        conn.close()
        assert FakeWorker.generation_ids == [10], 'router replayed a submitted generation'
    finally:
        Handler.pool.stop_health_monitor()
        router.shutdown()
        router.server_close()
        worker_a.shutdown()
        worker_a.server_close()
        worker_b.shutdown()
        worker_b.server_close()
