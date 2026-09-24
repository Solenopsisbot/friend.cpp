#!/usr/bin/env python3
"""Exercise real Python fan-out and native submission without loading weights."""
import asyncio
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import koboldcpp as server


class NativeBackend:
    def __init__(self, count, reject=None):
        self.ready = threading.Barrier(count, timeout=5)
        self.reject = reject
        self.submitted = {}
        self.released = set()
        self.lock = threading.Lock()

    def batch_generate_submit(self, inputs):
        if self.reject is not None:
            return self.reject
        with self.lock:
            request_id = len(self.submitted)
            self.submitted[request_id] = (inputs.seed, threading.get_ident())
            return request_id

    def batch_generate_result(self, request_id):
        # No request may finish until every child has been submitted. The old
        # gather-of-blocking-coroutines path breaks this barrier deterministically.
        self.ready.wait()
        seed, _ = self.submitted[request_id]
        return SimpleNamespace(status=1, stopreason=seed % 2,
                               text=f"sample-{seed}".encode(), prompt_tokens=7,
                               completion_tokens=seed - 99,
                               logprobs_json=json.dumps({"prompt": [], "completion": []}).encode(),
                               timing_json=b'{"total_seconds": 1}')

    def batch_generate_release(self, request_id):
        with self.lock:
            self.released.add(request_id)

    def generate(self, inputs):
        raise AssertionError("parallel child entered the singleton legacy generator")


class ParallelSamplingTests(unittest.TestCase):
    def setUp(self):
        self.handler = object.__new__(server.KcppServerRequestHandler)
        self.args = SimpleNamespace(parallelrequests=4, noshift=True,
                                    defaultgenamt=8, genlimit=0)
        self.params = {"prompt": "Hello", "n": 3, "seed": 100, "max_length": 8}

    def run_request(self, backend, api_format=4):
        with patch.object(server, "args", self.args), \
             patch.object(server, "handle", backend), \
             patch.object(server, "utfprint"):
            return asyncio.run(self.handler.generate_text(self.params, api_format, False))

    def test_native_samples_overlap_and_preserve_order(self):
        for api_format in (3, 4):
            with self.subTest(api_format=api_format):
                backend = NativeBackend(3)
                result = self.run_request(backend, api_format)
                self.assertEqual(len({thread for _, thread in backend.submitted.values()}), 3)
                self.assertEqual(backend.released, set(backend.submitted))
                self.assertEqual([c["index"] for c in result["choices"]], [0, 1, 2])
                texts = [c["text"] if api_format == 3 else c["message"]["content"]
                         for c in result["choices"]]
                self.assertEqual(texts, ["sample-100", "sample-101", "sample-102"])
                self.assertEqual([c["finish_reason"] for c in result["choices"]],
                                 ["length", "stop", "length"])
                self.assertEqual(result["usage"], {"prompt_tokens": 7,
                                                  "completion_tokens": 6,
                                                  "total_tokens": 13})

    def test_native_rejection_cannot_fall_back(self):
        for code in (-1, -2):
            with self.subTest(code=code):
                result = self.run_request(NativeBackend(3, reject=code))
                self.assertEqual(result["error"]["code"], 503)

    def test_legacy_samples_stay_on_calling_thread(self):
        self.args.parallelrequests = 1
        caller = threading.get_ident()
        threads = []

        def legacy_generate(genparams, stream_flag):
            threads.append(threading.get_ident())
            self.assertFalse(genparams["_parallel_sample_native"])
            return {"text": "legacy", "prompt_tokens": 7, "completion_tokens": 1,
                    "stopreason": 1}

        with patch.object(server, "generate", side_effect=legacy_generate):
            result = self.run_request(None)
        self.assertEqual(threads, [caller] * 3)
        self.assertEqual(len(result["choices"]), 3)

    def test_beam_request_is_explicitly_rejected(self):
        with patch.object(server, "args", self.args), patch.object(server, "utfprint"):
            result = asyncio.run(self.handler.generate_text(
                {"prompt": "Hello", "use_beam_search": True}, 4, False))
        self.assertEqual(result["error"]["code"], 400)


if __name__ == "__main__":
    unittest.main()
