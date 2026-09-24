#!/usr/bin/env python3
"""Exercise real Python fan-out and native submission without loading weights."""
import asyncio
import ctypes
import json
from pathlib import Path
import sys
import threading
import time
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


class StreamingBackend:
    def __init__(self):
        self.next_id = 0
        self.tokens = {}
        self.released = []

    def batch_generate_submit(self, inputs):
        request_id = self.next_id
        self.next_id += 1
        self.tokens[request_id] = [ctypes.c_char_p(f"tok-{request_id}".encode())]
        return request_id

    def batch_generate_stream_count(self, request_id):
        return len(self.tokens[request_id])

    def batch_generate_new_token(self, request_id, index):
        return self.tokens[request_id][index]

    def batch_generate_result(self, request_id):
        time.sleep(0.03)
        return SimpleNamespace(status=1, stopreason=1, text=self.tokens[request_id][0].value,
                               prompt_tokens=3, completion_tokens=1,
                               logprobs_json=None, timing_json=b'{"total_seconds": 0.1}')

    def batch_generate_release(self, request_id):
        self.released.append(request_id)


class ParallelSamplingTests(unittest.TestCase):
    def setUp(self):
        self.handler = object.__new__(server.KcppServerRequestHandler)
        self.args = SimpleNamespace(parallelrequests=4, noshift=True,
                                    defaultgenamt=8, genlimit=0, quiet=True,
                                    smartcontext=False, draftmodel="", usemtp=False,
                                    enableguidance=False, model_param="")
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

    def test_per_request_speculative_budget_is_passed_to_native(self):
        seen = []
        class BudgetBackend(NativeBackend):
            def batch_generate_submit(self, inputs):
                seen.append(inputs.friend_draft_max)
                return super().batch_generate_submit(inputs)
        for params, expected in (({}, -1), ({'speculative_tokens': 0}, 0),
                                 ({'speculative_tokens': 3}, 3),
                                 ({'num_speculative_tokens': 2}, 2),
                                 ({'speculative_tokens': 100}, 32)):
            with self.subTest(params=params):
                seen.clear()
                self.params.update(params)
                self.run_request(BudgetBackend(3))
                self.assertEqual(seen, [expected] * 3)
                for key in params:
                    self.params.pop(key)

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

    def test_streaming_parallel_fan_in_has_indexed_chunks(self):
        backend = StreamingBackend()
        events = []
        handler = object.__new__(server.KcppServerRequestHandler)
        handler.send_response = lambda *_args: None
        handler.send_header = lambda *_args: None
        handler.end_headers = lambda **_kwargs: None

        async def capture(data):
            events.append(data)

        handler.send_oai_sse_event = capture
        params = {"prompt": "Hello", "n": 2, "seed": 100, "max_length": 1,
                  "oai_uniqueid": 77}
        with patch.object(server, "args", self.args), \
             patch.object(server, "handle", backend), \
             patch.object(server, "friendlymodelname", "test-model"), \
             patch.object(server, "autoswapmode", False):
            asyncio.run(handler.handle_parallel_sse_stream(params, 4))
        payloads = [json.loads(event) for event in events if event != "[DONE]"]
        token_chunks = [item for item in payloads if item.get("choices", [{}])[0].get("delta", {}).get("content")]
        assert [item["choices"][0]["index"] for item in token_chunks] == [0, 1]
        finish_chunks = [item for item in payloads if item.get("choices") and
                         item["choices"][0]["finish_reason"] == "stop"]
        assert len(finish_chunks) == 2
        assert events[-1] == "[DONE]"
        assert set(backend.released) == {0, 1}

    def stream_handler(self, events):
        handler = object.__new__(server.KcppServerRequestHandler)
        handler.send_response = lambda *_args: None
        handler.send_header = lambda *_args: None
        handler.end_headers = lambda **_kwargs: None

        async def capture(data):
            events.append(data)

        handler.send_oai_sse_event = capture
        return handler

    def test_stream_tail_and_split_unicode_are_drained_before_release(self):
        for api_format in (3, 4):
            with self.subTest(api_format=api_format):
                backend = StreamingBackend()
                events = []
                handler = self.stream_handler(events)

                async def generate(child, _api, _stream):
                    index = int(child['oai_uniqueid'].rsplit('-', 1)[1])
                    child['_batch_request_id'] = index
                    # The first poll sees an incomplete UTF-8 sequence; the
                    # remaining bytes arrive as the task completes during wait.
                    backend.tokens[index] = [ctypes.c_char_p(b'caf\xc3')]
                    await asyncio.sleep(0.03)
                    backend.tokens[index].append(ctypes.c_char_p(b'\xa9!'))
                    return {'choices': [{'finish_reason': 'stop'}]}

                handler.generate_text = generate
                with patch.object(server, 'args', self.args), patch.object(server, 'handle', backend):
                    asyncio.run(handler.handle_parallel_sse_stream(dict(self.params, n=2), api_format))
                output = {0: '', 1: ''}
                finished = set()
                for event in events:
                    if event == '[DONE]':
                        continue
                    choice = json.loads(event)['choices'][0]
                    index = choice['index']
                    self.assertNotIn(index, finished)
                    output[index] += choice.get('text', '') if api_format == 3 else choice.get('delta', {}).get('content', '')
                    if choice['finish_reason'] is not None:
                        finished.add(index)
                self.assertEqual(output, {0: 'café!', 1: 'café!'})
                self.assertEqual(finished, {0, 1})
                self.assertCountEqual(backend.released, [0, 1])

    def test_disconnect_aborts_late_admission_and_releases_children(self):
        class DelayedBackend(StreamingBackend):
            def __init__(self):
                super().__init__()
                self.lock = threading.Lock()
                self.entered = threading.Event()
                self.allow_late = threading.Event()
                self.aborted = set()
                self.stopped = [threading.Event(), threading.Event()]

            def batch_generate_submit(self, inputs):
                with self.lock:
                    index = self.next_id
                    self.next_id += 1
                if index == 1:
                    self.entered.set()
                    assert self.allow_late.wait(5), 'monitor never cancelled'
                self.tokens[index] = []
                return index

            def batch_generate_abort(self, index):
                self.aborted.add(index)
                self.stopped[index].set()

            def batch_generate_result(self, index):
                assert self.stopped[index].wait(5), 'native child was orphaned'
                return SimpleNamespace(status=1, stopreason=-1, text=b'', prompt_tokens=3,
                                       completion_tokens=0, logprobs_json=None, timing_json=None)

        backend = DelayedBackend()
        events = []
        handler = self.stream_handler(events)
        handler.connection = object()

        async def disconnect(cancel):
            while not backend.entered.is_set():
                await asyncio.sleep(0.001)
            cancel()
            backend.allow_late.set()

        handler.monitor_connection = disconnect
        with patch.object(server, 'args', self.args), patch.object(server, 'handle', backend):
            asyncio.run(handler.handle_parallel_sse_stream(dict(self.params, n=2), 4))
        self.assertEqual(backend.aborted, {0, 1})
        self.assertCountEqual(backend.released, [0, 1])
        self.assertEqual(events, [])

    def test_streaming_rejection_is_reported_without_legacy_fallback(self):
        backend = NativeBackend(2, reject=-2)
        events = []
        handler = self.stream_handler(events)
        with patch.object(server, 'args', self.args), patch.object(server, 'handle', backend):
            asyncio.run(handler.handle_parallel_sse_stream(dict(self.params, n=2), 4))
        self.assertEqual(json.loads(events[0])['error']['code'], 503)
        self.assertEqual(events[-1], '[DONE]')

    def test_streaming_ineligible_request_never_starts_children(self):
        self.args.parallelrequests = 1
        with patch.object(server, 'args', self.args), patch.object(server, 'handle', None):
            result = asyncio.run(self.handler.handle_parallel_sse_stream(self.params, 4))
        self.assertEqual(result['error']['code'], 400)

    def test_serving_capabilities_report_backend_boundaries(self):
        self.args.profile_lanes = 2
        self.args.disaggregated_prefill = True
        with patch.object(server, 'args', self.args):
            capabilities = server.get_friend_serving_capabilities()
        self.assertTrue(capabilities['continuous_batching'])
        self.assertTrue(capabilities['scheduler_paged_kv'])
        self.assertTrue(capabilities['kv_socket_transport'])
        self.assertTrue(capabilities['disaggregated_prefill'])
        self.assertFalse(capabilities['backend_paged_kv'])
        self.assertFalse(capabilities['device_kv_transport'])
        self.assertFalse(capabilities['expert_parallel'])
        self.assertFalse(capabilities['context_parallel'])


if __name__ == "__main__":
    unittest.main()
