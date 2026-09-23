#!/usr/bin/env python3
"""Real-server regression checks for n-gram rollback, pause/resume and cache salts.

Usage: python3 tools/friend-bench/serving_regression.py --model ~/models/Bonsai-1.7B-Q1_0.gguf
Runs serial reference and speculative servers sequentially on a free local port.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def run(model, draft, suffix=False, profile_lanes=1, max_queued_requests=0):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def request(path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(base + path, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            body = response.read().decode()
            return body if path == '/metrics' else json.loads(body)

    def request_raw(path, data):
        req = urllib.request.Request(base + path, data=json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.read().decode()

    def metric(name):
        return float(next(line.split()[1] for line in request('/metrics').splitlines() if line.startswith(name + ' ')))

    with tempfile.TemporaryFile(mode='w+') as log:
        draft_flag = '--suffix-draft' if suffix else '--ngram-draft'
        slots = max(4, profile_lanes)
        command = [sys.executable, str(ROOT / 'koboldcpp.py'), '--model', str(model),
            '--port', str(port), '--contextsize', '4096', '--gpulayers', '99', '--skiplauncher',
            '--quiet', '--parallelrequests', str(slots), '--prefill-tokens', '32', '--schedule-tokens', '64',
            '--profile-lanes', str(profile_lanes), '--max-queued-requests', str(max_queued_requests),
            draft_flag, str(draft)]
        proc = subprocess.Popen(command,
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while True:
                try:
                    request('/api/extra/requests')
                    break
                except (OSError, ValueError):
                    if proc.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('server failed to start')
                    time.sleep(.1)
            lane_metrics = request('/metrics')
            for lane in range(profile_lanes):
                assert f'friend_batch_lane_running{{lane="{lane}"}} ' in lane_metrics, \
                    f'lane {lane} metric missing'
            assert metric('friend_batch_kv_pages_capacity') > 0, 'physical KV page capacity missing'
            prompt = 'Continue this repeating sequence without explanation: ' + 'alpha beta gamma delta ' * 80
            payload = dict(prompt=prompt, max_length=160, max_context_length=4096, temperature=0,
                           rep_pen=1, top_k=0, top_p=1, seed=123, cache_salt='test-A',
                           grammar='root ::= "alpha beta gamma delta " root')
            result_obj = request('/api/v1/generate', payload)['results'][0]
            result = result_obj['text']
            assert metric('friend_batch_kv_page_queries_total') > 0, 'physical KV page queries missing'
            timing = result_obj.get('timing')
            assert timing and timing['total_seconds'] >= timing['decode_seconds'] >= 0, timing
            assert timing['queue_seconds'] >= 0 and timing['prefill_seconds'] >= 0, timing
            lp = request('/api/v1/generate', dict(payload, max_length=8, logprobs=3, prompt_logprobs=3, cache_salt='logprobs'))['results'][0]
            assert isinstance(lp.get('logprobs'), list) and lp['logprobs'], 'batch completion logprobs missing'
            assert isinstance(lp.get('prompt_logprobs'), list) and lp['prompt_logprobs'], 'batch prompt logprobs missing'
            assert len(lp['logprobs'][0]['top_logprobs']) == 3, 'batch logprobs top-k mismatch'
            chat_lp = request('/v1/chat/completions', {
                'messages': [{'role': 'user', 'content': 'Answer with one short word: hello'}],
                'max_tokens': 4, 'temperature': 0, 'logprobs': True, 'top_logprobs': 2,
                'prompt_logprobs': 2, 'cache_salt': 'chat-logprobs', 'stream': False,
            })
            chat_choice = chat_lp['choices'][0]
            assert isinstance(chat_choice.get('logprobs'), dict), 'chat logprobs missing'
            assert chat_choice['logprobs'].get('content'), 'chat completion logprobs missing'
            assert chat_choice.get('prompt_logprobs'), 'chat prompt logprobs missing'
            stream_payload = {'messages': [{'role': 'user', 'content': 'Answer with six short words.'}],
                              'max_tokens': 6, 'temperature': 0, 'stream': True,
                              'stream_interval': 3, 'cache_salt': 'stream-interval'}
            stream_body = request_raw('/v1/chat/completions', stream_payload)
            stream_events = [line for line in stream_body.splitlines() if line.startswith('data: {')]
            assert stream_body.rstrip().endswith('data: [DONE]'), 'stream did not finish'
            assert 1 <= len(stream_events) <= 4, f'unexpected stream event count: {len(stream_events)}'
            if max_queued_requests:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    long_future = pool.submit(request, '/api/v1/generate',
                                              dict(payload, max_length=96, cache_salt='queue-cap'))
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        active = [r for r in request('/api/extra/requests')
                                  if r['state'] in ('prefill', 'generating')]
                        if active:
                            break
                        time.sleep(.01)
                    overloaded = request('/api/v1/generate',
                                         dict(payload, max_length=4, cache_salt='queue-cap-2'))
                    assert overloaded.get('error', {}).get('code') == 503, overloaded
                    long_future.result(timeout=120)
            # Different namespaces must not reuse live/retained KV, including base profiles.
            before = metric('friend_batch_reused_tokens_total')
            other = dict(payload, cache_salt='test-B', max_length=4)
            request('/api/v1/generate', other)
            assert metric('friend_batch_reused_tokens_total') == before, 'cross-salt reuse'
            request('/api/v1/generate', other)
            assert metric('friend_batch_reused_tokens_total') > before, 'same-salt reuse missing'
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(request, '/api/v1/generate', payload)
                deadline = time.monotonic() + 10
                target = None
                while time.monotonic() < deadline:
                    active = [r for r in request('/api/extra/requests') if r['state'] == 'generating']
                    if active:
                        target = active[0]['id']
                        break
                    time.sleep(.005)
                assert target is not None, 'no active generation found'
                offload_before = metric('friend_batch_offloaded_bytes')
                assert request('/api/extra/requests/pause', {'id': target})['accepted']
                while time.monotonic() < deadline:
                    state = next(r for r in request('/api/extra/requests') if r['id'] == target)
                    if state['state'] == 'paused':
                        break
                    time.sleep(.005)
                assert state['state'] == 'paused', state
                assert state['paused_bytes'] > 0, state
                assert metric('friend_batch_offloaded_bytes') > offload_before, 'offload metric missing'
                # Another request must make progress while the first has released its slot.
                request('/api/v1/generate', dict(other, cache_salt='test-C'))
                assert not future.done(), 'paused request completed'
                assert request('/api/extra/requests/resume', {'id': target})['accepted']
                resumed = future.result(timeout=120)['results'][0]['text']
                assert resumed == result, 'pause/resume changed output'
            # Four long, low-priority requests fill every sequence slot. A short
            # urgent request must be admitted by snapshotting one victim.
            if profile_lanes == 1 and not max_queued_requests:
                preempt_before = metric('friend_batch_preemptions_total')
                low_payload = dict(payload, max_length=128, grammar='', priority=100, bypass_eos_token=True,
                                   cache_salt='preempt-low')
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    low_futures = [pool.submit(request, '/api/v1/generate', dict(low_payload, seed=200 + i)) for i in range(4)]
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        active = [r for r in request('/api/extra/requests') if r['state'] == 'generating']
                        if len(active) >= 4:
                            break
                        time.sleep(.01)
                    urgent = request('/api/v1/generate', dict(payload, max_length=4, priority=-100,
                                                             cache_salt='preempt-urgent'))
                    assert urgent['results'][0]['text'], 'urgent request returned no output'
                    assert metric('friend_batch_preemptions_total') > preempt_before, 'priority preemption missing'
                    for future in low_futures:
                        future.result(timeout=120)
            if draft:
                proposed = metric('friend_batch_draft_proposed_tokens_total')
                accepted = metric('friend_batch_draft_accepted_tokens_total')
                print(f'  speculation observed: proposed={proposed:.0f}, accepted={accepted:.0f}', flush=True)
                assert proposed > 0 and accepted > 0, 'speculative path was not exercised'
            print(f'PASS draft={draft}, lanes={profile_lanes}: namespace isolation, reuse, pause/resume', flush=True)
            return result
        except BaseException:
            log.seek(0)
            print(log.read()[-10000:], file=sys.stderr)
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--profile-lanes', type=int, default=1)
    parser.add_argument('--max-queued-requests', type=int, default=0)
    args = parser.parse_args()
    if args.profile_lanes < 1 or args.profile_lanes > 32:
        parser.error('--profile-lanes must be between 1 and 32')
    reference = run(args.model.expanduser(), 0, profile_lanes=args.profile_lanes,
                    max_queued_requests=args.max_queued_requests)
    speculative = run(args.model.expanduser(), 4, suffix=True, profile_lanes=args.profile_lanes,
                      max_queued_requests=args.max_queued_requests)
    assert reference == speculative, 'speculation changed greedy output'
    print('PASS speculative output matches no-drafter reference')
