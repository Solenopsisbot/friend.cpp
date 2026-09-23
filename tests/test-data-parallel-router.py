import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "friend-bench"))
from data_parallel_router import Pool


def pick(pool, payload):
    index, _ = pool.choose(json.dumps(payload).encode())
    pool.done(index)
    return index


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
    namespaced = pool.request_id(2, 41)
    assert pool.owner(namespaced) == (2, 41)
