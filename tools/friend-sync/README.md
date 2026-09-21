# friend-sync

Syncing friend.cpp with its two upstreams used to take an afternoon. These tools turn it into one command plus whatever conflicts actually need a human.

- `sync.sh` (a wrapper around `sync.py`) fetches, merges, applies the mechanical rules, checks the known traps, builds, and runs the smoke tests.
- `smoke.py` starts real servers against small models and checks the friend.cpp features end to end. You can run it on its own after any change.

Both are stdlib Python 3. Neither ever pushes.

## Sync

```bash
tools/friend-sync/sync.sh --status     # fetch both remotes, show how far behind concedo is, do nothing else
tools/friend-sync/sync.sh              # the whole thing
tools/friend-sync/sync.sh --continue   # after resolving conflicts / fixing a build or smoke failure
tools/friend-sync/sync.sh --abort      # abort the merge in progress and switch back
```

What a full run does:

1. Fetches `upstream/concedo` (koboldcpp) and `prismml/prism` and prints the commit counts. If there's nothing new, it stops there.
2. Creates `sync/<yyyy-mm-dd>` from `concedo` (it adds `.2`, `.3` if that name is taken, or use `--branch`).
3. Merges koboldcpp first, then Prism. Each merge is committed separately. The commit message lists what the rules did and which files were resolved by hand, so add your resolution notes to it.
4. Stops at the first content conflict. It prints the conflicted files and the resolution notes from previous syncs for those files: blue-noise in the samplers, DFly on the fused DFlash encoder, Metal GDN fusions, and so on. It never resolves one itself. Resolve, `git add`, then `--continue`.
5. After both merges:
   - checks that every `src/models/*.cpp` with real code is `#include`d in `src/llama-model.cpp`, since kobold unity-builds the models;
   - checks that the top-level `kernels/` (a Makefile build artifact) isn't staged or committed;
   - builds;
   - runs `smoke.py`.
6. Prints the next steps. Merging into `concedo` and pushing are yours to do.

State is kept in `<git-dir>/friend-sync.json`, so `--continue` works from a fresh shell. The merge targets are pinned to the SHAs fetched at the start.

### Mechanical rules (applied automatically)

- **modify/delete where our side deleted the file**: the file stays deleted if it's a llama.cpp test, doc, CI, CMake or example file (`tests/`, `docs/`, `.github/`, `ci/`, `cmake/`, `examples/`, `CMakeLists.txt`, `*.cmake`, `tools/*`, `AGENTS.md`, `CONTRIBUTING.md` and similar). It also stays deleted if its whole directory is gone on our side, which covers the ggml-hexagon/opencl/sycl/webgpu backends that kobold drops.
- **Prism merge only**: files that Prism *adds* under `tests/`, `docs/`, `.github/`, `examples/` or `tools/` are dropped.
- **Never touched by either rule**: `tools/server`, `tools/ui`, `tools/mtmd` and `tools/friend-*`. Kobold compiles mtmd, and server/ui carry our API fields. Everything under `src/`, `ggml/`, `common/` and friends is always left to a normal merge.

What got dropped is printed, grouped by directory, and listed in the merge commit. Library files that only a dropped tool used (for example `common/kv-mean-center.*` last time) are *not* detected. They're inert because the Makefile doesn't build them, but delete them if you care.

### Options

| flag | |
|---|---|
| `--status` / `--dry-run` | fetch and report only |
| `--only upstream` / `--only prismml` | merge just one source |
| `--at upstream=<commit>` | merge a source only up to a commit/tag instead of the branch tip |
| `--base BRANCH` | sync a branch other than `concedo` |
| `--branch NAME` | name the sync branch yourself |
| `--no-fetch` | use the remote-tracking refs as they are |
| `--no-build`, `--no-smoke` | skip those stages |
| `--clean-build` | `make clean` first. Try this if you see link errors from stale objects; the kcpp mappers have bitten before |
| `-j N` | make jobs (default 4, or `FRIEND_SYNC_JOBS`) |
| `--smoke-args '...'` | passed to smoke.py, e.g. `'--skip batch'` |

The build command depends on the platform:

- macOS: `make -j4 LLAMA_METAL=1 koboldcpp_default`
- Linux with `nvcc` or `CUDA_PATH`: `make -j4 LLAMA_CUBLAS=1 koboldcpp_cublas`, and the smoke servers get `--usecuda`
- anything else: `koboldcpp_default`

To use something else, set `FRIEND_SYNC_MAKE` to the full command. On a shared desktop, a memory-capped version is kinder:

```bash
FRIEND_SYNC_MAKE="systemd-run --user --scope -p MemoryMax=6G nice make -j3 LLAMA_CUBLAS=1 LLAMA_CUDA_CCBIN=... koboldcpp_cublas" \
FRIEND_SMOKE_FLAGS=--usecuda tools/friend-sync/sync.sh
```

## Smoke tests

```bash
python3 tools/friend-sync/smoke.py                                  # the default models that exist
python3 tools/friend-sync/smoke.py --models ~/models/Bonsai-1.7B-Q1_0.gguf
python3 tools/friend-sync/smoke.py --only cache,batch --keep-logs /tmp/smoke
FRIEND_SMOKE_MODELS=~/models/a.gguf,~/models/b.gguf python3 tools/friend-sync/smoke.py
```

Default models (missing ones are skipped, not failed):

- `~/models/Bonsai-1.7B-Q1_0.gguf`: Q1_0, the upstream type
- `~/models/Ternary-Bonsai-1.7B-PQ2_0.gguf`: PQ2_0, the Prism-only type
- `~/models/Qwen3.5-0.8B-Q4_0.gguf`: hybrid recurrent/attention

It uses ports 5090-5095, whichever are free.

| check | what passes |
|---|---|
| `coherent` | sampled output is non-empty, printable and not degenerate |
| `determinism` | same seed twice gives the same text |
| `blue_noise` | `blue_noise: true` generates, is seed-deterministic, and `rng_type: lowbias32` works |
| `head` | the model's own head, extracted with `tools/friend-heads/extract_head.py` and loaded as `--head-pool self=...`, reproduces base greedy output exactly. Base output is unchanged afterwards, and an unknown head errors |
| `cache` | persona A, persona B, A again: the log shows `[Prompt cache: reusing`, and A's output equals a fresh `--cache-ram 0` server. This server runs without `--quiet`, since `--quiet` hides that line |
| `steer` | `/api/extra/steer/build` returns ok, `"steer"` changes greedy output, and output returns to base without it |
| `gpu_cpu` | 16 greedy tokens agree between the GPU and `--gpulayers 0` |
| `batch` | `--parallelrequests 4 --noshift`: mixed requests (base, persona, sampled, head, steer) fired concurrently match the same requests run one at a time, and they really went through the batch worker (`BatchRequest:` in the log) |

A run takes about 70 s for all three models on an M5, with up to four server launches per model. It exits non-zero if anything fails, and prints a summary table plus the path to the server logs. Extracted heads are cached in `~/.cache/friend-smoke`. The `head` check needs `uv` and is skipped without it.

If a check fails after a sync, read its server log before touching the test. Every check here encodes a guarantee FRIEND.md makes.
