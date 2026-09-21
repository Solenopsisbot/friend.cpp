#!/usr/bin/env python3
"""
friend.cpp upstream sync: fetch koboldcpp + PrismML, merge both into a dated branch,
apply the mechanical conflict rules we already learned the hard way, check the
known traps, build, and run the smoke tests. Never pushes.

    tools/friend-sync/sync.sh              # fetch, report, merge, check, build, smoke
    tools/friend-sync/sync.sh --status     # fetch + "how far behind are we", nothing else
    tools/friend-sync/sync.sh --continue   # after resolving a conflict by hand (or fixing a
                                           # build/smoke failure): carry on where it stopped
    tools/friend-sync/sync.sh --abort      # abandon an in-progress merge, back to where you were

Order: upstream/concedo (koboldcpp) first, then prismml/prism. Content conflicts are
NEVER auto-resolved; the run stops at the first one with a report and a reminder of
the resolution notes for the files involved.

Mechanical rules (applied automatically, and listed in the merge commit message):
  * modify/delete where our side deleted a llama.cpp test/doc/CI/CMake/example file
    (kobold strips those) -> keep it deleted.
  * PrismML merge only: files prism ADDS under tests/, docs/, .github/, examples/ or
    tools/ -> dropped, except tools/server, tools/ui and tools/mtmd (mtmd is compiled
    into koboldcpp; server/ui carry friend.cpp's API surface), and friend.cpp's own
    tools/friend-*.

State lives in <git-dir>/friend-sync.json, so --continue works from a new shell.
The merge targets are pinned to the SHAs fetched at the start; re-fetching mid-sync
does not move them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (label, remote, branch, human name). Order matters: kobold first, then prism on top.
SOURCES = [
    ("upstream", "upstream", "concedo", "upstream koboldcpp"),
    ("prismml", "prismml", "prism", "PrismML llama.cpp"),
]

# modify/delete conflicts where OUR side deleted the file: keep it deleted if it matches.
KEEP_DELETED = [
    "tests/*", "docs/*", ".github/*", "ci/*", "cmake/*", ".devops/*", "examples/*",
    "CMakeLists.txt", "*/CMakeLists.txt", "*.cmake", "CMakePresets.json",
    "tools/*",  # narrowed by TOOLS_KEEP below
    # llama.cpp's top-level contributor docs, which kobold doesn't carry
    "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md", "CODEOWNERS", "CLAUDE.md",
]
# Also kept deleted, whatever the path: a modify/delete whose parent directory has no
# tracked files left on our side, i.e. we removed the whole component (kobold drops the
# ggml-hexagon/opencl/sycl/webgpu backends, for instance).
# prism-added files under these get dropped...
DROP_ADDED = ["tests/*", "docs/*", ".github/*", "examples/*", "tools/*"]
# ...except under these (kept by both rules above)
TOOLS_KEEP = ["tools/server/*", "tools/ui/*", "tools/mtmd/*", "tools/friend-*"]

# Resolution notes shown when a conflict touches a matching path (from the 2026-09-21 sync).
HINTS = [
    (["gpttype_adapter.cpp", "src/llama-sampler.cpp", "common/sampling.*"],
     "blue-noise sampling: our polymorphic llama_dist_rng / KcppTokenRng must stay threaded "
     "through every sampling path; upstream's transactional backend sampling replays via "
     "clone() + nextf() on accept."),
    (["tools/server/*"],
     "request parsing lives in server-schema.cpp: blue_noise / rng_type (and adapter fields) "
     "are schema fields there."),
    (["tools/ui/*"], "blue_noise / rng_type settings live in the UI settings registry + chat store."),
    (["src/models/dflash.cpp", "src/models/dspark.cpp", "common/speculative*"],
     "DFly was ported onto upstream's fused DFlash encoder (build_dfly_context_fusion), NOT "
     "prism's n_embd_out widening. DSpark uses upstream's n_max ctor arg and pos0 naming, plus "
     "friend_rewind/flush/resync hooks for prompt reuse."),
    (["src/llama-model.cpp", "src/models/*"],
     "kobold unity-builds models: every src/models/*.cpp must be #included in "
     "src/llama-model.cpp (dspark.cpp was silently missed once)."),
    (["Makefile"],
     "keep the KCPP_ARCH_*_DEPS for kcpp-{quant,repack}mapper (they #include arch/*/ sources) "
     "and the CUDA_PATH search order; the Metal kernels copy into top-level kernels/ is a "
     "build artifact."),
    (["ggml/src/ggml-cpu/*repack*", "ggml/src/ggml-cpu/arch/*"],
     "keep kobold's trimmed repack; prism's Q1_0/PQ2_0 repack goes in gated like kobold's other "
     "x86 repacks."),
    (["ggml/src/ggml-metal/*"],
     "keep both GDN fusions (cache write-through + prism row indexing) on the new can_fuse API; "
     "the Hadamard support check belongs in the MUL_MAT case."),
    (["ggml/src/ggml-vulkan/*"],
     "PTQ1_0 is a per-type shader on top of upstream's shared mul_mm (coopmat2 falls back to "
     "dequant + f16)."),
    (["src/models/qwen35*"], "prism's joint q/k l2-norm view uses upstream's build_gdn_l2_norm."),
    (["friend/*", "koboldcpp.py", "expose.*", "FRIEND.md"],
     "friend.cpp-owned surface (adapters, prompt cache, steering): ours wins unless upstream "
     "fixed something underneath it."),
]


# ---------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------

def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def say(s: str = ""):
    print(s, flush=True)


def die(msg: str, code: int = 1):
    say(c("31", "error: ") + msg)
    sys.exit(code)


def git(*args: str, check: bool = True, capture: bool = True) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=capture, text=True)
    if check and p.returncode != 0:
        die(f"git {' '.join(args)} failed:\n{(p.stderr or p.stdout or '').strip()}")
    return (p.stdout or "").strip() if capture else ""


def git_rc(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def match(path: str, pats: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in pats)


def is_tools_kept(path: str) -> bool:
    return match(path, TOOLS_KEEP)


ROOT = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE,
                           capture_output=True, text=True).stdout.strip() or HERE.parents[1])
GIT_DIR = Path(subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip())
STATE = GIT_DIR / "friend-sync.json"


def load_state() -> dict | None:
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(st: dict):
    STATE.write_text(json.dumps(st, indent=2))


def merge_in_progress() -> bool:
    return (GIT_DIR / "MERGE_HEAD").exists()


# ---------------------------------------------------------------------------------
# stage 1: fetch + report
# ---------------------------------------------------------------------------------

def fetch_and_report(base: str, do_fetch: bool, only: set[str], at: dict[str, str] | None = None) -> list[dict]:
    """Returns one entry per source with the pinned sha and how far behind `base` is.
    `at` overrides a source's target (label -> commit-ish), e.g. to sync only up to a
    release instead of the branch tip."""
    at = at or {}
    out = []
    for label, remote, branch, human in SOURCES:
        if label not in only:
            continue
        if not git("remote", "get-url", remote, check=False):
            die(f"remote '{remote}' is not configured (expected {human})")
        if do_fetch:
            say(f"fetching {remote} {branch} ...")
            rc, msg = git_rc("fetch", "--no-tags", remote, branch)
            if rc != 0:
                die(f"fetch {remote} failed:\n{msg}")
        ref = at.get(label, f"{remote}/{branch}")
        sha =git("rev-parse", "--verify", ref + "^{commit}")
        behind = int(git("rev-list", "--count", f"{base}..{sha}"))
        ahead = int(git("rev-list", "--count", f"{sha}..{base}"))
        newest = git("log", "-1", "--format=%cs %s", sha)
        out.append(dict(label=label, remote=remote, branch=branch, human=human, ref=ref,
                        sha=sha, behind=behind, ahead=ahead, newest=newest))
    say()
    say(f"{'source':<18} {'behind':>7} {'(ours ahead)':>13}  newest upstream commit")
    for s in out:
        say(f"{s['ref']:<18} {s['behind']:>7} {s['ahead']:>13}  {s['newest'][:70]}")
    say(f"(counts are commits reachable from the upstream ref but not from {base}; prism's "
        "count shrinks after the kobold merge because both carry llama.cpp history)")
    return out


# ---------------------------------------------------------------------------------
# stage 2: merges
# ---------------------------------------------------------------------------------

def unmerged() -> list[tuple[str, str]]:
    """[(xy, path)] for every unmerged index entry, xy as in `git status --porcelain`."""
    res = []
    for line in git("status", "--porcelain=v1", "--untracked-files=no").splitlines():
        xy, path = line[:2], line[3:]
        if "U" in xy or xy in ("AA", "DD"):
            res.append((xy, path))
    return res


XY_NAMES = {"UU": "both modified", "AA": "both added", "DU": "deleted by us", "UD": "deleted by them",
            "AU": "added by us", "UA": "added by them", "DD": "both deleted"}


def apply_mechanical(stage: dict) -> dict:
    """Apply the rules to the current (in-progress) merge. Idempotent: safe to call again
    on --continue after the user touched things. Returns what it did."""
    kept_deleted, dropped = [], []
    gone_dirs: dict[str, bool] = {}

    def component_gone(path: str) -> bool:
        d = os.path.dirname(path)
        if not d:
            return False
        if d not in gone_dirs:
            gone_dirs[d] = not git("ls-tree", "--name-only", "HEAD", "--", d + "/")
        return gone_dirs[d]

    for xy, path in unmerged():
        if xy == "DU" and ((match(path, KEEP_DELETED) and not is_tools_kept(path)) or component_gone(path)):
            git("rm", "-q", "-f", "--", path)
            kept_deleted.append(path)
        elif xy == "DD":
            git("rm", "-q", "-f", "--", path)
    if stage["label"] == "prismml":
        added = git("diff", "--cached", "--name-only", "--diff-filter=A", "HEAD").splitlines()
        for path in added:
            if match(path, DROP_ADDED) and not is_tools_kept(path):
                dropped.append(path)
        # batch the removals: prism can add hundreds of files
        for i in range(0, len(dropped), 200):
            git("rm", "-q", "-f", "--", *dropped[i:i + 200])
    return {"kept_deleted": kept_deleted, "dropped": dropped}


def hints_for(paths: list[str]) -> list[str]:
    out = []
    for pats, text in HINTS:
        hit = [p for p in paths if match(p, pats)]
        if hit:
            out.append(f"  - {', '.join(hit[:4])}{' ...' if len(hit) > 4 else ''}:\n      {text}")
    return out


def conflict_report(st: dict, stage: dict, conflicts: list[tuple[str, str]]):
    say()
    say(c("33", f"merge of {stage['ref']} ({stage['sha'][:10]}) stopped: {len(conflicts)} conflict(s)"))
    for xy, path in conflicts:
        say(f"  {XY_NAMES.get(xy, xy):<16} {path}")
    mech = stage.get("mechanical", {})
    if mech.get("kept_deleted") or mech.get("dropped"):
        say(f"(already applied: kept {len(mech.get('kept_deleted', []))} deleted, "
            f"dropped {len(mech.get('dropped', []))} prism-added files)")
    h = hints_for([p for _, p in conflicts])
    if h:
        say("\nresolution notes for these files:")
        for line in h:
            say(line)
    say("\ngeneral rules: never take a side wholesale on friend.cpp-owned code (friend/, "
        "blue-noise, adapters, prompt cache, steering); `friend.cpp:` comments mark our changes.")
    say("\nnext:")
    say("  1. resolve the files above, `git add` them (or `git rm` for deletions)")
    say("  2. tools/friend-sync/sync.sh --continue")
    say("  (or tools/friend-sync/sync.sh --abort to give up; the branch is left for you to delete)")


def commit_message(stage: dict, hand: list[str]) -> str:
    mech = stage.get("mechanical", {})
    lines = [f"Merge {stage['human']} ({stage['ref']} {stage['sha'][:10]}) into friend.cpp", "",
             f"Brings in {stage['behind']} commits."]
    if mech.get("kept_deleted"):
        lines += ["", "Kept deleted (modify/delete on files kobold strips):"]
        lines += [f"- {p}" for p in mech["kept_deleted"][:40]]
        if len(mech["kept_deleted"]) > 40:
            lines.append(f"- ... and {len(mech['kept_deleted']) - 40} more")
    if mech.get("dropped"):
        lines += ["", f"Dropped {len(mech['dropped'])} prism-added tests/docs/CI/examples/tools files."]
    # files the rules settled on a later --continue were still "conflicts" when the stop was reported
    auto = set(mech.get("kept_deleted", [])) | set(mech.get("dropped", []))
    hand = [p for p in hand if p not in auto]
    if hand:
        lines += ["", "Resolved by hand:"] + [f"- {p}" for p in hand]
    lines += ["", "(merged with tools/friend-sync; edit this message with git commit --amend)"]
    return "\n".join(lines) + "\n"


def finish_merge(st: dict, stage: dict) -> bool:
    """Commit the in-progress merge if it's fully resolved. False (with a report) if not."""
    stage["mechanical"] = merge_mech(stage.get("mechanical", {}), apply_mechanical(stage))
    left = unmerged()
    if left:
        save_state(st)
        conflict_report(st, stage, left)
        return False
    # leftover markers in anything resolved by hand. Only those files: `git diff --check`
    # over the whole merge flags "=======" lines in embd_res/*merges*.embd and friends.
    markers = []
    for path in stage.get("conflicts", []):
        f = ROOT / path
        if not f.is_file():
            continue
        for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if line.startswith(("<<<<<<< ", ">>>>>>> ")) or line in ("<<<<<<<", ">>>>>>>"):
                markers.append(f"{path}:{n}: {line[:60]}")
    if markers:
        save_state(st)
        say(c("31", "conflict markers are still staged:"))
        for l in markers[:20]:
            say("  " + l)
        say("fix them, `git add`, then --continue")
        return False
    if not check_kernels_not_staged():
        save_state(st)
        return False
    msgfile = GIT_DIR / "friend-sync-msg.txt"
    msgfile.write_text(commit_message(stage, stage.get("conflicts", [])))
    git("commit", "-q", "--no-verify", "-F", str(msgfile))
    stage["merged"] = git("rev-parse", "HEAD")
    save_state(st)
    say(c("32", f"merged {stage['ref']} -> {stage['merged'][:10]}") + summary_mech(stage))
    return True


def merge_mech(a: dict, b: dict) -> dict:
    return {k: sorted(set(a.get(k, [])) | set(b.get(k, []))) for k in ("kept_deleted", "dropped")}


def summary_mech(stage: dict) -> str:
    m = stage.get("mechanical", {})
    parts = []
    if m.get("kept_deleted"):
        parts.append(f"kept {len(m['kept_deleted'])} deleted")
    if m.get("dropped"):
        parts.append(f"dropped {len(m['dropped'])} prism-added files")
    return f" ({', '.join(parts)})" if parts else ""


def print_dropped(stage: dict):
    m = stage.get("mechanical", {})
    for key, title in (("kept_deleted", "kept deleted"), ("dropped", "dropped (prism-added)")):
        items = m.get(key, [])
        if not items:
            continue
        # group by top two path components so 300 test files read as one line
        groups: dict[str, int] = {}
        for p in items:
            g = "/".join(p.split("/")[:2]) if "/" in p else p
            groups[g] = groups.get(g, 0) + 1
        say(f"  {title}: " + ", ".join(f"{g}{'' if n == 1 else f' ({n})'}" for g, n in sorted(groups.items())))


def run_stage(st: dict, stage: dict) -> bool:
    if stage.get("merged"):
        return True
    if int(git("rev-list", "--count", f"HEAD..{stage['sha']}")) == 0:
        say(f"{stage['ref']}: already contained, nothing to merge")
        stage["merged"] = "noop"
        save_state(st)
        return True
    say(f"\nmerging {stage['ref']} ({stage['sha'][:10]}, {stage['behind']} commits) ...")
    rc, out = git_rc("merge", "--no-ff", "--no-commit", "--no-edit", stage["sha"])
    if rc != 0 and not merge_in_progress():
        die(f"git merge failed before producing a merge state:\n{out}")
    stage["mechanical"] = apply_mechanical(stage)
    # anything still unmerged after the rules is for the human; remember which, for the message
    stage["conflicts"] = [p for _, p in unmerged()]
    save_state(st)
    if stage["mechanical"]["kept_deleted"] or stage["mechanical"]["dropped"]:
        print_dropped(stage)
    return finish_merge(st, stage)


# ---------------------------------------------------------------------------------
# stage 3: post-merge checks, build, smoke
# ---------------------------------------------------------------------------------

def has_code(path: Path) -> bool:
    """False for a .cpp that is only includes and comments (src/models/qwen3tts.cpp: the
    class lives entirely in models.h), which doesn't need to be in the unity build."""
    src = path.read_text(errors="replace")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"^\s*#[^\n]*", "", src, flags=re.M)
    return bool(src.strip())


def check_unity() -> bool:
    """kobold compiles src/models/*.cpp as one unit via #includes in src/llama-model.cpp,
    so a model file a merge adds is silently not built until someone includes it there
    (the link then fails, or worse, an arch just isn't registered)."""
    model_cpp = ROOT / "src/llama-model.cpp"
    have = sorted(p.name for p in (ROOT / "src/models").glob("*.cpp"))
    inc = set(re.findall(r'^\s*#\s*include\s+"models/([^"]+\.cpp)"', model_cpp.read_text(errors="replace"), re.M))
    missing = [n for n in have if n not in inc and has_code(ROOT / "src/models" / n)]
    dangling = sorted(n for n in inc if not (ROOT / "src/models" / n).exists())
    if missing or dangling:
        say(c("31", "unity build: FAIL"))
        for n in missing:
            say(f'  src/models/{n} is not built: add  #include "models/{n}"  to src/llama-model.cpp')
        for n in dangling:
            say(f"  src/llama-model.cpp includes models/{n}, which doesn't exist")
        return False
    say(f"unity build: ok ({len(inc)} of {len(have)} src/models/*.cpp included, the rest are header-only)"
        if len(inc) != len(have) else f"unity build: ok (all {len(have)} src/models/*.cpp included)")
    return True


def check_kernels_not_staged() -> bool:
    staged = git("diff", "--cached", "--name-only", "--", "kernels/").splitlines()
    tracked = git("ls-files", "--", "kernels/").splitlines()
    ignored = subprocess.run(["git", "check-ignore", "-q", "kernels/ggml-metal.metal"], cwd=ROOT).returncode == 0
    ok = True
    if staged or tracked:
        say(c("31", f"top-level kernels/ is in the index ({len(set(staged) | set(tracked))} files): "
              "it's a Makefile build artifact. `git rm -r --cached kernels/`"))
        ok = False
    if not ignored:
        say(c("33", "warning: /kernels/ is no longer gitignored (the merge touched .gitignore?)"))
    return ok


def build_command(jobs: int) -> tuple[list[str], list[str]]:
    """(make command, extra smoke-server flags) for this machine. FRIEND_SYNC_MAKE overrides."""
    if os.environ.get("FRIEND_SYNC_MAKE"):
        return shlex.split(os.environ["FRIEND_SYNC_MAKE"]), shlex.split(os.environ.get("FRIEND_SMOKE_FLAGS", ""))
    nice = ["nice"] if shutil.which("nice") else []
    if platform.system() == "Darwin":
        return nice + ["make", f"-j{jobs}", "LLAMA_METAL=1", "koboldcpp_default"], []
    if shutil.which("nvcc") or os.environ.get("CUDA_PATH"):
        return nice + ["make", f"-j{jobs}", "LLAMA_CUBLAS=1", "koboldcpp_cublas"], ["--usecuda"]
    return nice + ["make", f"-j{jobs}", "koboldcpp_default"], []


def post_merge(st: dict, args) -> bool:
    say("\n--- post-merge checks ---")
    ok = check_unity()
    ok = check_kernels_not_staged() and ok
    if git("ls-tree", "-r", "--name-only", "HEAD", "--", "kernels/"):
        say(c("31", "kernels/ was committed on this branch; remove it with git rm -r --cached kernels/ "
              "and amend"))
        ok = False
    if not ok:
        say("\nfix the above, commit, then tools/friend-sync/sync.sh --continue")
        return False

    if args.no_build:
        say("build: skipped (--no-build)")
    else:
        cmd, smoke_flags = build_command(args.jobs)
        log = GIT_DIR / "friend-sync-build.log"
        if args.clean_build:
            say("make clean ...")
            subprocess.run(["make", "clean"], cwd=ROOT, capture_output=True)
        say(f"build: {shlex.join(cmd)}  (log: {log})")
        with open(log, "w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            tail = log.read_text(errors="replace").splitlines()
            errs = [l for l in tail if " error" in l or "error:" in l][:15]
            say(c("31", f"build FAILED (exit {rc}). First errors:"))
            for l in errs or tail[-25:]:
                say("  " + l)
            say("\nfix, commit, then tools/friend-sync/sync.sh --continue  (--clean-build if you suspect "
                "stale objects: the kcpp mappers have bitten before)")
            return False
        say(c("32", "build: ok"))
        if smoke_flags and not os.environ.get("FRIEND_SMOKE_FLAGS"):
            os.environ["FRIEND_SMOKE_FLAGS"] = " ".join(smoke_flags)

    if args.no_smoke or args.no_build:
        say("smoke: skipped" + (" (no build)" if args.no_build and not args.no_smoke else ""))
        return True
    say("\n--- smoke tests ---")
    rc = subprocess.run([sys.executable, str(HERE / "smoke.py"), *shlex.split(args.smoke_args)], cwd=ROOT).returncode
    if rc != 0:
        say(c("31", "\nsmoke tests FAILED."))
        say("fix (or decide the failure is expected), commit, then tools/friend-sync/sync.sh --continue")
        return False
    return True


def next_steps(st: dict):
    base = st["base"]
    br = st["branch"]
    say("\n" + c("32", f"sync branch {br} is ready.") + " Nothing was pushed.")
    for s in st["stages"]:
        if s.get("merged") and s["merged"] != "noop":
            say(f"  {s['ref']:<18} merged as {s['merged'][:10]}{summary_mech(s)}")
    steps = [
        (f"git log --first-parent {base}..{br}", "review; each merge message lists what the rules did"),
        ("", "and which files were resolved by hand: add *how* (git commit --amend if HEAD is the merge)"),
        (f"git switch {base} && git merge --ff-only {br}", "when happy"),
        (f"git push origin {base}", "only if you mean it (origin = friend.cpp; upstream remotes can't push)"),
        (f"git branch -d {br}", "tidy up"),
    ]
    say("\nnext steps:")
    w = max(len(cmd) for cmd, _ in steps)
    for cmd, why in steps:
        say(f"  {cmd:<{w}}  # {why}")
    say("  also worth a look: FRIEND.md if a feature moved; a CUDA build on your CUDA box.")


# ---------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------

def cmd_status(args):
    fetch_and_report(args.base, not args.no_fetch, set(args.only), args.at_map)
    st = load_state()
    if st:
        say(f"\nsync in progress on {st['branch']} (started {st['started']}); --continue or --abort")


def cmd_start(args):
    if load_state():
        st = load_state()
        die(f"a sync is already in progress on {st['branch']}: use --continue or --abort")
    if merge_in_progress():
        die("a merge is already in progress in this checkout; finish or abort it first")
    dirty = git("status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        die("working tree has uncommitted changes to tracked files; commit or stash them first:\n" + dirty)
    git("rev-parse", "--verify", args.base + "^{commit}")

    sources = fetch_and_report(args.base, not args.no_fetch, set(args.only), args.at_map)
    todo = [s for s in sources if s["behind"] > 0]
    if not todo:
        say(c("32", f"\n{args.base} already contains every selected upstream. Nothing to do."))
        return 0

    branch = args.branch
    if not branch:
        branch = f"sync/{dt.date.today().isoformat()}"
        n = 2
        while git("rev-parse", "--verify", "--quiet", branch, check=False):
            branch = f"sync/{dt.date.today().isoformat()}.{n}"
            n += 1
    elif git("rev-parse", "--verify", "--quiet", branch, check=False):
        die(f"branch {branch} already exists")

    orig = git("rev-parse", "--abbrev-ref", "HEAD")
    git("switch", "-q", "-c", branch, args.base)
    st = dict(branch=branch, base=args.base, orig=orig, started=dt.datetime.now().isoformat(timespec="seconds"),
              base_sha=git("rev-parse", args.base), stages=todo)
    save_state(st)
    say(f"\ncreated {branch} from {args.base} ({st['base_sha'][:10]})")
    return drive(st, args)


def drive(st: dict, args) -> int:
    for stage in st["stages"]:
        if not run_stage(st, stage):
            return 2
    if not post_merge(st, args):
        return 3
    next_steps(st)
    STATE.unlink(missing_ok=True)
    return 0


def cmd_continue(args) -> int:
    st = load_state()
    if not st:
        die("no sync in progress (nothing in " + str(STATE) + ")")
    cur = git("rev-parse", "--abbrev-ref", "HEAD")
    if cur != st["branch"]:
        die(f"you're on {cur}, the sync is on {st['branch']}: git switch {st['branch']}")
    for stage in st["stages"]:
        if stage.get("merged"):
            continue
        if merge_in_progress():
            # the user resolved (some of) the conflicts of this stage
            if not finish_merge(st, stage):
                return 2
            break
        # merge state vanished: either the user committed the merge by hand, or aborted it
        head_parents = git("log", "-1", "--format=%P").split()
        if stage["sha"] in head_parents or int(git("rev-list", "--count", f"HEAD..{stage['sha']}")) == 0:
            stage["merged"] = git("rev-parse", "HEAD")
            say(f"{stage['ref']}: merge already committed ({stage['merged'][:10]})")
            save_state(st)
        break
    return drive(st, args)


def cmd_abort(args) -> int:
    st = load_state()
    if merge_in_progress():
        git("merge", "--abort")
        say("merge aborted")
    if st:
        dirty = git("status", "--porcelain=v1", "--untracked-files=no")
        if dirty:
            die("tracked files are modified, not switching branches:\n" + dirty)
        git("switch", "-q", st.get("orig") or st["base"])
        STATE.unlink(missing_ok=True)
        say(f"back on {st.get('orig') or st['base']}. The sync branch {st['branch']} is left in place "
            f"(git branch -D {st['branch']} to delete it).")
    else:
        say("no sync state found")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--status", "--dry-run", action="store_true", help="fetch and report only")
    mode.add_argument("--continue", dest="cont", action="store_true", help="resume after fixing a stop")
    mode.add_argument("--abort", action="store_true", help="abort the in-progress merge and go back")
    ap.add_argument("--base", default="concedo", help="friend.cpp branch to sync (default concedo)")
    ap.add_argument("--branch", help="name for the sync branch (default sync/<yyyy-mm-dd>)")
    ap.add_argument("--only", nargs="+", choices=[s[0] for s in SOURCES], default=[s[0] for s in SOURCES],
                    help="merge only these sources")
    ap.add_argument("--no-fetch", action="store_true", help="use the remote-tracking refs as they are")
    ap.add_argument("--no-build", action="store_true", help="skip build (and so smoke tests)")
    ap.add_argument("--no-smoke", action="store_true", help="skip smoke tests")
    ap.add_argument("--clean-build", action="store_true", help="make clean before building")
    ap.add_argument("--jobs", "-j", type=int, default=int(os.environ.get("FRIEND_SYNC_JOBS", "4")),
                    help="make -j (default 4, env FRIEND_SYNC_JOBS)")
    ap.add_argument("--smoke-args", default=os.environ.get("FRIEND_SYNC_SMOKE_ARGS", ""),
                    help="extra arguments for smoke.py, e.g. '--skip batch'")
    ap.add_argument("--at", nargs="+", metavar="SOURCE=COMMIT", default=[],
                    help="merge a source only up to COMMIT instead of its branch tip, e.g. upstream=v1.121")
    args = ap.parse_args()
    args.at_map = {}
    for item in args.at:
        k, _, v = item.partition("=")
        if k not in [s[0] for s in SOURCES] or not v:
            ap.error(f"--at wants SOURCE=COMMIT with SOURCE in {[s[0] for s in SOURCES]}")
        args.at_map[k] = v
    if args.status:
        cmd_status(args)
        return 0
    if args.abort:
        return cmd_abort(args)
    if args.cont:
        return cmd_continue(args)
    return cmd_start(args)


if __name__ == "__main__":
    sys.exit(main())
