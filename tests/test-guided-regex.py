#!/usr/bin/env python3
"""Focused checks for the portable guided-regex grammar compiler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from friend.guided_regex import compile_regex

assert compile_regex(r"^[A-Z]{2}-\d{4}$").startswith("root ::= regex-")
assert "regex-3 ::= " in compile_regex(r"^item-(yes|no)?$")
assert "A-Za-z0-9_" in compile_regex(r"^\w+$")
for invalid in (r"(?=secret)secret", r"(a)\1", r"a{65}"):
    try:
        compile_regex(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted unsupported regex: {invalid}")
print("guided regex checks passed")
