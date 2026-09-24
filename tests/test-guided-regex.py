#!/usr/bin/env python3
"""Focused checks for the portable guided-regex grammar compiler."""
import ast
import json
import threading

source = open("koboldcpp.py", encoding="utf-8").read()
tree = ast.parse(source)
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
    "regex_to_gbnf", "cached_regex_to_gbnf"}]
namespace = {"json": json, "threading": threading,
             "structured_grammar_lock": threading.Lock(), "structured_grammar_cache": {}}
exec(compile(ast.Module(body=selected, type_ignores=[]), "koboldcpp.py", "exec"), namespace)

compile_regex = namespace["regex_to_gbnf"]
cached = namespace["cached_regex_to_gbnf"]

assert compile_regex(r"^[A-Z]{2}-\d{4}$").startswith("root ::= [A-Z] [A-Z]")
assert '"i" "t" "e" "m" "-"' in compile_regex(r"^item-(yes|no)?$")
assert "[A-Za-z0-9_]" in compile_regex(r"^\w+$")
assert compile_regex(r"(?=secret)secret") == ""
compiled = cached(r"^(cat|dog)$")
assert compiled == cached(r"^(cat|dog)$")
assert compiled.startswith("root ::= ")
print("guided regex checks passed")
