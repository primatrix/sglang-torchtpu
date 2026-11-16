"""Compare the optimized HLO two processes compiled for the same JAX function.

Usage: hlo_compare.py <dump_dir_a> <dump_dir_b> <marker> [<marker> ...]

Each dump dir is an ``XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_as_text``
output. A module is picked when its ``after_optimizations`` text contains every
marker (a shape string such as ``bf16[2000,32,64]``). For the picked module of
each side this prints: fusion count, per-fusion op histogram, the element types
of elementwise arithmetic inside fusions (bf16 arithmetic computed as f32 is
XLA "excess precision"), converts, and a short unified diff of the op lines
with instruction names stripped.
"""

import collections
import difflib
import glob
import os
import re
import sys

ARITH = ("multiply", "add", "subtract", "negate")
_LINE = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(\S+)\s+(\w[\w\-]*)\(")


def pick(dump_dir, markers):
    hits = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "*after_optimizations*.txt"))):
        if "buffer-assignment" in f or "memory-usage" in f:
            continue
        text = open(f).read()
        if all(m in text for m in markers):
            hits.append((len(text), f, text))
    return hits


def computations(text):
    """name -> list of body lines."""
    out, name, body = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$", line)
        if m and not line.startswith(" "):
            name, body = m.group(1), []
            continue
        if line.startswith("}") and name:
            out[name] = body
            name = None
            continue
        if name:
            body.append(line)
    return out


def summary(text):
    comps = computations(text)
    fused = {n: b for n, b in comps.items() if "fused" in n or "fusion" in n}
    arith_types, converts, hist = collections.Counter(), collections.Counter(), {}
    for n, body in fused.items():
        h = collections.Counter()
        for line in body:
            m = _LINE.match(line)
            if not m:
                continue
            ty, op = m.group(1), m.group(2)
            h[op] += 1
            dtype = ty.split("[")[0]
            if op in ARITH:
                arith_types[(op, dtype)] += 1
            if op == "convert":
                src = re.search(r"convert\((\w+)\[", line)
                converts[f"{src.group(1) if src else '?'}->{dtype}"] += 1
        hist[n] = dict(h)
    return {"fusions": len(fused), "arith_dtypes": dict(arith_types),
            "converts": dict(converts), "fusion_ops": hist}


def normalized(text):
    lines = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            lines.append(f"{m.group(1)} {m.group(2)}")
    return lines


def main():
    a_dir, b_dir, markers = sys.argv[1], sys.argv[2], sys.argv[3:]
    sides = {}
    for tag, d in (("A", a_dir), ("B", b_dir)):
        hits = pick(d, markers)
        print(f"== {tag} {d}: {len(hits)} module(s) with {markers}")
        for size, f, _ in hits:
            print(f"   {os.path.basename(f)} ({size} chars)")
        if hits:
            sides[tag] = max(hits)  # the largest: the whole step, not a helper
    for tag, (_, f, text) in sides.items():
        s = summary(text)
        print(f"== {tag} summary of {os.path.basename(f)}")
        print(f"   fusions={s['fusions']}")
        print(f"   arith dtypes in fusions={s['arith_dtypes']}")
        print(f"   converts in fusions={s['converts']}")
        for n, h in sorted(s["fusion_ops"].items()):
            if any(op in h for op in ARITH):
                print(f"   {n}: {h}")
    if len(sides) == 2:
        diff = list(difflib.unified_diff(normalized(sides["A"][2]), normalized(sides["B"][2]),
                                         "A", "B", lineterm="", n=0))
        print(f"== normalized op-line diff: {len(diff)} lines (first 80)")
        print("\n".join(diff[:80]))


if __name__ == "__main__":
    main()
