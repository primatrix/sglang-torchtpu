"""Report the large copies in an XLA dump (``--xla_dump_to``), with their neighbours.

For exp-wr4cl7mbh6: the torch op on a SparseCore cache spent ~16 x 1046 us in
``copy`` ops of ``u32[262144,128]`` that the direct JAX call does not have. This
reads every ``*after_optimizations*.txt`` module under ``dump_dir``, keeps the
largest one (the attention step), and prints for each copy of at least
``--min-elems`` elements:

* ``HLOCOPY``: the copy itself (shape with layout and tiling),
* ``HLOSRC``: the instruction producing its operand,
* ``HLOUSE``: up to 3 instructions reading its result.

It also prints the custom calls (Pallas / SparseCore kernels) whose result has a
``[*, 128]`` u32 shape, so the gather output's layout shows up next to the copy.
"""

import argparse
import glob
import os
import re

INSTR = re.compile(r"^\s*(?:ROOT\s+)?(%[\w.\-]+)\s*=\s*(.*)$")
SHAPE = re.compile(r"[a-z]\w*\[([\d,]*)\]")


def elems(text):
    m = SHAPE.search(text)
    if not m or not m.group(1):
        return 0
    n = 1
    for d in m.group(1).split(","):
        n *= int(d)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--tag", required=True)
    p.add_argument("--min-elems", type=int, default=8 * 1024 * 1024)
    p.add_argument("--width", type=int, default=420)
    a = p.parse_args()
    files = glob.glob(os.path.join(a.dump_dir, "**", "*after_optimizations*.txt"), recursive=True)
    files = [f for f in files if "buffer-assignment" not in f and "memory-usage" not in f]
    if not files:
        print(f"HLONONE {a.tag} no after_optimizations dump under {a.dump_dir}")
        return
    f = max(files, key=os.path.getsize)
    print(f"HLOFILE {a.tag} {os.path.basename(f)} ({len(files)} modules)")
    defs, lines = {}, open(f, encoding="utf-8", errors="replace").read().splitlines()
    for ln in lines:
        m = INSTR.match(ln)
        if m:
            defs[m.group(1)] = ln.strip()
    w = a.width
    copies = 0
    for name, ln in defs.items():
        body = ln.split("=", 1)[1]
        if " copy(" not in body or elems(body) < a.min_elems:
            continue
        copies += 1
        if copies > 4:  # all copies are alike; the count is printed below
            continue
        print(f"HLOCOPY {a.tag} {ln[:w]}")
        ops = re.findall(r"copy\(([^)]*)\)", body)
        for op in re.findall(r"%[\w.\-]+", ops[0] if ops else ""):
            print(f"HLOSRC  {a.tag} {defs.get(op, op + ' (not found)')[:w]}")
        users = [d for n, d in defs.items() if n != name and re.search(re.escape(name) + r"\b", d.split("=", 1)[1])]
        for u in users[:3]:
            print(f"HLOUSE  {a.tag} {u[:w]}")
    print(f"HLOCOPIES {a.tag} {copies} copies of >= {a.min_elems} elements")
    seen = 0
    for name, ln in defs.items():
        body = ln.split("=", 1)[1]
        if "custom-call(" in body and "u32[" in body.split("custom-call(")[0] and seen < 6:
            seen += 1
            print(f"HLOCC   {a.tag} {ln[:w]}")


if __name__ == "__main__":
    main()
