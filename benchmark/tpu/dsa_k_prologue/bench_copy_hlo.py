"""Test only: what is the ~120 us SparseCore copy before the first gather?

The extend_c1 timeline (exp-5knzugwt13, SparseCore layout, chunk 128) shows
``copy.165`` / ``copy.166`` (OFFLOAD_DATA_FORMATTING) running ~120 us on the
SparseCore before the first native gather, not hidden; the TensorCore-layout
``target`` has a same-size ``copy.111``. This runs the same sweep variants with
``jax.jit`` wrapped so each step's compiled HLO text is captured, then prints,
for the step modules (the ones that call the attention kernel):

* ``COPY``: every copy / copy-start whose result is >= 1 MiB: name, shape,
  layout, operand, and its users;
* ``DEF``: the defining line of each operand of those copies;
* ``PARAM``: the entry parameters (cache layouts as passed in).
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VAR = "sc_phys_c128_b16"
DT = {"u32": 4, "s32": 4, "f32": 4, "bf16": 2, "u8": 1, "s8": 1, "f8e4m3fn": 1, "u16": 2}
CAP = []


def nbytes(shape_txt):
    m = re.match(r"\(?(\w+)\[([\d,]*)\]", shape_txt)
    if not m or m.group(1) not in DT:
        return 0
    n = 1
    for d in filter(None, m.group(2).split(",")):
        n *= int(d)
    return n * DT[m.group(1)]


def report(tag, text):
    lines = text.splitlines()
    defs = {}
    for ln in lines:
        m = re.match(r"\s*(?:ROOT )?(%[\w.\-]+) = (.*)", ln)
        if m:
            defs[m.group(1)] = m.group(2)
    for ln in lines:
        if "parameter(" in ln and "ENTRY" not in ln and re.search(r"= \S+\[", ln):
            m = re.match(r"\s*(%[\w.\-]+) = (\S+) parameter\((\d+)\)", ln)
            if m and nbytes(m.group(2)) >= 1 << 20:
                print("PARAM " + json.dumps({"v": tag, "n": m.group(1), "i": m.group(3),
                                             "shape": m.group(2)[:120]}), flush=True)
    for name, rhs in defs.items():
        if not re.search(r"\b(copy|copy-start)\(", rhs):
            continue
        shape = rhs.split(" ")[0]
        if nbytes(shape) < 1 << 20:
            continue
        ops = re.findall(r"(?:copy|copy-start)\(([^)]*)\)", rhs)
        operands = [o.strip().split(" ")[-1] for o in (ops[0].split(",") if ops else [])]
        users = [n for n, r in defs.items() if re.search(re.escape(name) + r"\b", r)][:6]
        print("COPY " + json.dumps({"v": tag, "n": name, "mb": round(nbytes(shape) / 2**20, 1),
                                    "rhs": rhs[:400], "operands": operands,
                                    "users": users}), flush=True)
        for o in operands:
            if o in defs:
                print("DEF " + json.dumps({"v": tag, "n": o, "rhs": defs[o][:400]}), flush=True)
        for u in users:
            print("USE " + json.dumps({"v": tag, "of": name, "n": u, "rhs": defs[u][:300]}),
                  flush=True)


def main():
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax

    orig = jax.jit

    class Wrapped:
        def __init__(self, jf):
            self.jf, self.done = jf, False

        def __call__(self, *a, **k):
            if not self.done:
                self.done = True
                try:
                    CAP.append(self.jf.lower(*a, **k).compile().as_text())
                except Exception as e:  # report, do not hide
                    print(f"HLO_ERROR {e!r}"[:500], flush=True)
            return self.jf(*a, **k)

        def __getattr__(self, n):
            return getattr(self.jf, n)

    jax.jit = lambda f=None, **kw: Wrapped(orig(f, **kw))
    import bench_target_gather_sweep as sw

    sys.argv = ["sweep", "--cases", "pr2198_extend_c1", "--only", VAR,
                "--iters", "2", "--repeats", "1", "--xprof-dir", "/tmp/copy_hlo"]
    sw.main()
    jax.jit = orig
    steps = [t for t in CAP if "MLA-" in t]
    print("HLO_MODULES " + json.dumps({"captured": len(CAP), "steps": len(steps)}), flush=True)
    for i, t in enumerate(steps):
        tag = "sc" if "native_sc" in t else "tc"
        report(f"{tag}{i}", t)
    print("COPY_DONE", flush=True)


if __name__ == "__main__":
    main()
