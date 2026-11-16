"""Recover each Source file from its copy (drop header, undo patches and prefix) and compare sha256."""
import hashlib, json, pathlib
DST = pathlib.Path("sglang_tpu/attention/pallas/source_dsa")
m = json.loads((DST / "MANIFEST.json").read_text())
old, new = m["import_prefix"]
bad = 0
for rel, info in m["files"].items():
    text = (DST / rel).read_text().split("\n", 4)[4]
    for p in m.get("patches", []):
        if p["file"] == rel:
            if "import_added" in p:
                text = text.replace(p["import_added"] + "\n", "", 1)
            text = text.replace(p["new"], p["old"])
    got = hashlib.sha256(text.replace(new, old).encode()).hexdigest()
    ok = got == info["sha256"]
    bad += not ok
    print("OK  " if ok else "BAD ", rel)
print("ALL OK" if not bad else f"{bad} MISMATCH")
