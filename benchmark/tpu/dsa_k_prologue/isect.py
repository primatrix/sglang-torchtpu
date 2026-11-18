import glob, json, os, sys
import numpy as np
G = (2, 4, 8, 16, 32, 64, 128)
for root in sys.argv[1:]:
    files = sorted(glob.glob(f"{root}/**/raw-*.npy", recursive=True))
    print("FILES " + json.dumps({"root": root[-60:], "n": len(files),
          "names": [os.path.basename(f) for f in files]}), flush=True)
    for f in files:
        r = np.load(f)
        k = r.shape[1]
        nv = (r >= 0).sum(1)
        out = {"f": os.path.basename(f), "shape": list(r.shape), "valid_p50": int(np.median(nv)),
               "full": float(np.mean(nv == k)), "max_pos": int(r.max())}
        sets = [np.sort(x[x >= 0]) for x in r]
        for g in G:
            inter, rem = [], []
            for b in range(0, len(sets) - g + 1, g):
                s = sets[b]
                for t in sets[b + 1:b + g]:
                    s = np.intersect1d(s, t, assume_unique=True)
                n = sum(len(t) for t in sets[b:b + g])
                inter.append(len(s))
                # shared+private: gather |I| once plus each query's private rows
                rem.append((n - (g - 1) * len(s)) / max(n, 1))
            if inter:
                a = np.array(inter)
                out[str(g)] = [round(a.mean() / k, 4), round(float(np.percentile(a, 10)) / k, 4),
                               round(float(np.percentile(a, 90)) / k, 4), round(float(np.mean(rem)), 4)]
        # how many of a 128-query block's queries pick each row
        cov = []
        for b in range(0, len(sets) - 127, 128):
            u, c = np.unique(np.concatenate(sets[b:b + 128]), return_counts=True)
            cov.append(np.bincount(np.minimum(c, 128) // 16, minlength=9)[:9] / len(u))
        if cov:
            out["cov16"] = [round(float(x), 3) for x in np.mean(cov, 0)]
        print("ISECT " + json.dumps(out), flush=True)
print("ISECT_DONE", flush=True)
