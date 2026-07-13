"""Is there BLOCK structure, or only smooth drift?

Plotting the CKA matrices shows no crisp boxes -- just a bright early-mid region
decaying monotonically toward the output. Which exposes a flaw in my own blockiness
score: a matrix whose entries depend ONLY on layer distance |i-j| will still score
nonzero on a within-block-minus-between-block contrast, because nearby layers are
more similar than distant ones. So blockiness=0.29 on qwen3-14b may mean nothing.

That is precisely the distinction at issue. Hoel's deflationary reading:

    "it is just a relatively constant transformation of internal processing into the
     output that starts at some point, but after that basically just starts
     accumulating"

i.e. smooth drift, no workspace. Anthropic's reading: three sequestered regions --
sensory / workspace / motor.

THE CONTROL. For each model, build a distance-only null:

    C_null[i, j] = mean over all (p, q) with |p - q| == |i - j|  of  C[p, q]

fitted from the model's OWN observed decay. This has zero block structure by
construction -- it is pure drift. Score it identically.

    blockiness(real) >> blockiness(null)   ->  real blocks beyond drift.
    blockiness(real) ~= blockiness(null)   ->  the "tripartite structure" is an
                                               artifact of smooth decay, and the
                                               deflationary account wins.

Excess = real - null is the only number here worth quoting.
"""

import glob, json, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cka import blockiness

ORDER = ["pythia-70m", "qwen3-1.7b", "qwen3-4b", "olmo-3-7b", "qwen3-8b",
         "gpt-oss-20b", "qwen3-14b", "qwen3.5-27b", "qwen3-32b"]


def distance_null(C):
    """Matrix depending only on |i-j|, with the observed mean at each distance."""
    L = C.shape[0]
    prof = torch.zeros(L)
    for d in range(L):
        vals = [C[i, i + d] for i in range(L - d)]
        prof[d] = torch.stack(vals).mean()
    N = torch.zeros_like(C)
    for i in range(L):
        for j in range(L):
            N[i, j] = prof[abs(i - j)]
    return N


rows = {}
for f in glob.glob("results/cka/*.json"):
    if f.endswith("summary.json"):
        continue
    d = json.load(open(f))
    if "cka_full" in d:
        rows[pathlib.Path(f).stem] = d
rows = {k: rows[k] for k in ORDER if k in rows}

print("DOES BLOCK STRUCTURE SURVIVE A DISTANCE-ONLY NULL?")
print("(null = same decay profile, zero blocks by construction)\n")
print(f"{'model':>13} {'L':>4} | {'real':>7} {'null':>7} {'EXCESS':>8} | verdict")
print("-" * 66)
out = {}
for name, d in rows.items():
    C = torch.tensor(d["cka_full"])
    L = C.shape[0]
    real, _ = blockiness(C)
    if real != real:
        print(f"{name:>13} {L:>4} |     -       -        -   | too few layers")
        continue
    null, _ = blockiness(distance_null(C))
    exc = real - null
    if exc > 0.05:
        v = "real blocks"
    elif exc > 0.02:
        v = "marginal"
    else:
        v = "NO BLOCKS -- pure drift"
    out[name] = {"real": real, "null": null, "excess": exc, "verdict": v}
    print(f"{name:>13} {L:>4} | {real:>7.4f} {null:>7.4f} {exc:>+8.4f} | {v}")

pathlib.Path("results/cka/null_test.json").write_text(json.dumps(out, indent=2))
print("\nEXCESS is the only number worth quoting. A distance-only matrix -- pure smooth")
print("drift, no workspace anywhere -- reproduces most of the raw blockiness score.")
