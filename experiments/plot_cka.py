"""Render the CKA matrices. Both Anthropic and Hoel argue from what the figure LOOKS
like, so a scalar cannot settle it -- and my blockiness scalar already fooled me once
(it maximised itself by carving single layers off as their own "blocks").

Emits results/cka/cka_grid.png: one heatmap per model, ordered by size, with the best
3-block split drawn on. The question is whether a sensory / workspace / motor structure
is visible, and whether it sharpens at 27B -- the size where the ASCII-face "nose"
readout emerges (rank 2 at 27B vs 164 at 14B).
"""

import glob, json, pathlib, sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cka import blockiness

ORDER = ["pythia-70m", "qwen3-1.7b", "qwen3-4b", "olmo-3-7b", "qwen3-8b",
         "gpt-oss-20b", "qwen3-14b", "qwen3.5-27b", "qwen3-32b"]

rows = {}
for f in glob.glob("results/cka/*.json"):
    if f.endswith("summary.json"):
        continue
    d = json.load(open(f))
    if "cka_full" in d:
        rows[pathlib.Path(f).stem] = d
rows = {k: rows[k] for k in ORDER if k in rows}
if not rows:
    raise SystemExit("no CKA matrices found")

n = len(rows)
cols = min(4, n)
r = (n + cols - 1) // cols
fig, axes = plt.subplots(r, cols, figsize=(4.1 * cols, 4.3 * r), squeeze=False)

for ax in axes.flat:
    ax.axis("off")

for i, (name, d) in enumerate(rows.items()):
    ax = axes[i // cols][i % cols]
    ax.axis("on")
    C = torch.tensor(d["cka_full"])
    L = C.shape[0]
    im = ax.imshow(C, cmap="magma", vmin=0, vmax=1, origin="lower",
                   interpolation="nearest")
    sc, cuts = blockiness(C)
    if cuts:
        for c in cuts:
            ax.axhline(c - 0.5, color="#4de0ff", lw=1.0, alpha=0.9)
            ax.axvline(c - 0.5, color="#4de0ff", lw=1.0, alpha=0.9)
    star = "  ★ nose emerges here" if name == "qwen3.5-27b" else ""
    ax.set_title(f"{name}   L={L}   blockiness={sc:.3f}{star}",
                 fontsize=9.5, pad=6)
    ax.set_xlabel("layer", fontsize=8)
    ax.set_ylabel("layer", fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle("Linear CKA between J-space representations, layer x layer\n"
             "(Anthropic's published lenses; cyan = best 3-block split, >=15% per block)",
             fontsize=12)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cb.set_label("CKA", fontsize=9)
out = pathlib.Path("results/cka/cka_grid.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
