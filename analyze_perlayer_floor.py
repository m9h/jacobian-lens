"""Restate the OLMo post-training result as excess over a PER-LAYER refit floor.

The published headline -- cos(J_base, J_instruct) = 0.69, "~31% move" -- is a mean over 11
layers whose independent-refit floors differ by an order of magnitude (0.884 at layer 0,
0.9998 at layer 30). Pooling them means the same cosine carries very different weight at
either end. This recomputes per layer and reports movement RELATIVE to what two independent
fits of the SAME model already differ by.

floor_l   = cos between our lens and Neuronpedia's independent fit, at layer l
excess_l  = (floor_l - cos_l) / (floor_l - 0)   ... how far past the noise the arm moved,
            as a fraction of the distance from the floor to total dissimilarity

Needs no GPU: it reads published lenses.
"""
import torch
from huggingface_hub import hf_hub_download

OURS = "mhough/olmo3-jacobian-lenses"
NP_REPO, NP_PATH = ("neuronpedia/jacobian-lens",
                    "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt")
ARMS = [("Instruct", "lenses/olmo-3-7b-instruct.pt"),
        ("Think", "lenses/olmo-3-7b-think.pt"),
        ("RL-Zero math", "lenses/olmo-3-7b-rl-zero-math.pt"),
        ("RL-Zero code", "lenses/olmo-3-7b-rl-zero-code.pt")]

def J(repo, path):
    return torch.load(hf_hub_download(repo, path), map_location="cpu", weights_only=False)["J"]

base = J(OURS, "lenses/olmo-3-1025-7b.pt")
np_j = J(NP_REPO, NP_PATH)
layers = sorted(set(base) & set(np_j))
def cos(a, b):
    x = a.float().flatten().double(); y = b.float().flatten().double()
    return float(x @ y / (x.norm() * y.norm()))

floor = {l: cos(base[l], np_j[l]) for l in layers}
print("PER-LAYER REFIT FLOOR (our fit vs Neuronpedia's independent fit, same model)")
print("  " + "  ".join(f"L{l}:{floor[l]:.3f}" for l in layers))
print()
hdr = f"{'arm':14s} {'pooled cos':>11} {'pooled move':>12} | {'excess over per-layer floor':>28}"
print(hdr); print("-" * len(hdr))
for name, path in ARMS:
    a = J(OURS, path)
    ls = [l for l in layers if l in a]
    cs = {l: cos(base[l], a[l]) for l in ls}
    pooled = sum(cs.values()) / len(cs)
    # movement beyond what two fits of the same model already differ by
    exc = {l: max(0.0, (floor[l] - cs[l]) / max(floor[l], 1e-9)) for l in ls}
    mean_exc = sum(exc.values()) / len(exc)
    early = [l for l in ls if l <= 9]; late = [l for l in ls if l >= 21]
    print(f"{name:14s} {pooled:11.4f} {100*(1-pooled):11.1f}% | "
          f"mean {100*mean_exc:5.1f}%   early(L0-9) {100*sum(exc[l] for l in early)/len(early):5.1f}%"
          f"   late(L21-30) {100*sum(exc[l] for l in late)/len(late):5.1f}%")
print()
print("Reading: 'pooled move' is the published statistic. 'excess' asks how far the arm moved")
print("BEYOND the distance two independent fits of the same model already sit apart, per layer.")
