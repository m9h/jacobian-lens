"""Is the sensory / workspace / motor block structure real?

Erik Hoel's critique (Jul 2026) concedes that most of the paper is baked into
reportability by construction -- but singles out ONE part that is not:

    the CKA layer x layer matrix, whose sharp tripartite block structure is
    "arguably one of the most important parts of the entire paper, precisely
    because it's not (entirely) baked-in by the measure's grounding in
    reportability."

His falsification rests on a third-party demo he explicitly will not stand behind
("this isn't my analysis... it's not an exact replication (which would be important
to see)"). We have Anthropic's code, their published converged lenses for 38 models,
and a GPU. So run it properly.

And Anthropic buried a caveat about their own headline figure:

    "We note that in some models the transition is more gradual, sometimes
     containing sub-blocks, and that the observed sharpness is exaggerated by
     layer subsampling."

That is a checkable claim about their own figure. So this script does two things:

  1. Compute the CKA matrix at FULL layer resolution for each open model.
  2. Recompute it SUBSAMPLED to ~the number of layers shown in the paper's figure,
     and measure how much the subsampling inflates apparent blockiness.

Method. The J-lens gives, for vocabulary token t at layer l, a lens vector
    v[l,t] = J_l^T @ W_U[t]
(the transposed row -- the repo uses exactly this for steering). Stack those over a
token sample to get a representation of the J-space at layer l, then take linear CKA
between every pair of layers. No forward passes needed: the lens and the unembedding
are enough.

Blockiness score: fit the best contiguous 3-block segmentation of the CKA matrix by
exhaustive search over the two cut points, and report

    within-block mean CKA  -  between-block mean CKA

A sharp sensory/workspace/motor structure scores high. A gradual drift scores ~0.
Reported at full resolution and subsampled, so the inflation is explicit.
"""

import argparse, gc, itertools, json, pathlib
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

REPO = "neuronpedia/jacobian-lens"
# tag -> (hf model, neuronpedia id, lens filename)
MODELS = {
    "pythia-70m":  ("EleutherAI/pythia-70m-deduped", "pythia-70m-deduped", "pythia-70m-deduped_jacobian_lens.pt"),
    "qwen3-1.7b":  ("Qwen/Qwen3-1.7B",  "qwen3-1.7b",  "Qwen3-1.7B_jacobian_lens.pt"),
    "qwen3-4b":    ("Qwen/Qwen3-4B",    "qwen3-4b",    "Qwen3-4B_jacobian_lens.pt"),
    "qwen3-8b":    ("Qwen/Qwen3-8B",    "qwen3-8b",    "Qwen3-8B_jacobian_lens.pt"),
    "qwen3-14b":   ("Qwen/Qwen3-14B",   "qwen3-14b",   "Qwen3-14B_jacobian_lens.pt"),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B", "llama3.1-8b", "llama3.1-8b_jacobian_lens.pt"),
    "olmo-3-7b":   ("allenai/Olmo-3-1025-7B", "olmo-3-1025-7b", "Olmo-3-1025-7B_jacobian_lens.pt"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "gpt-oss-20b", "gpt-oss-20b_jacobian_lens.pt"),
}
OUT = pathlib.Path("results/cka")
DEV, DTYPE = "cuda", torch.float32
N_TOKENS = 4096          # vocabulary sample used to build the J-space representation
PAPER_LAYERS = 12        # roughly what the paper's figure subsamples to


def purge(name):
    import shutil
    from huggingface_hub.constants import HF_HUB_CACHE
    d = pathlib.Path(HF_HUB_CACHE) / f"models--{name.replace('/', '--')}"
    if d.exists():
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(d)
        print(f"  purged {sz:.1f}GB", flush=True)


def linear_cka(X, Y):
    """Linear CKA between two [n, d] representations (columns centred)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y).norm() ** 2
    xtx = (X.T @ X).norm()
    yty = (Y.T @ Y).norm()
    return (xty / (xtx * yty)).item()


def blockiness(C, min_frac=0.15):
    """Best contiguous 3-block segmentation: within-block mean minus between-block mean.

    Each block must hold at least ``min_frac`` of the layers. Without that floor the
    search cheats: it carves off a SINGLE layer as its own "block", which has CKA 1.0
    with itself by construction, and the score is maximised by a degenerate split that
    is not a sensory/workspace/motor structure at all. (Observed on qwen3-1.7b: the
    unconstrained optimum was cuts (1, 26) on 27 layers -- i.e. {L0} {L1..25} {L26}.)
    """
    L = C.shape[0]
    m = max(2, int(round(min_frac * L)))
    if L < 3 * m:
        return float("nan"), None
    best, cuts = -1e9, None
    for a, b in itertools.combinations(range(m, L - m + 1), 2):
        if a < m or b - a < m or L - b < m:
            continue
        seg = [(0, a), (a, b), (b, L)]
        win, wn, bet, bn = 0.0, 0, 0.0, 0
        for i, (s0, e0) in enumerate(seg):
            for j, (s1, e1) in enumerate(seg):
                blk = C[s0:e0, s1:e1]
                if blk.numel() == 0:
                    continue
                if i == j:
                    win += blk.sum().item(); wn += blk.numel()
                else:
                    bet += blk.sum().item(); bn += blk.numel()
        if wn == 0 or bn == 0:
            continue
        score = win / wn - bet / bn
        if score > best:
            best, cuts = score, (a, b)
    return best, cuts


def run(tag, do_purge):
    res = OUT / f"{tag}.json"
    if res.exists():
        print(f"[{tag}] cached", flush=True)
        return json.loads(res.read_text())

    hf_name, np_id, lens_file = MODELS[tag]
    print(f"\n[{tag}] {hf_name}", flush=True)
    lens = jlens.JacobianLens.load(
        hf_hub_download(REPO, f"{np_id}/jlens/Salesforce-wikitext/{lens_file}"))
    layers = sorted(lens.jacobians)

    hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16).eval()
    W_U = hf.get_output_embeddings().weight.detach()          # [vocab, d_model]
    del hf; gc.collect()

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(W_U.shape[0], generator=g)[:N_TOKENS]
    Wt = W_U[idx].to(DEV, DTYPE)                              # [n_tokens, d_model]
    del W_U; gc.collect(); torch.cuda.empty_cache()

    # J-space representation at layer l: v[l,t] = J_l^T @ W_U[t]  ->  [n_tokens, d_model]
    reps = {}
    for l in layers:
        J = lens.jacobians[l].to(DEV, DTYPE)
        reps[l] = Wt @ J                                      # (J^T W^T)^T
        del J
    torch.cuda.empty_cache()

    L = len(layers)
    C = torch.zeros(L, L)
    for i in range(L):
        for j in range(i, L):
            v = linear_cka(reps[layers[i]], reps[layers[j]])
            C[i, j] = C[j, i] = v

    full_score, full_cuts = blockiness(C)

    # Anthropic: "the observed sharpness is exaggerated by layer subsampling"
    step = max(1, L // PAPER_LAYERS)
    sub = list(range(0, L, step))[:PAPER_LAYERS]
    Cs = C[sub][:, sub]
    sub_score, sub_cuts = blockiness(Cs)

    out = {
        "model": hf_name, "n_layers": L, "d_model": lens.d_model,
        "lens_n_prompts": lens.n_prompts,
        "cka_full": C.tolist(),
        "blockiness_full": full_score, "cuts_full": full_cuts,
        "blockiness_subsampled": sub_score, "cuts_subsampled": sub_cuts,
        "n_subsampled": len(sub),
        "inflation": (sub_score - full_score) if full_score == full_score else None,
    }
    res.write_text(json.dumps(out))
    print(f"  layers={L}  blockiness full={full_score:.4f} (cuts {full_cuts})", flush=True)
    print(f"           subsampled to {len(sub)} = {sub_score:.4f} (cuts {sub_cuts})"
          f"   inflation {sub_score - full_score:+.4f}", flush=True)

    del reps, Wt; gc.collect(); torch.cuda.empty_cache()
    if do_purge:
        purge(hf_name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--purge", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for t in a.models:
        try:
            rows[t] = run(t, a.purge)
        except Exception as e:
            print(f"[{t}] FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 74)
    print("IS THE SENSORY / WORKSPACE / MOTOR BLOCK STRUCTURE REAL?")
    print("=" * 74)
    print(f"{'model':>12} {'layers':>7} | {'blockiness':>11} {'subsampled':>11} "
          f"{'inflation':>10} | cuts (full)")
    print("-" * 74)
    for t, r in rows.items():
        print(f"{t:>12} {r['n_layers']:>7} | {r['blockiness_full']:>11.4f} "
              f"{r['blockiness_subsampled']:>11.4f} "
              f"{r['blockiness_subsampled'] - r['blockiness_full']:>+10.4f} | "
              f"{r['cuts_full']}")
    print("\nblockiness = within-block mean CKA - between-block mean CKA, best 3-block split.")
    print("A sharp tripartite structure scores high. A gradual drift scores ~0.")
    print("inflation > 0 confirms Anthropic's own caveat that subsampling exaggerates it.")
    (OUT / "summary.json").write_text(json.dumps(
        {t: {k: v for k, v in r.items() if k != "cka_full"} for t, r in rows.items()},
        indent=2))


if __name__ == "__main__":
    main()
