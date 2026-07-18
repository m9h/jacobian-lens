"""The scaling curve, on Anthropic's OWN published lenses. Plus the gate.

Why this supersedes scaling_sweep.py:

    scaling_sweep.py fit every lens on a fixed 100 prompts. Anthropic's published
    config.yaml files show that is their *floor* (`--min_prompts 100`), not their
    fit size: they fit to convergence (`--stop_at_delta 0.002`, cap 1000), and the
    Qwen3 lenses actually consumed

        1.7B -> 466    4B -> 479    8B -> 461    14B -> 615    32B -> 615

    prompts. Every lens I fit was under-fit by 4.6-6x. That invalidated the sweep
    and fully explains its anomaly (my J-lens degrading with scale while the logit
    lens improved).

They publish converged lenses for the exact Qwen3 sizes I wanted. So: don't fit
anything. Use theirs. My fitting bug then cannot contaminate the result, and the
curve becomes a straight replication -- their lens, their model, their evals, their
pass@k metric -- with the logit lens as the floor.

GATE (per model, where we have a local under-fit lens to compare):
    cosine(our 100-prompt J, their converged J), and the eval delta it causes.
    This documents what the bug cost and proves the fix.

Then the question the whole exercise is for, now asked with a trustworthy lens:

    does `association` -- a concept EVOKED BUT NEVER NAMED -- lift off the floor
    as the model scales?
"""

import argparse, gc, json, pathlib
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

from lens_eval_comparison import evaluate, KS

REPO = "neuronpedia/jacobian-lens"
# np_id -> (hf model, lens filename, prompts Anthropic used to converge)
PUBLISHED = {
    "1.7B": ("Qwen/Qwen3-1.7B", "qwen3-1.7b", "Qwen3-1.7B_jacobian_lens.pt", 466),
    "4B":   ("Qwen/Qwen3-4B",   "qwen3-4b",   "Qwen3-4B_jacobian_lens.pt",   479),
    "8B":   ("Qwen/Qwen3-8B",   "qwen3-8b",   "Qwen3-8B_jacobian_lens.pt",   461),
    "14B":  ("Qwen/Qwen3-14B",  "qwen3-14b",  "Qwen3-14B_jacobian_lens.pt",  615),
    # The decisive pair for the association eval (102 vignettes, concept never named),
    # which is the ROBUST version of the one-prompt ASCII-face test:
    #   27B  = HYBRID (48/64 linear attention) -- the only model where "nose" surfaced
    #   32B  = DENSE, and LARGER -- the capability-vs-architecture control
    "27B":  ("Qwen/Qwen3.5-27B", "qwen3.5-27b", "Qwen3.5-27B_jacobian_lens.pt", 672),
    "32B":  ("allenai/Olmo-3-1125-32B", "olmo-3-1125-32b",
             "Olmo-3-1125-32B_jacobian_lens.pt", 470),
}
HEADLINE = ("association", "poetry")
DEV, DTYPE = "cuda", torch.bfloat16
OUT = pathlib.Path("results/published")
EVALS = pathlib.Path("data/evaluations")
UNDERFIT = pathlib.Path("results/scaling")   # our old 100-prompt lenses


def purge(name):
    import shutil
    from huggingface_hub.constants import HF_HUB_CACHE
    d = pathlib.Path(HF_HUB_CACHE) / f"models--{name.replace('/', '--')}"
    if d.exists():
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(d)
        print(f"  purged {sz:.1f}GB ({name})", flush=True)


def gate(tag, theirs):
    """Compare our under-fit 100-prompt lens to their converged one."""
    p = UNDERFIT / f"lens_{tag}.pt"
    if not p.exists():
        return None
    ours = jlens.JacobianLens.load(str(p))
    shared = sorted(set(ours.jacobians) & set(theirs.jacobians))
    if not shared:
        return None
    cos = []
    for l in shared:
        a = ours.jacobians[l].float().flatten()
        b = theirs.jacobians[l].float().to(a.device).flatten()
        cos.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    return {"n_prompts_ours": ours.n_prompts,
            "n_prompts_theirs": theirs.n_prompts,
            "mean_cosine": sum(cos) / len(cos),
            "min_cosine": min(cos)}


def run(tag, do_purge):
    res = OUT / f"eval_{tag}.json"
    if res.exists():
        print(f"[{tag}] cached", flush=True)
        return json.loads(res.read_text())

    hf_name, np_id, lens_file, n_conv = PUBLISHED[tag]
    print(f"\n[{tag}] {hf_name}  (their lens: {n_conv} prompts to converge)", flush=True)

    lens_path = hf_hub_download(REPO, f"{np_id}/jlens/Salesforce-wikitext/{lens_file}")
    theirs = jlens.JacobianLens.load(lens_path)
    layers = sorted(theirs.jacobians.keys())
    print(f"  lens: {len(layers)} layers, d_model={theirs.d_model}, "
          f"n_prompts={theirs.n_prompts}", flush=True)

    tok = AutoTokenizer.from_pretrained(hf_name)
    hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=DTYPE).to(DEV).eval()
    model = jlens.from_hf(hf, tok)

    g = gate(tag, theirs)
    if g:
        print(f"  GATE  our {g['n_prompts_ours']}-prompt lens vs their "
              f"{g['n_prompts_theirs']}-prompt lens: "
              f"mean cosine {g['mean_cosine']:.4f} (min {g['min_cosine']:.4f})",
              flush=True)

    out = {"model": hf_name, "d_model": theirs.d_model,
           "n_layers": len(layers), "their_n_prompts": theirs.n_prompts,
           "gate_vs_our_underfit": g, "evals": {}}

    for path in sorted(EVALS.glob("lens-eval-*.json")):
        slug = path.stem.replace("lens-eval-", "")
        items = json.loads(path.read_text())["items"]
        out["evals"][slug] = {
            "j_lens": evaluate(theirs, model, tok, items, slug, layers, True),
            "logit_lens": evaluate(theirs, model, tok, items, slug, layers, False),
        }
        e = out["evals"][slug]
        star = " *" if slug in HEADLINE else "  "
        print(f"  {slug:14s}{star} J@1={e['j_lens']['pass@1']:.3f} "
              f"J@10={e['j_lens']['pass@10']:.3f} | "
              f"L@1={e['logit_lens']['pass@1']:.3f} "
              f"L@10={e['logit_lens']['pass@10']:.3f}", flush=True)

    res.write_text(json.dumps(out, indent=2))
    del hf, model, theirs
    gc.collect(); torch.cuda.empty_cache()
    if do_purge:
        purge(hf_name)
    return out


def report(tags):
    rows = {t: json.loads((OUT / f"eval_{t}.json").read_text())
            for t in tags if (OUT / f"eval_{t}.json").exists()}
    if not rows:
        return
    print("\n" + "=" * 78)
    print("REPLICATION: J-lens beats logit lens? (mean pass@10 over the six evals)")
    print("=" * 78)
    for t, r in rows.items():
        j = sum(e["j_lens"]["pass@10"] for e in r["evals"].values()) / 6
        l = sum(e["logit_lens"]["pass@10"] for e in r["evals"].values()) / 6
        ok = "PASS" if j > l else "FAIL"
        print(f"  {t:>5}  J={j:.3f}  logit={l:.3f}   {ok}")

    print("\n" + "=" * 78)
    print("THE CURVE: concept EVOKED BUT NEVER NAMED  (association, their lens)")
    print("=" * 78)
    print(f"{'model':>6} {'d_model':>8} {'layers':>7} | "
          + "  ".join(f"{'J@' + str(k):>7}" for k in KS)
          + " | " + "  ".join(f"{'L@' + str(k):>7}" for k in KS))
    print("-" * 78)
    for t, r in rows.items():
        a, b = r["evals"]["association"]["j_lens"], r["evals"]["association"]["logit_lens"]
        print(f"{t:>6} {r['d_model']:>8} {r['n_layers']:>7} | "
              + "  ".join(f"{a['pass@' + str(k)]:>7.3f}" for k in KS) + " | "
              + "  ".join(f"{b['pass@' + str(k)]:>7.3f}" for k in KS))

    print("\n" + "=" * 78)
    print("ALL EVALS, J-lens pass@10 (logit floor in parens)")
    print("=" * 78)
    slugs = sorted(next(iter(rows.values()))["evals"])
    print(f"{'eval':14s} " + "  ".join(f"{t:>15}" for t in rows))
    print("-" * 78)
    for s in slugs:
        star = " *" if s in HEADLINE else "  "
        cells = [f"{r['evals'][s]['j_lens']['pass@10']:.3f} "
                 f"({r['evals'][s]['logit_lens']['pass@10']:.3f})" for r in rows.values()]
        print(f"{s:12s}{star} " + "  ".join(f"{c:>15}" for c in cells))
    print("\n  * = operationalizes the flagship 'unstated concept' claim")

    gates = [(t, r["gate_vs_our_underfit"]) for t, r in rows.items()
             if r.get("gate_vs_our_underfit")]
    if gates:
        print("\n" + "=" * 78)
        print("WHAT THE UNDER-FITTING BUG COST (our 100-prompt J vs their converged J)")
        print("=" * 78)
        for t, g in gates:
            print(f"  {t:>5}  ours {g['n_prompts_ours']:>4} prompts vs theirs "
                  f"{g['n_prompts_theirs']:>4}  ->  mean cosine {g['mean_cosine']:.4f}")

    lifted = [t for t, r in rows.items()
              if r["evals"]["association"]["j_lens"]["pass@10"] > 0.05]
    print("\n" + "=" * 78 + "\nVERDICT\n" + "=" * 78)
    if lifted:
        print(f"`association` lifts off the floor at {lifted[0]}.")
        print("The flagship phenomenon is REAL and has an emergence threshold.")
    else:
        print("`association` stays at FLOOR across every size, on ANTHROPIC'S OWN")
        print("CONVERGED LENSES. The under-fitting explanation is now excluded.")
        print("If the other five evals replicate (J > logit), the harness is sound,")
        print("and this is a real finding about what the J-lens reads.")
    (OUT / "curve.json").write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(PUBLISHED), choices=list(PUBLISHED))
    ap.add_argument("--purge", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for t in a.models:
        run(t, a.purge)
    report(a.models)


if __name__ == "__main__":
    main()
