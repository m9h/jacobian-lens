"""Does the J-space's flagship property emerge with scale?

On Qwen3-0.6B, the two evals that operationalize the headline claim -- `association`
(a vignette evokes a concept, e.g. "grief", that is NEVER named) and `poetry` (the
unstated rhyme target) -- score ZERO for the J-lens at every rank out to 50, while
the surface-lexical evals (typo, multilingual) work fine.

Two explanations, indistinguishable from one model:

  (a) MODEL CAPACITY. A 0.6B model does not carry abstract, never-stated concepts
      anywhere, so no lens can read them out. The method is fine.
  (b) METHOD. The J-lens surfaces lexical/surface content, and the flagship
      "unstated concept" demos do not generalise.

Sweeping scale separates them. If `association` lifts off zero at some size, the
phenomenon is real and has an EMERGENCE THRESHOLD -- a number nobody has published.
If it stays at floor through 14B while typo/multilingual keep working, that is a
much more serious finding about what the lens actually reads.

Design notes:
  * Qwen3 only -- one family, one tokenizer, one pretraining recipe. Scale is then
    the only variable that moves. Mixing families would confound it.
  * Every model gets its OWN fitted lens (a lens is model-specific) AND its own
    logit-lens baseline, so the J-vs-logit contrast is available at every size.
  * Resumable: a fitted lens or a completed eval is never recomputed.

Usage:
    python experiments/scaling_sweep.py                       # all sizes
    python experiments/scaling_sweep.py --models 0.6B 1.7B    # what fits on an 8GB card
    python experiments/scaling_sweep.py --models 8B 14B       # on the Spark
"""

import argparse, gc, json, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

from randomization_control import corpus, DEV
from lens_eval_comparison import evaluate, KS

MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
    "14B": "Qwen/Qwen3-14B",
}
# The two evals that carry the flagship claim. Everything else is context.
HEADLINE = ("association", "poetry")

OUT = pathlib.Path("results/scaling")
EVALS = pathlib.Path("data/evaluations")


def fit_or_load(model, tok, tag, n_fit):
    path = OUT / f"lens_{tag}.pt"
    if path.exists():
        print(f"  lens: cached ({path.name})", flush=True)
        return jlens.JacobianLens.load(str(path))
    fit_p, _ = corpus(tok, n_fit, 0)
    print(f"  lens: fitting on {len(fit_p)} prompts...", flush=True)
    lens = jlens.fit(model, fit_p, checkpoint_path=str(OUT / f"ckpt_{tag}.pt"))
    lens.save(str(path))
    return lens


def purge_weights(name):
    """Delete a model's HF snapshot. Disk on the Spark is tight (~70GB free) and
    the sweep pulls ~56GB of weights; they are all re-downloadable."""
    import shutil
    from huggingface_hub.constants import HF_HUB_CACHE
    d = pathlib.Path(HF_HUB_CACHE) / f"models--{name.replace('/', '--')}"
    if d.exists():
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(d)
        print(f"  purged {sz:.1f}GB of weights ({name})", flush=True)


def run_model(tag, n_fit, purge=False):
    res_path = OUT / f"eval_{tag}.json"
    if res_path.exists():
        print(f"[{tag}] cached", flush=True)
        return json.loads(res_path.read_text())

    name = MODELS[tag]
    print(f"\n[{tag}] {name}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    # NB: do NOT use device_map="auto". Accelerate offloads to CPU/meta and its
    # offload hooks are incompatible with the backward hooks jlens.fit() installs
    # ("Cannot copy out of meta tensor"). The GB10 has 119GB unified memory; every
    # model in this sweep fits whole. Load it whole.
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    cfg = hf.config
    model = jlens.from_hf(hf, tok)
    lens = fit_or_load(model, tok, tag, n_fit)
    layers = sorted(lens.jacobians.keys())

    out = {
        "model": name,
        "n_layers": cfg.num_hidden_layers,
        "d_model": cfg.hidden_size,
        "evals": {},
    }
    for path in sorted(EVALS.glob("lens-eval-*.json")):
        slug = path.stem.replace("lens-eval-", "")
        items = json.loads(path.read_text())["items"]
        out["evals"][slug] = {
            n: evaluate(lens, model, tok, items, slug, layers, use_j)
            for n, use_j in (("j_lens", True), ("logit_lens", False))
        }
        e = out["evals"][slug]
        star = " *" if slug in HEADLINE else "  "
        print(f"  {slug:14s}{star} J@1={e['j_lens']['pass@1']:.3f} "
              f"J@10={e['j_lens']['pass@10']:.3f} | "
              f"L@1={e['logit_lens']['pass@1']:.3f} "
              f"L@10={e['logit_lens']['pass@10']:.3f}", flush=True)

    res_path.write_text(json.dumps(out, indent=2))
    del hf, model, lens
    gc.collect()
    torch.cuda.empty_cache()
    if purge:
        purge_weights(name)
    return out


def report(tags):
    rows = {}
    for t in tags:
        p = OUT / f"eval_{t}.json"
        if p.exists():
            rows[t] = json.loads(p.read_text())
    if not rows:
        return

    print("\n" + "=" * 74)
    print("THE HEADLINE CURVE  --  concept evoked but NEVER NAMED (J-lens pass@k)")
    print("=" * 74)
    print(f"{'model':>7} {'layers':>7} {'d_model':>8} | "
          + "  ".join(f"{'assoc@' + str(k):>10}" for k in (1, 10, 50)))
    print("-" * 74)
    for t, r in rows.items():
        a = r["evals"]["association"]["j_lens"]
        print(f"{t:>7} {r['n_layers']:>7} {r['d_model']:>8} | "
              + "  ".join(f"{a['pass@' + str(k)]:>10.3f}" for k in (1, 10, 50)))

    print("\n" + "=" * 74)
    print("ALL EVALS, J-lens pass@10 (logit-lens floor in parens)")
    print("=" * 74)
    slugs = sorted(next(iter(rows.values()))["evals"])
    print(f"{'eval':14s} " + "  ".join(f"{t:>14}" for t in rows))
    print("-" * 74)
    for s in slugs:
        star = " *" if s in HEADLINE else "  "
        cells = []
        for t, r in rows.items():
            e = r["evals"][s]
            cells.append(f"{e['j_lens']['pass@10']:.3f} ({e['logit_lens']['pass@10']:.3f})")
        print(f"{s:12s}{star} " + "  ".join(f"{c:>14}" for c in cells))
    print("\n  * = operationalizes the flagship 'unstated concept' claim")

    lifted = [t for t, r in rows.items()
              if r["evals"]["association"]["j_lens"]["pass@10"] > 0.05]
    print("\n" + "=" * 74 + "\nVERDICT\n" + "=" * 74)
    if lifted:
        print(f"`association` lifts off the floor at {lifted[0]}.")
        print("The flagship phenomenon is REAL and has an emergence threshold.")
        print("-> Report the threshold. Nobody has published this curve.")
    else:
        print("`association` stays at FLOOR across every size tested.")
        print("Meanwhile typo/multilingual work at every size. So the lens reads")
        print("surface-lexical content but NOT abstract never-stated concepts --")
        print("at least up to the largest model swept. That is a real finding about")
        print("what the J-lens actually reads, and it needs the biggest model you")
        print("can reach before you publish it.")
    (OUT / "curve.json").write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--n-fit", type=int, default=100)
    ap.add_argument("--purge", action="store_true",
                    help="delete each model's weights after its eval (tight disk)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for t in a.models:
        run_model(t, a.n_fit, purge=a.purge)
    report(a.models)


if __name__ == "__main__":
    main()
