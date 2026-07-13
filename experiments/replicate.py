"""REPLICATION GATE. Run this before any variant. Nothing downstream is
interpretable until this passes.

The scaling sweep produced an anomaly -- the J-lens got *worse* with model size
while the logit lens got better -- and it turned out to be my own bug: I fit every
lens on a fixed 100 prompts. Anthropic's published config.yaml shows they do not do
that. They fit to CONVERGENCE:

    --n_prompts 1000 --min_prompts 100 --stop_window 10 --stop_at_delta 0.002

and the prompts actually consumed grow with model width:

    gemma-2-2b   -> 454 prompts    (d_model 2304)
    gemma-3-12b  -> 775 prompts    (d_model 3840)

100 is their *floor*, the point at which convergence checking begins. So my lenses
were under-fit, and progressively more so with scale. That fully explains the
anomaly and invalidates the sweep.

This script establishes the gate Anthropic's own artifacts make possible. The paper's
quantitative lens comparison lives in appendix *figures* (no table to match), but
they published the fitted lens weights AND the per-prompt convergence traces, which
is a stronger and more exact target than any figure.

  GATE 1 -- FITTING. Fit gemma-2-2b with their protocol; compare our J to their
            published J, layer by layer (relative Frobenius error, cosine).
            Passing means our estimator is theirs.

  GATE 2 -- SCORING. Run our six evals with THEIR lens and with OURS. The two
            should agree; and the J-lens should beat the logit lens, which is the
            paper's qualitative claim. Passing means our scoring harness is sound.

Only if both pass do the variants (randomization control, scaling curve) mean
anything.
"""

import argparse, json, math, pathlib
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens
from jlens.fitting import jacobian_for_prompt

from lens_eval_comparison import evaluate

HF_MODEL = "google/gemma-2-2b"
NP_LENS = ("neuronpedia/jacobian-lens", "gemma-2-2b/jlens/Salesforce-wikitext")
DEV, DTYPE = "cuda", torch.bfloat16
OUT = pathlib.Path("results/replicate")

# Their fit hyperparameters, verbatim from config.yaml.
N_MAX, MIN_PROMPTS, STOP_WINDOW, STOP_AT_DELTA = 1000, 100, 10, 0.002
MAX_SEQ_LEN, DIM_BATCH, MAX_CHARS, SKIP_FIRST = 128, 128, 2000, 16


def their_corpus(tok, n):
    """Salesforce/wikitext-103-raw-v1 train, text[:2000], >=128 tokens -- as in config.yaml."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    out, it = [], iter(ds)
    while len(out) < n:
        t = next(it)["text"].strip()[:MAX_CHARS]
        if len(t) < 400 or t.startswith("="):
            continue
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) >= MAX_SEQ_LEN:
            out.append(t)
    return out


def fit_to_convergence(model, prompts, source_layers, log_csv):
    """Mirror Neuronpedia's fit_lens.py: running mean of per-prompt Jacobians,
    stop when mean_rel_change stays under STOP_AT_DELTA for STOP_WINDOW prompts.

    mean_rel_change is computed exactly as jlens.fitting.fit does:
        max_l  ||J_p - Jbar|| / ((n+1) * ||Jbar||)
    """
    jac_sum, n_done, under, rows = None, 0, 0, []
    for p in prompts:
        try:
            per_prompt, seq_len, n_valid = jacobian_for_prompt(
                model, p, source_layers, dim_batch=DIM_BATCH,
                max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST)
        except Exception:
            continue
        if jac_sum is None:
            jac_sum = {l: torch.zeros_like(per_prompt[l]) for l in source_layers}
        if n_done == 0:
            mrc = float("nan")
        else:
            mrc = max(
                ((per_prompt[l] - jac_sum[l] / n_done).norm()
                 / ((n_done + 1) * (jac_sum[l] / n_done).norm())).item()
                for l in source_layers)
        for l in source_layers:
            jac_sum[l] += per_prompt[l]
        n_done += 1
        rows.append((n_done, seq_len, n_valid, mrc))
        if n_done >= MIN_PROMPTS and mrc == mrc and mrc < STOP_AT_DELTA:
            under += 1
            if under >= STOP_WINDOW:
                print(f"  converged at {n_done} prompts "
                      f"(mean_rel_change={mrc:.6f} < {STOP_AT_DELTA} "
                      f"for {STOP_WINDOW} consecutive)", flush=True)
                break
        else:
            under = 0
        if n_done % 50 == 0:
            print(f"    {n_done:4d} prompts  mean_rel_change={mrc:.6f}", flush=True)
    log_csv.write_text("n_done,seq_len,n_valid_positions,mean_rel_change\n" +
                       "\n".join(f"{a},{b},{c},{d:.8f}" for a, b, c, d in rows))
    jac_mean = {l: jac_sum[l] / n_done for l in source_layers}
    d_model = next(iter(jac_mean.values())).shape[-1]
    return jlens.JacobianLens(jacobians=jac_mean, n_prompts=n_done, d_model=d_model), n_done


def compare_lenses(mine, theirs):
    """Per-layer relative Frobenius error and cosine similarity."""
    shared = sorted(set(mine.jacobians) & set(theirs.jacobians))
    rows = []
    for l in shared:
        a = mine.jacobians[l].float()
        b = theirs.jacobians[l].float().to(a.device)
        rel = ((a - b).norm() / b.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item()
        rows.append((l, rel, cos))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fit", action="store_true",
                    help="reuse a previously fitted lens")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading {HF_MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    hf = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=DTYPE).to(DEV).eval()
    model = jlens.from_hf(hf, tok)

    print("downloading Anthropic's published lens", flush=True)
    repo, prefix = NP_LENS
    their_path = hf_hub_download(repo, f"{prefix}/gemma-2-2b_jacobian_lens.pt")
    theirs = jlens.JacobianLens.load(their_path)
    layers = sorted(theirs.jacobians.keys())
    print(f"  theirs: {len(layers)} layers, d_model={theirs.d_model}, "
          f"n_prompts={theirs.n_prompts}", flush=True)

    # ---------------- GATE 1: fitting ----------------
    mine_path = OUT / "lens_mine.pt"
    if a.skip_fit and mine_path.exists():
        mine = jlens.JacobianLens.load(str(mine_path))
        n_used = mine.n_prompts
    else:
        print(f"\nGATE 1: fitting our own lens with THEIR protocol "
              f"(converge @ delta<{STOP_AT_DELTA}, min={MIN_PROMPTS}, max={N_MAX})", flush=True)
        prompts = their_corpus(tok, N_MAX)
        mine, n_used = fit_to_convergence(
            model, prompts, layers, OUT / "our_convergence.csv")
        mine.save(str(mine_path))

    rows = compare_lenses(mine, theirs)
    mean_rel = sum(r[1] for r in rows) / len(rows)
    mean_cos = sum(r[2] for r in rows) / len(rows)
    print(f"\n  our prompts to converge: {n_used}   (theirs: 454)")
    print(f"  {'layer':>5} {'rel_err':>9} {'cosine':>9}")
    for l, rel, cos in rows[::4]:
        print(f"  {l:>5} {rel:>9.4f} {cos:>9.4f}")
    print(f"  {'MEAN':>5} {mean_rel:>9.4f} {mean_cos:>9.4f}")
    gate1 = mean_cos > 0.95
    print(f"\n  GATE 1: {'PASS' if gate1 else 'FAIL'} "
          f"(mean cosine {mean_cos:.4f} vs their published J)")

    # ---------------- GATE 2: scoring ----------------
    print("\nGATE 2: our eval harness, their lens vs ours, J vs logit", flush=True)
    ev = pathlib.Path("data/evaluations")
    report = {}
    for path in sorted(ev.glob("lens-eval-*.json")):
        slug = path.stem.replace("lens-eval-", "")
        items = json.loads(path.read_text())["items"]
        report[slug] = {
            "theirs_j": evaluate(theirs, model, tok, items, slug, layers, True),
            "ours_j": evaluate(mine, model, tok, items, slug, layers, True),
            "logit": evaluate(theirs, model, tok, items, slug, layers, False),
        }
        r = report[slug]
        print(f"  {slug:14s} THEIR-J@10={r['theirs_j']['pass@10']:.3f}  "
              f"OUR-J@10={r['ours_j']['pass@10']:.3f}  "
              f"LOGIT@10={r['logit']['pass@10']:.3f}", flush=True)

    (OUT / "gate2.json").write_text(json.dumps(report, indent=2))
    tj = sum(r["theirs_j"]["pass@10"] for r in report.values()) / len(report)
    oj = sum(r["ours_j"]["pass@10"] for r in report.values()) / len(report)
    lg = sum(r["logit"]["pass@10"] for r in report.values()) / len(report)
    agree = abs(tj - oj) < 0.05
    beats = tj > lg
    print(f"\n  mean pass@10   their-J={tj:.3f}  our-J={oj:.3f}  logit={lg:.3f}")
    print(f"  our lens agrees with theirs: {'YES' if agree else 'NO'}")
    print(f"  J-lens beats logit lens:     {'YES' if beats else 'NO'}  "
          f"(the paper's qualitative claim)")
    gate2 = agree and beats

    print("\n" + "=" * 66)
    print(f"GATE 1 (fitting) {'PASS' if gate1 else 'FAIL'}   "
          f"GATE 2 (scoring) {'PASS' if gate2 else 'FAIL'}")
    print("=" * 66)
    if gate1 and gate2:
        print("Harness reproduces Anthropic's artifacts. Variants are now interpretable.")
        print("Re-run the scaling sweep with convergence-based fitting.")
    else:
        print("STOP. Do not interpret any variant until this passes.")
        if not gate1:
            print("  -> our J differs from theirs: fitting protocol / corpus is wrong.")
        if not gate2:
            print("  -> our scoring differs, or the J>logit direction does not hold.")


if __name__ == "__main__":
    main()
