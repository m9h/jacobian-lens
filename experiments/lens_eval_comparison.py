"""J-lens vs. plain logit lens on Anthropic's OWN lens-quality evals.

The next-token-prediction comparison (logit_lens_baseline.py) is the wrong yardstick:
the J-lens is not for predicting the next token, it is for surfacing *latent concepts*
that appear nowhere in the prompt and are not the next token. Anthropic ship six eval
sets built to score exactly that (data/evaluations/, tagged §methods-comparison).

So run their evals, their metric, and change only the transport:

    pass@k = mean over items of the fraction of `intermediates` whose
             min-over-layers lens rank <= k, read at a single position.

    J-lens >> logit lens  -> the Jacobian earns its keep on the task it is for.
    J-lens ~= logit lens  -> the latent concepts are recoverable by unembedding the
                             raw residual stream. "J-space" is then the logit-lens
                             phenomenon with a better name.

Readout positions are taken from data/evaluations/README.md:
  association, typo, multihop, multilingual, order-ops -> final prompt token
  poetry                                               -> last newline token
"""

import json, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

MODEL, DEV, DTYPE = "Qwen/Qwen3-0.6B", "cuda", torch.bfloat16
OUT = pathlib.Path("results")
KS = (1, 5, 10, 50)

NUMWORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty",
}
# Per the eval README: order-ops intermediates are KEYS expanded to synonym sets.
OPWORD = {
    "multiplication": ["*", "×", "multiplication", "times", "multiply"],
    "addition": ["+", "addition", "plus", "add"],
    "subtraction": ["-", "subtraction", "minus", "subtract"],
    "division": ["/", "division", "divide", "divided"],
}


def synonyms(word: str, expand_ops: bool) -> list[str]:
    forms = [word]
    if expand_ops:
        if word in NUMWORD:
            forms.append(NUMWORD[word])
        if word in OPWORD:
            forms += OPWORD[word]
    out = []
    for f in forms:
        out += [f, " " + f, f.capitalize(), " " + f.capitalize()]
    return list(dict.fromkeys(out))


def single_token_ids(tok, word, expand_ops):
    ids = []
    for form in synonyms(word, expand_ops):
        t = tok(form, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.append(t[0])
    return ids


def readout_pos(tok, prompt, slug):
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if slug == "poetry":
        toks = tok.convert_ids_to_tokens(ids)
        nl = [i for i, t in enumerate(toks) if "Ċ" in t or "\n" in t]
        if nl:
            return nl[-1], len(ids)
    return len(ids) - 1, len(ids)


@torch.no_grad()
def evaluate(lens, model, tok, items, slug, layers, use_jacobian):
    expand = slug == "order-ops"
    per_item = []
    for it in items:
        prompt = it["prompt"]
        pos, n = readout_pos(tok, prompt, slug)
        readouts, *_ = lens.apply(
            model, prompt, layers=layers, max_seq_len=n, use_jacobian=use_jacobian
        )
        best = {}  # intermediate -> min rank over layers & synonyms
        for w in it["intermediates"]:
            cand = single_token_ids(tok, w, expand)
            if not cand:
                best[w] = None          # not representable as a single token
                continue
            r_min = 10**9
            for l in layers:
                lg = readouts[l][pos]
                for tid in cand:
                    r = int((lg > lg[tid]).sum().item()) + 1
                    r_min = min(r_min, r)
            best[w] = r_min
        per_item.append(best)
    scored = [{w: r for w, r in b.items() if r is not None} for b in per_item]
    n_skip = sum(1 for b in per_item for r in b.values() if r is None)
    out = {}
    for k in KS:
        fr = [
            sum(r <= k for r in b.values()) / len(b)
            for b in scored if b
        ]
        out[f"pass@{k}"] = sum(fr) / max(len(fr), 1)
    out["n_items"] = len(items)
    out["n_targets_skipped_multitoken"] = n_skip
    return out


def main():
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(str(OUT / "lens_trained.pt"))
    layers = sorted(lens.jacobians.keys())

    report = {}
    for path in sorted(pathlib.Path("data/evaluations").glob("lens-eval-*.json")):
        slug = path.stem.replace("lens-eval-", "")
        items = json.loads(path.read_text())["items"]
        report[slug] = {}
        for name, use_j in (("j_lens", True), ("logit_lens", False)):
            report[slug][name] = evaluate(
                lens, model, tok, items, slug, layers, use_j
            )
        r = report[slug]
        print(f"{slug:14s} n={r['j_lens']['n_items']:>3}  "
              f"J pass@1={r['j_lens']['pass@1']:.3f} pass@10={r['j_lens']['pass@10']:.3f}  |  "
              f"LOGIT pass@1={r['logit_lens']['pass@1']:.3f} "
              f"pass@10={r['logit_lens']['pass@10']:.3f}", flush=True)

    (OUT / "lens_eval_comparison.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print(f"{'eval':14s} " + "  ".join(f"{'J@'+str(k):>7}" for k in KS)
          + " | " + "  ".join(f"{'L@'+str(k):>7}" for k in KS))
    print("-" * 78)
    agg = {n: {k: [] for k in KS} for n in ("j_lens", "logit_lens")}
    for slug, r in report.items():
        row = f"{slug:14s} "
        for n in ("j_lens", "logit_lens"):
            for k in KS:
                agg[n][k].append(r[n][f"pass@{k}"])
        row += "  ".join(f"{r['j_lens'][f'pass@{k}']:>7.3f}" for k in KS) + " | "
        row += "  ".join(f"{r['logit_lens'][f'pass@{k}']:>7.3f}" for k in KS)
        print(row)
    print("-" * 78)
    mean = {n: {k: sum(v[k]) / len(v[k]) for k in KS} for n, v in agg.items()}
    print(f"{'MEAN':14s} "
          + "  ".join(f"{mean['j_lens'][k]:>7.3f}" for k in KS) + " | "
          + "  ".join(f"{mean['logit_lens'][k]:>7.3f}" for k in KS))

    j1, l1 = mean["j_lens"][1], mean["logit_lens"][1]
    j10, l10 = mean["j_lens"][10], mean["logit_lens"][10]
    print("\n" + "=" * 78 + "\nVERDICT\n" + "=" * 78)
    print(f"mean pass@1   J={j1:.3f}  logit={l1:.3f}   ({j1 - l1:+.3f})")
    print(f"mean pass@10  J={j10:.3f}  logit={l10:.3f}   ({j10 - l10:+.3f})")
    if j10 > 1.5 * max(l10, 1e-9):
        print("\nJ-lens CLEARLY beats the logit lens on the task the lens is FOR.")
        print("The Jacobian earns its keep. The next-token result was the wrong yardstick.")
    elif j10 > 1.15 * max(l10, 1e-9):
        print("\nJ-lens beats the logit lens, but modestly. Real, incremental.")
        print("J-space results must be reported against the LOGIT-LENS floor, not chance.")
    else:
        print("\nJ-lens does NOT beat the logit lens on Anthropic's own lens-quality evals.")
        print("The latent concepts are recoverable by unembedding the raw residual stream.")
        print("'J-space' would then be largely the logit-lens phenomenon, renamed --")
        print("and Trask's residual-stream objection has real force.")


if __name__ == "__main__":
    main()
