"""Reproduce the paper's flagship demo, across scale, on Anthropic's own lenses.

The README's headline figure: on the ASCII-art face, selecting the `^` (the nose)
shows the lens reading out "nose" at mid layers -- a word that never appears in the
prompt. That single claim is the cleanest statement of what the J-space is supposed
to be, and it is the thing to reproduce before believing anything else.

At Qwen3-1.7B it does NOT reproduce: "nose" never gets near the top, for either lens.
But the J-lens readout at that position is coherent and on-topic (top-5 all `^`-like),
while the logit lens emits noise ('elry', 'uff', 'ideal'). So the lens is working --
the *concept* just is not there in a 1.7B model. Anthropic's own public demo of this
example runs on a 27B.

So: sweep it. If "nose" surfaces at some size, the flagship phenomenon is real, the
harness is validated, and the `association` floor at small scale is a CAPACITY result
with an emergence threshold. If it never surfaces, that is a much harder finding.

Also prints the rank-metric pathology in situ: pass@k scores a lens by
min-over-layers rank, which hands a noisy lens one lottery ticket per layer. A
confident lens correctly reading `^` puts an unrelated word like "nose" far down at
EVERY layer; a diffuse one parks it at middling rank somewhere by chance. That is why
the logit lens can "beat" the J-lens on rank while being qualitatively worthless.
"""

import argparse, gc, json, pathlib
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens
import jlens.examples as ex

REPO = "neuronpedia/jacobian-lens"
PUBLISHED = {
    "1.7B":  ("Qwen/Qwen3-1.7B",   "qwen3-1.7b",   "Qwen3-1.7B_jacobian_lens.pt"),
    "4B":    ("Qwen/Qwen3-4B",     "qwen3-4b",     "Qwen3-4B_jacobian_lens.pt"),
    "8B":    ("Qwen/Qwen3-8B",     "qwen3-8b",     "Qwen3-8B_jacobian_lens.pt"),
    "14B":   ("Qwen/Qwen3-14B",    "qwen3-14b",    "Qwen3-14B_jacobian_lens.pt"),
    "27B":   ("Qwen/Qwen3.5-27B",  "qwen3.5-27b",  "Qwen3.5-27B_jacobian_lens.pt"),
    "32B":   ("Qwen/Qwen3-32B",    "qwen3-32b",    "Qwen3-32B_jacobian_lens.pt"),
}
OUT = pathlib.Path("results/ascii")
DEV, DTYPE = "cuda", torch.bfloat16


def purge(name):
    import shutil
    from huggingface_hub.constants import HF_HUB_CACHE
    d = pathlib.Path(HF_HUB_CACHE) / f"models--{name.replace('/', '--')}"
    if d.exists():
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(d)
        print(f"  purged {sz:.1f}GB ({name})", flush=True)


def face_prompt():
    lst = [v for v in vars(ex).values()
           if isinstance(v, (list, tuple)) and v and isinstance(v[0], ex.Example)][0]
    return [e for e in lst if e.slug == "ascii-face"][0].prompt


def run(tag, do_purge):
    res = OUT / f"{tag}.json"
    if res.exists():
        print(f"[{tag}] cached", flush=True)
        return json.loads(res.read_text())

    hf_name, np_id, lens_file = PUBLISHED[tag]
    print(f"\n[{tag}] {hf_name}", flush=True)
    lens = jlens.JacobianLens.load(
        hf_hub_download(REPO, f"{np_id}/jlens/Salesforce-wikitext/{lens_file}"))
    layers = sorted(lens.jacobians)
    tok = AutoTokenizer.from_pretrained(hf_name)
    hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=DTYPE).to(DEV).eval()
    model = jlens.from_hf(hf, tok)

    prompt = face_prompt()
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    pos = [i for i, t in enumerate(toks) if "^" in t][0]

    nose = list({t[0] for t in (tok(w, add_special_tokens=False)["input_ids"]
                                for w in ("nose", " nose", "Nose", " Nose"))})

    out = {"model": hf_name, "d_model": lens.d_model, "n_layers": len(layers),
           "n_prompts": lens.n_prompts, "caret_pos": pos}
    for use_j, key in ((True, "j_lens"), (False, "logit_lens")):
        r, *_ = lens.apply(model, prompt, layers=layers,
                           max_seq_len=len(ids), use_jacobian=use_j)
        best, best_l, tops = 10**9, None, {}
        for l in layers:
            lg = r[l][pos]
            rk = min(int((lg > lg[t]).sum().item()) + 1 for t in nose)
            if rk < best:
                best, best_l = rk, l
            tops[l] = tok.convert_ids_to_tokens(lg.topk(5).indices.tolist())
        out[key] = {"best_rank_nose": best, "best_layer": best_l,
                    "top5_at_best": tops[best_l],
                    "top5_midstack": tops[layers[len(layers) // 2]]}
        print(f"  {key:11s} best rank(nose)={best:>7}  at layer {best_l}", flush=True)
        print(f"              top5 there : {tops[best_l]}", flush=True)
        print(f"              top5 mid   : {tops[layers[len(layers)//2]]}", flush=True)

    res.write_text(json.dumps(out, indent=2))
    del hf, model, lens
    gc.collect(); torch.cuda.empty_cache()
    if do_purge:
        purge(hf_name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["1.7B"], choices=list(PUBLISHED))
    ap.add_argument("--purge", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {t: run(t, a.purge) for t in a.models}

    print("\n" + "=" * 72)
    print('FLAGSHIP DEMO: does the lens read "nose" at the `^`? (never in the prompt)')
    print("=" * 72)
    print(f"{'model':>6} {'d_model':>8} | {'J rank(nose)':>13} {'layer':>6} | "
          f"{'LOGIT rank':>11}")
    print("-" * 72)
    for t, r in rows.items():
        print(f"{t:>6} {r['d_model']:>8} | {r['j_lens']['best_rank_nose']:>13} "
              f"{r['j_lens']['best_layer']:>6} | "
              f"{r['logit_lens']['best_rank_nose']:>11}")
    print("\nRANK 1 = the paper's claim reproduces. Anything in the hundreds = absent.")


if __name__ == "__main__":
    main()
