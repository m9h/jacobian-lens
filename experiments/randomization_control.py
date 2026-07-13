"""Model-randomization control for the Jacobian lens (Adebayo et al. 2018, adapted).

The deflationary hypothesis (Trask): a J-lens readout looks semantic because the
residual stream is additive and the unembedding is trained -- not because the
blocks learned a workspace. If so, the lens should still produce coherent
readouts when the blocks know nothing.

To test that fairly we randomize ONLY the transformer blocks and keep the trained
embedding / final norm / unembedding. Randomizing everything would make token
identity meaningless and the control would pass for the wrong reason.

Two readouts discriminate the hypotheses at each layer:

  next_acc  lens top-1 == the true next token in the corpus.
            Requires learned structure. Trained model should climb with depth.

  echo      lens top-1 == the token AT this position.
            Pure residual passthrough of the embedding. Needs no learning at all.

  TRAINED:  next_acc climbs, echo stays low   -> the lens reads computation.
  RANDOM:   next_acc at floor, echo high      -> the lens reads the architecture.
                                                 Coherent-looking, and empty.
"""

import argparse, json, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

MODEL = "Qwen/Qwen3-0.6B"
DEV = "cuda"
DTYPE = torch.bfloat16


def load(randomize_blocks: bool):
    """Trained model, or one whose BLOCKS are random but whose embedding /
    final norm / unembedding are the trained ones.

    NB: `model._init_weights(mod)` is a no-op on an already-initialized model in
    transformers v5 -- it silently leaves the weights trained. Build a fresh
    random model from the config instead and transplant the trained pieces in.
    """
    trained = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE)
    if not randomize_blocks:
        return trained.to(DEV).eval(), AutoTokenizer.from_pretrained(MODEL)

    rand = AutoModelForCausalLM.from_config(trained.config).to(DTYPE)
    ref = trained.model.layers[0].self_attn.q_proj.weight
    got = rand.model.layers[0].self_attn.q_proj.weight
    assert not torch.equal(ref, got), "blocks were not randomized"

    rand.model.embed_tokens.load_state_dict(trained.model.embed_tokens.state_dict())
    rand.model.norm.load_state_dict(trained.model.norm.state_dict())
    rand.lm_head.load_state_dict(trained.lm_head.state_dict())
    assert torch.equal(
        rand.model.embed_tokens.weight, trained.model.embed_tokens.weight
    ), "embedding transplant failed"

    del trained
    return rand.to(DEV).eval(), AutoTokenizer.from_pretrained(MODEL)


def corpus(tok, n_fit, n_eval, seq_len=128):
    from datasets import load_dataset
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    out, it = [], iter(ds)
    while len(out) < n_fit + n_eval:
        t = next(it)["text"].strip()
        if len(t) < 400 or t.startswith("="):
            continue
        ids = tok(t, add_special_tokens=False)["input_ids"][:seq_len]
        if len(ids) == seq_len:
            out.append(tok.decode(ids))
    return out[:n_fit], out[n_fit:]


@torch.no_grad()
def score(lens, model, tok, prompts, layers, skip=8, use_jacobian=True):
    """next_acc and echo per layer, over all scored positions.

    use_jacobian=False skips the J_l transport -> vanilla logit lens.
    """
    hit_next = {l: 0 for l in layers}
    hit_echo = {l: 0 for l in layers}
    total = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"][0]
        readouts, *_ = lens.apply(
            model, p, layers=layers, max_seq_len=len(ids), use_jacobian=use_jacobian
        )
        for l in layers:
            logits = readouts[l]                      # [pos, vocab]
            top1 = logits.argmax(-1).cpu()            # [pos]
            n = min(len(top1), len(ids) - 1)
            for i in range(skip, n):
                if top1[i].item() == ids[i + 1].item():
                    hit_next[l] += 1
                if top1[i].item() == ids[i].item():
                    hit_echo[l] += 1
        total += max(0, min(len(ids) - 1, len(top1)) - skip)
    return (
        {l: hit_next[l] / max(total, 1) for l in layers},
        {l: hit_echo[l] / max(total, 1) for l in layers},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fit", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    outdir = pathlib.Path(a.out); outdir.mkdir(exist_ok=True)
    report = {}

    for cond, rand in (("trained", False), ("random_blocks", True)):
        print(f"\n{'='*60}\n{cond}\n{'='*60}", flush=True)
        hf, tok = load(rand)
        model = jlens.from_hf(hf, tok)
        fit_p, eval_p = corpus(tok, a.n_fit, a.n_eval)

        lens = jlens.fit(
            model, fit_p,
            checkpoint_path=str(outdir / f"ckpt_{cond}.pt"),
        )
        lens.save(str(outdir / f"lens_{cond}.pt"))

        layers = sorted(lens.jacobians.keys())
        nxt, echo = score(lens, model, tok, eval_p, layers)
        report[cond] = {"layers": layers, "next_acc": nxt, "echo": echo}

        print(f"\n{'layer':>6} {'next_acc':>10} {'echo':>10}")
        for l in layers:
            print(f"{l:>6} {nxt[l]:>10.4f} {echo[l]:>10.4f}", flush=True)

        del hf, model, lens
        torch.cuda.empty_cache()

    (outdir / "randomization_control.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    t, r = report["trained"], report["random_blocks"]
    ls = t["layers"]
    tn, rn = max(t["next_acc"].values()), max(r["next_acc"].values())
    te, re_ = max(t["echo"].values()), max(r["echo"].values())
    print(f"peak next_acc  trained={tn:.4f}  random={rn:.4f}")
    print(f"peak echo      trained={te:.4f}  random={re_:.4f}")
    print()
    if rn < 0.02 and re_ > 0.2:
        print("Lens on random blocks predicts nothing but ECHOES the input.")
        print("-> Coherent-looking readouts survive with zero learned structure.")
        print("-> Trask's objection has real force; J-lens output must be scored")
        print("   against this floor, not against uniform-random chance.")
    elif rn < 0.02:
        print("Lens on random blocks is empty on both metrics.")
        print("-> J-lens is sensitive to learned structure. It PASSES the")
        print("   randomization sanity check. Trask's objection does not bite.")
    else:
        print("Lens on random blocks predicts real next tokens. Investigate:")
        print("-> likely leakage (embed/unembed tying doing the work).")


if __name__ == "__main__":
    main()
