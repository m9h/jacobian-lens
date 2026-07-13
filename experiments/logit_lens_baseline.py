"""Does the Jacobian earn its keep? J-lens vs. a plain logit lens.

The randomization control (experiments/randomization_control.py) killed the strong
form of the deflationary objection: on random blocks the J-lens reads out nothing,
so its structure requires learned weights.

It did NOT test the better form of that objection, which is about the TRAINED model:

    a J-lens coheres across layers for the same reason a logit lens does -- every
    block writes additively into one residual stream, and the unembedding reads
    that same space. J_l = E[dh_final/dh_l] is an averaged linear map from layer l
    into the final residual basis. That is an analytically-derived tuned lens.

So: hold the model, the lens, the prompts and the metric fixed, and toggle only the
transport. `apply(..., use_jacobian=False)` skips `J_l` and unembeds the raw residual
-- i.e. the vanilla logit lens.

    J-lens >> logit lens   -> the Jacobian is doing real work.
    J-lens ~= logit lens   -> `J_l` adds little the residual stream did not already
                              hand you, and "J-space" is a logit lens with a better
                              name. Every J-lens figure then needs THIS floor, not
                              chance, as its baseline.
"""

import json, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

from randomization_control import MODEL, DEV, DTYPE, corpus, score

OUT = pathlib.Path("results")


def main():
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)

    # Same corpus call, same seed-free deterministic stream -> same held-out set.
    _, eval_p = corpus(tok, 100, 32)
    lens = jlens.JacobianLens.load(str(OUT / "lens_trained.pt"))
    layers = sorted(lens.jacobians.keys())

    report = {}
    for name, use_j in (("j_lens", True), ("logit_lens", False)):
        nxt, echo = score(lens, model, tok, eval_p, layers, use_jacobian=use_j)
        report[name] = {"next_acc": nxt, "echo": echo}
        print(f"\n{name}: peak next_acc={max(nxt.values()):.4f} "
              f"peak echo={max(echo.values()):.4f}", flush=True)

    j, l = report["j_lens"], report["logit_lens"]
    print(f"\n{'layer':>5} | {'J next':>8} {'LOGIT next':>11} {'delta':>8} |"
          f" {'J echo':>8} {'LOGIT echo':>11}")
    print("-" * 64)
    for ly in layers:
        k = str(ly) if str(ly) in j["next_acc"] else ly
        dn = j["next_acc"][k] - l["next_acc"][k]
        print(f"{ly:>5} | {j['next_acc'][k]:>8.4f} {l['next_acc'][k]:>11.4f} "
              f"{dn:>+8.4f} | {j['echo'][k]:>8.4f} {l['echo'][k]:>11.4f}")

    jp, lp = max(j["next_acc"].values()), max(l["next_acc"].values())
    (OUT / "logit_lens_baseline.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 64 + "\nVERDICT\n" + "=" * 64)
    print(f"peak next_acc   J-lens={jp:.4f}   logit-lens={lp:.4f}   "
          f"ratio={jp / max(lp, 1e-9):.2f}x")
    if lp < 1e-9:
        print("Logit lens is at floor. The Jacobian transport is doing ALL the work.")
    elif jp > 1.5 * lp:
        print("J-lens substantially beats the logit lens. The Jacobian earns its keep;")
        print("the readout is not reducible to residual-stream passthrough.")
    elif jp > 1.1 * lp:
        print("J-lens modestly beats the logit lens. Real but incremental --")
        print("report J-space results against the LOGIT-LENS floor, not chance.")
    else:
        print("J-lens ~= logit lens. `J_l` adds little over unembedding the raw")
        print("residual. 'J-space' is then largely the logit-lens phenomenon, and")
        print("Trask's residual-stream objection has force after all.")


if __name__ == "__main__":
    main()
