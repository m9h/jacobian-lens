"""What IS the dominant output direction?  Companion to final_block_control.py.

For each arm, take the top left-singular (output-side) vectors of J_0 and J_30 and of
dJ_0 = J_0(instruct) - J_0(base), push them through the arm's own final RMSNorm scale and
unembedding, and print the tokens at both ends.  Brauer found a Chinese-vs-Latin direction on
DeepSeek-V3; this asks what OLMo-3-7B's is, and whether post-training's layer-0 movement is a
readable direction or not.

Weights are read from raw safetensors byte ranges (see scratchpad range_fetch.sh): a bf16
[vocab, d] lm_head and a bf16 [d] norm weight per arm.  No model forward is run.
"""
from __future__ import annotations
import argparse, json
import torch
from tokenizers import Tokenizer


def load_lens(p):
    st = torch.load(p, map_location="cpu", weights_only=False)
    J = st.get("J") or st.get("jacobians")
    return {int(k): v.double() for k, v in J.items()}


def bf16_bin(path, shape):
    buf = open(path, "rb").read()
    return torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(*shape).float()


def top_vecs(A, k=3):
    U, S, V = torch.svd_lowrank(A, q=k + 24, niter=6)
    return U[:, :k].float(), S[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True, help="dir with <arm>.lm_head.weight.bin / .model.norm.weight.bin / tokenizer.json")
    ap.add_argument("--base-lens", required=True); ap.add_argument("--instruct-lens", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--topn", type=int, default=25)
    a = ap.parse_args()
    tok = Tokenizer.from_file(f"{a.heads}/tokenizer.json")
    arms = {"base": ("Olmo-3-1025-7B", a.base_lens), "instruct": ("Olmo-3-7B-Instruct", a.instruct_lens)}
    lens = {n: load_lens(p) for n, (_, p) in arms.items()}
    heads = {}
    for n, (m, _) in arms.items():
        W = bf16_bin(f"{a.heads}/{m}.lm_head.weight.bin", (100278, 4096))
        g = bf16_bin(f"{a.heads}/{m}.model.norm.weight.bin", (4096,))
        heads[n] = (W, g)

    def unembed(n, u):
        W, g = heads[n]
        logits = W @ (u * g)                     # RMSNorm scale then unembed; norm's 1/rms is a scalar
        top = torch.topk(logits, a.topn).indices.tolist(); bot = torch.topk(-logits, a.topn).indices.tolist()
        return [tok.decode([i]) for i in top], [tok.decode([i]) for i in bot]

    res = {}
    targets = {
        "J0_base": ("base", lens["base"][0]), "J0_instruct": ("instruct", lens["instruct"][0]),
        "J30_base": ("base", lens["base"][30]), "J30_instruct": ("instruct", lens["instruct"][30]),
        "dJ0_via_base_head": ("base", lens["instruct"][0] - lens["base"][0]),
        "dJ0_via_instruct_head": ("instruct", lens["instruct"][0] - lens["base"][0]),
        "dJ30_via_base_head": ("base", lens["instruct"][30] - lens["base"][30]),
    }
    for name, (arm, A) in targets.items():
        U, S = top_vecs(A, 3)
        share = (S.pow(2) / A.pow(2).sum()).tolist()
        res[name] = {"sv_share_top3": share, "dirs": []}
        for j in range(3):
            pos, neg = unembed(arm, U[:, j])
            res[name]["dirs"].append({"pos": pos, "neg": neg})
            print(f"{name} u{j+1} (share {share[j]:.3f})\n  +: {' | '.join(pos[:15])}\n  -: {' | '.join(neg[:15])}", flush=True)
    # cross-arm alignment of the layer-0 and layer-30 top directions
    Ub0, _ = top_vecs(lens["base"][0], 1); Ui0, _ = top_vecs(lens["instruct"][0], 1)
    Ub30, _ = top_vecs(lens["base"][30], 1); Ui30, _ = top_vecs(lens["instruct"][30], 1)
    res["alignment"] = {"u1_J0_base_vs_instruct": abs(float(Ub0[:, 0] @ Ui0[:, 0])),
                        "u1_J30_base_vs_instruct": abs(float(Ub30[:, 0] @ Ui30[:, 0])),
                        "u1_J0_base_vs_J30_base": abs(float(Ub0[:, 0] @ Ub30[:, 0])),
                        "u1_J0_instruct_vs_J30_instruct": abs(float(Ui0[:, 0] @ Ui30[:, 0]))}
    print(json.dumps(res["alignment"], indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
