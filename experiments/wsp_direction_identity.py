import json, sys, os, numpy as np
D = sys.argv[1]; OUT = sys.argv[2]
def bf16(path, shape):
    b = np.fromfile(path, dtype=np.uint16).reshape(shape)
    return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
def bf16_bits_from_f32(x):  # round-to-nearest-even like torch
    u = x.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)
meta = json.load(open(f"{D}/qwen_base/meta.json"))
P = "model.language_model."
man = json.load(open(f"{D}/p50c/manifest.json")); z = np.load(f"{D}/p50c/edit.npz")
res = {"layers": {}}
U = {}
for i, t in enumerate(man["tensors"]):
    name = t["name"]; L = name.split(".layers.")[1].split(".")[0]
    bits = np.fromfile(f"{D}/qwen_base/{name}.bin", dtype=np.uint16).reshape(t["shape"])
    e = bits.copy().reshape(-1); e[z[f"t{i}_idx"]] = z[f"t{i}_val"]; e = e.reshape(t["shape"])
    W = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    W2 = (e.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    dW = W2 - W
    cols = np.where(np.abs(dW).sum(0) > 0)[0]
    S_ = np.linalg.svd(dW[:, cols], compute_uv=True)
    u1, s, vt = S_[0][:, 0], S_[1], S_[2]
    share = (s**2 / (s**2).sum())
    # fit W' = W - alpha * u (u^T W) on the chosen columns
    proj = u1 @ W[:, cols]           # u^T w_c
    coef = u1 @ dW[:, cols]          # component of change along u
    alpha = -np.sum(coef * proj) / np.sum(proj**2)
    resid = np.linalg.norm(dW[:, cols] + alpha * np.outer(u1, proj)) / np.linalg.norm(dW[:, cols])
    # are the chosen columns the ones writing u most?
    allproj = np.abs(u1 @ W)
    rank_of_chosen = np.argsort(-allproj)
    top64 = set(rank_of_chosen[:64].tolist()); chosen = set(cols.tolist())
    # share of u in the base MLP's output: ||u^T W||^2 / ||W||_F^2 vs 1/1024 baseline
    u_share_in_W = float((allproj**2).sum() / (W**2).sum())
    # top singular directions of base W
    Wb = np.linalg.svd(W, compute_uv=True)
    cos_u_topW = [float(abs(Wb[0][:, k] @ u1)) for k in range(5)]
    U[L] = u1
    res["layers"][L] = dict(n_cols=int(len(cols)), sv_share_top3=[float(x) for x in share[:3]], s_top3=[float(x) for x in s[:3]],
        alpha_fit=float(alpha), fit_residual_rel=float(resid), chosen_in_top64_by_u_projection=len(top64 & chosen),
        chosen_cols_rank_range=[int(np.where(rank_of_chosen==c)[0][0]) for c in sorted(cols, key=lambda c: np.where(rank_of_chosen==c)[0][0])][:: 63],
        relative_change=float(np.linalg.norm(dW)/np.linalg.norm(W)), u_share_in_base_W=u_share_in_W,
        u_share_uniform_baseline=1/1024, cos_u_vs_base_W_top5_left_sv=cos_u_topW)
    print(L, json.dumps(res["layers"][L]), flush=True)
res["cos_u15_u19"] = float(abs(U["15"] @ U["19"]))
# p10c direction(s)
m10 = json.load(open(f"{D}/p10c/manifest.json")); z10 = np.load(f"{D}/p10c/edit.npz")
a_vecs = []
for i, t in enumerate(m10["tensors"]):
    if t["form"] == "lowrank":
        a = z10[f"t{i}_a"]; b = z10[f"t{i}_b"]
        a_vecs.append((t["name"], a, b))
# for each lowrank edit, direction = column of a (residual side); for down_proj/o_proj shape [1024, in], a is [1024,k]
if a_vecs:
    dirs = {}
    for name, a, b in a_vecs:
        v = a[:, 0].astype(np.float64); v /= np.linalg.norm(v); dirs[name] = v
    first = list(dirs.values())[0]
    res["p10c"] = dict(n_lowrank=len(a_vecs), ranks=sorted(set(int(a.shape[1]) for _, a, _ in a_vecs)),
        cos_p10c_first_vs_u15=float(abs(first @ U["15"])), cos_p10c_first_vs_u19=float(abs(first @ U["19"])),
        min_pairwise_cos_within_p10c=float(min(abs(dirs[x] @ dirs[y]) for x in dirs for y in dirs if x < y)),
        p10c_names=list(dirs)[:5])
    print("p10c", json.dumps(res["p10c"]), flush=True)
# unembed through tied embeddings and final norm
E = bf16(f"{D}/qwen_base/{P}embed_tokens.weight.bin", meta[P+"embed_tokens.weight"]["shape"])
g = bf16(f"{D}/qwen_base/{P}norm.weight.bin", (1024,))
tok = json.load(open(f"{D}/tokenizer.json"))["model"]["vocab"]; inv = {v: k for k, v in tok.items()}
added = {a["id"]: a["content"] for a in json.load(open(f"{D}/tokenizer.json")).get("added_tokens", [])}
def name(i): return added.get(i, inv.get(i, f"<{i}>")).replace("Ġ", " ").replace("Ċ", "\\n")
res["unembed"] = {}
for L, u in U.items():
    for lab, vec in (("raw", u), ("normed", g * u)):
        lg = E @ vec
        o = np.argsort(lg)
        res["unembed"][f"L{L}_{lab}"] = dict(plus=[name(int(i)) for i in o[::-1][:25]], minus=[name(int(i)) for i in o[:25]])
        print(f"L{L} {lab} +:", " | ".join(res["unembed"][f"L{L}_{lab}"]["plus"][:20])); print(f"L{L} {lab} -:", " | ".join(res["unembed"][f"L{L}_{lab}"]["minus"][:20]), flush=True)
# relation of u to the unembedding geometry
mu = E.mean(0); Ec = E - mu
G = Ec.T @ Ec
w, V = np.linalg.eigh(G); V = V[:, ::-1]; w = w[::-1]
res["embedding_geometry"] = {}
for L, u in U.items():
    pcs = V.T @ u
    cum = np.cumsum(pcs**2)
    res["embedding_geometry"][f"L{L}"] = dict(cos_u_vs_mean_embedding=float(abs(mu @ u)/np.linalg.norm(mu)),
        share_in_top1_pc=float(cum[0]), share_in_top4_pc=float(cum[3]), share_in_top16_pc=float(cum[15]), share_in_top64_pc=float(cum[63]),
        uniform_baseline_top16=16/1024)
    print(L, json.dumps(res["embedding_geometry"][f"L{L}"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
