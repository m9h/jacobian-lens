"""Final-block inherited-direction control (Brauer, "J-lens shouldn't target the final layer by
default", LessWrong Adb55vLqt33LEyBfz, 25 Sep 2026), applied to a post-training PAIR.

Brauer's finding: with target = final layer, one output direction written by the last block can
be the top Jacobian direction at EVERY earlier layer (DeepSeek-V3: a Chinese-vs-Latin direction).
Our flagship number, cos(J_base, J_instruct) low at early layers, is also a final-target
statistic.  If post-training mostly rewrites the last block's dominant output direction, every
earlier layer's Jacobian inherits that change and "early layers move most" is partly an artifact
of the target, not of the early layers.

This script needs no model and no GPU.  For two lens files (same layout as the Hub /
jlens files: {"J": {layer: d x d}, ...}) it computes, in float64:

  1. spectrum: top-k singular values of J_l per arm; share of ||J_l||_F^2 in the top-1 / top-k
     left-singular (output-side) directions
  2. inheritance: |cos(u1(J_l), u1(J_30))| per arm -- does the final transport's top output
     direction persist as the top direction at every layer?
  3. movement: for dJ_l = J_l(instruct) - J_l(base): share of ||dJ_l||^2 in its own top-1/top-k
     directions, and the share captured by the top-k output directions of J_30 (both arms)
  4. the projected-out cosine: remove the top-k output directions of J_30 (union over arms) from
     both J_l and recompute cos(base, instruct).  If cos at L0 jumps from 0.25 toward the late-
     layer values, the movement was inherited from the target side.

Usage: python experiments/final_block_control.py BASE.pt INSTRUCT.pt --out results/x.json
"""
from __future__ import annotations
import argparse, json, time
import torch

torch.set_num_threads(max(1, torch.get_num_threads()))


def load(p):
    st = torch.load(p, map_location="cpu", weights_only=False)
    J = st.get("J") or st.get("jacobians")
    return {int(k): v.double() for k, v in J.items()}


def topk_svd(A, k, q_extra=24, niter=6):
    U, S, V = torch.svd_lowrank(A, q=k + q_extra, niter=niter)
    return U[:, :k], S[:k]


def cos(a, b):
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def orth_union(*Us):
    Q, _ = torch.linalg.qr(torch.cat(Us, dim=1))
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ref-layer", type=int, default=None, help="layer whose output directions are 'the last block' (default: max layer)")
    ap.add_argument("--label", default="jlens")
    a = ap.parse_args()
    t0 = time.time()
    B, I_ = load(a.base), load(a.instruct)
    layers = sorted(set(B) & set(I_))
    ref = a.ref_layer if a.ref_layer is not None else max(layers)
    k = a.k
    d = B[layers[0]].shape[-1]
    Id = torch.eye(d, dtype=torch.float64)

    # output-side directions of the reference (last) transport, per arm and union
    Ub, Sb = topk_svd(B[ref], k)
    Ui, Si = topk_svd(I_[ref], k)
    dref = I_[ref] - B[ref]
    Ud, Sd = topk_svd(dref, k)
    ref_info = {"layer": ref,
                "top_sv_share_base": [(Sb[:j].pow(2).sum() / B[ref].pow(2).sum()).item() for j in (1, 4, k)],
                "top_sv_share_instruct": [(Si[:j].pow(2).sum() / I_[ref].pow(2).sum()).item() for j in (1, 4, k)],
                "cos_u1_base_instruct": abs(cos(Ub[:, 0], Ui[:, 0])),
                "offidentity_top_sv_share_base": [(topk_svd(B[ref] - Id, k)[1][:j].pow(2).sum() / (B[ref] - Id).pow(2).sum()).item() for j in (1, 4, k)]}

    per = {}
    for l in layers:
        Jb, Ji = B[l], I_[l]
        dJ = Ji - Jb
        ub, sb = topk_svd(Jb, k); ui, si = topk_svd(Ji, k); ud, sd = topk_svd(dJ, k)
        nb2, ni2, nd2 = Jb.pow(2).sum(), Ji.pow(2).sum(), dJ.pow(2).sum()
        row = {
            "cos_base_instruct": cos(Jb, Ji),
            "cos_offidentity": cos(Jb - Id, Ji - Id),
            "sv_share_base": [(sb[:j].pow(2).sum() / nb2).item() for j in (1, 4, k)],
            "sv_share_instruct": [(si[:j].pow(2).sum() / ni2).item() for j in (1, 4, k)],
            "dJ_sv_share": [(sd[:j].pow(2).sum() / nd2).item() for j in (1, 4, k)],
            "inherit_u1_base": abs(cos(ub[:, 0], Ub[:, 0])),
            "inherit_u1_instruct": abs(cos(ui[:, 0], Ui[:, 0])),
            "dJ_u1_vs_dref_u1": abs(cos(ud[:, 0], Ud[:, 0])),
            "dJ_u1_vs_ref_u1_base": abs(cos(ud[:, 0], Ub[:, 0])),
            "dJ_u1_vs_ref_u1_instruct": abs(cos(ud[:, 0], Ui[:, 0])),
        }
        # how much of the movement lives in the last block's top-k output directions?
        for kk in (1, 4, k):
            Q = orth_union(Ub[:, :kk], Ui[:, :kk])          # <= 2kk dims
            P = Q @ (Q.T @ dJ)
            row[f"dJ_share_in_ref_top{kk}"] = (P.pow(2).sum() / nd2).item()
            # projected-out cosine (remove those output directions from both arms)
            Jb_p = Jb - Q @ (Q.T @ Jb); Ji_p = Ji - Q @ (Q.T @ Ji)
            row[f"cos_projected_out_ref_top{kk}"] = cos(Jb_p, Ji_p)
            row[f"cos_offidentity_projected_out_ref_top{kk}"] = cos(Jb_p - (Id - Q @ Q.T), Ji_p - (Id - Q @ Q.T))
            # and removing the top directions of the CHANGE at the reference layer
            Qd = Ud[:, :kk]
            Jb_q = Jb - Qd @ (Qd.T @ Jb); Ji_q = Ji - Qd @ (Qd.T @ Ji)
            row[f"cos_projected_out_dref_top{kk}"] = cos(Jb_q, Ji_q)
            row[f"dJ_share_in_dref_top{kk}"] = ((Qd @ (Qd.T @ dJ)).pow(2).sum() / nd2).item()
        per[str(l)] = row
        print(f"L{l:2d} cos {row['cos_base_instruct']:.4f} | sv1 share b/i {row['sv_share_base'][0]:.3f}/{row['sv_share_instruct'][0]:.3f} "
              f"| inherit u1 b/i {row['inherit_u1_base']:.3f}/{row['inherit_u1_instruct']:.3f} | dJ sv1 share {row['dJ_sv_share'][0]:.3f} "
              f"| dJ in ref top1/4/{k}: {row['dJ_share_in_ref_top1']:.3f}/{row['dJ_share_in_ref_top4']:.3f}/{row[f'dJ_share_in_ref_top{k}']:.3f} "
              f"| cos proj-out top1/4/{k}: {row['cos_projected_out_ref_top1']:.4f}/{row['cos_projected_out_ref_top4']:.4f}/{row[f'cos_projected_out_ref_top{k}']:.4f}", flush=True)

    res = {"label": a.label, "base": a.base, "instruct": a.instruct, "layers": layers, "k": k,
           "reference": ref_info, "per_layer": per, "seconds": time.time() - t0}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
