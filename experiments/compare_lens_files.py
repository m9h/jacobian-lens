"""Per-layer float64 comparison of two or more saved Jacobian-lens files.

    python experiments/compare_lens_files.py a.pt b.pt [c.pt ...] [--out table.json]

Reads the Neuronpedia / anthropics-jlens layout ({"J": {layer: [d, d]}, ...}) and the
older {"jacobians": ...} layout. For every pair and every shared layer it reports the
flattened cosine, the relative Frobenius error, the cosine of the off-identity parts
(J - I), and each file's identity distance ||J - I|| / ||I||. Everything is computed in
float64 (see spinning-up-in-mech-interp PITFALLS #28: float32 cosines on d_model^2
vectors overshoot 1 by up to 0.006).

Used 2026-10-15 for results/jpp_vs_jlens_qwen3-4b.json (Neuronpedia's J-lens vs J++ lens
of the same model) and results/rope_refit_tf513/fourway_f64.json (our two OLMo-3 base
lenses vs Neuronpedia's withdrawn and current ones).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import torch


def load_maps(path: str) -> tuple[dict[int, torch.Tensor], dict]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    J = st.get("J") or st.get("jacobians")
    if J is None:
        raise ValueError(f"{path}: no 'J' or 'jacobians' key (keys={list(st)})")
    meta = {k: v for k, v in st.items() if k not in ("J", "jacobians")}
    return {int(k): v for k, v in J.items()}, meta


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    args = ap.parse_args()

    names = [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    maps, metas = {}, {}
    for n, f in zip(names, args.files):
        maps[n], metas[n] = load_maps(f)
        print(f"{n}: {len(maps[n])} layers, n_prompts={metas[n].get('n_prompts')}, "
              f"provenance={'yes' if 'provenance' in metas[n] else 'no'}")

    layers = sorted(set.intersection(*(set(m) for m in maps.values())))
    pairs = list(itertools.combinations(names, 2))
    rows = []
    hdr = f"{'L':>3} " + " ".join(f"{a[:7]}|{b[:7]:>7}" for a, b in pairs) + "   id: " + " ".join(names)
    print(hdr)
    for l in layers:
        M = {n: maps[n][l].to(torch.float64) for n in names}
        I = torch.eye(M[names[0]].shape[-1], dtype=torch.float64)
        row = {"layer": l}
        cells = []
        for a, b in pairs:
            row[f"cos_{a}__{b}"] = cos(M[a], M[b])
            row[f"rel_{a}__{b}"] = ((M[a] - M[b]).norm() / M[a].norm()).item()
            row[f"cos_offidentity_{a}__{b}"] = cos(M[a] - I, M[b] - I)
            cells.append(f"{row[f'cos_{a}__{b}']:15.5f}")
        ids = []
        for n in names:
            row[f"id_{n}"] = ((M[n] - I).norm() / I.norm()).item()
            ids.append(f"{row[f'id_{n}']:.4f}")
        print(f"{l:3d} " + " ".join(cells) + "   " + " ".join(ids))
        rows.append(row)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
