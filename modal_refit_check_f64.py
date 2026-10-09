"""Float64 recomputation of the lens comparisons from modal_olmo_refit_tf513.py.

The float32 cosines printed by that run (and by jlens_lab.artifacts.compare) exceed 1.0 at
deep layers, which a cosine cannot. Jacobians here are ~identity-dominated (identity_distance
0.22) so the flattened 4096^2 dot product mixes 4096 O(1) terms with 16.7M tiny ones; this
recomputes every pairwise statistic in float64 so the erratum can quote numbers that are not
precision artefacts. CPU only.

    modal run modal_refit_check_f64.py
"""
import modal
# Definitions copied verbatim from modal_olmo_refit_tf513.py: Modal mounts only the entry
# script, and an identical image definition reuses the built image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.6",
        # OLMo-3 (Olmo3ForCausalLM, YaRN RoPE) and Ministral-3
        # (Mistral3ForConditionalGeneration) only exist in transformers 5.x -- 4.57
        # predates the OLMo-3 release. This app is pure torch (no Flax), so the old
        # "<5 keeps the Flax classes" rationale does not apply. Verified locally on
        # transformers 5.9.0 / torch 2.12: both classes import. The anchor gate is what
        # confirms the weights (and YaRN RoPE) actually load correctly under this pin.
        "transformers>=5.13,<6",   # RoPE/YaRN per-layer fix
        "accelerate", "safetensors", "datasets", "numpy", "huggingface_hub",
    )
    .run_commands(
        "pip install git+https://github.com/m9h/jacobian-lens.git",
        "pip install git+https://github.com/m9h/jlens-lab.git",
    )
)

cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("jlens-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"}

BASE = "allenai/Olmo-3-1025-7B"
ANCHOR_ID = "olmo-3-1025-7b"          # Neuronpedia id of the published lens


def _slug(arm: str) -> str:
    return arm.split("/")[-1].lower()

app = modal.App("olmo-refit-check-f64")


@app.function(image=image, volumes={"/cache": cache, "/out": out}, timeout=30 * 60,
              env=ENV, memory=24 * 1024, cpu=4)
def check() -> dict:
    import json, pathlib, torch
    from huggingface_hub import HfApi, hf_hub_download
    from jlens_lab import artifacts

    lenses = {
        "ours_buggy": artifacts.load_lens(f"/out/olmo_ladder/{_slug(BASE)}/lens.pt"),
        "ours_tf513": artifacts.load_lens(f"/out/olmo_ladder_tf513/{_slug(BASE)}/lens.pt"),
    }
    files = [f for f in HfApi().list_repo_files(artifacts.REPO) if f.startswith(ANCHOR_ID + "/")]
    pts = [f for f in files if f.endswith(".pt")]
    lenses["neuronpedia"] = artifacts.load_lens(hf_hub_download(artifacts.REPO, pts[0]))

    def j(lens, l):
        return lens.jacobians[l].to(torch.float64)

    meta = {k: {"n_prompts": int(v.n_prompts), "layers": sorted(int(x) for x in v.jacobians),
                "dtype": str(next(iter(v.jacobians.values())).dtype)} for k, v in lenses.items()}
    shared = sorted(set.intersection(*[set(v.jacobians) for v in lenses.values()]))
    pairs = [("ours_buggy", "ours_tf513"), ("ours_buggy", "neuronpedia"), ("ours_tf513", "neuronpedia")]
    res = {}
    for a, b in pairs:
        per = {}
        for l in shared:
            x, y = j(lenses[a], l).flatten(), j(lenses[b], l).flatten()
            cos32 = torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=0).item()
            per[int(l)] = {"cos64": (x @ y / (x.norm() * y.norm())).item(),
                           "cos32_repro": cos32,
                           "rel_err64": ((x - y).norm() / y.norm()).item()}
        res[f"{a}__vs__{b}"] = per
    ident = {}
    for k, v in lenses.items():
        l = max(v.jacobians)                                  # last source layer, as jlens_lab
        J = j(v, l); I = torch.eye(J.shape[0], dtype=J.dtype)
        ident[k] = {"layer": int(l), "identity_distance64": ((J - I).norm() / I.norm()).item(),
                    "identity_distance32": artifacts.identity_distance(v),
                    "max_abs": J.abs().max().item(), "diag_mean": J.diag().mean().item()}
    r = {"meta": meta, "shared_layers": [int(l) for l in shared], "pairs": res, "identity": ident}
    pathlib.Path("/out/olmo_ladder_tf513/check_f64.json").write_text(json.dumps(r, indent=2))
    out.commit()
    return r


@app.local_entrypoint()
def main():
    import json
    r = check.remote()
    print("META", json.dumps(r["meta"]))
    print("IDENT", json.dumps(r["identity"]))
    for pair, per in r["pairs"].items():
        print("PAIR", pair)
        for l, d in per.items():
            print(f"  L{int(l):>2}  cos64 {d['cos64']:.5f}  cos32 {d['cos32_repro']:.5f}  rel64 {d['rel_err64']:.4f}")
