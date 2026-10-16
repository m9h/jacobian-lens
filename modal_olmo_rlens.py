"""R-lens (LRP-backward Jacobian lens) refit of the two ladder endpoints, 2026-10-15.

Estimator control for the post-training ladder (see notes/jpp_lens_2026-10.md section 5): the
J-lens and the J++/R-lens disagree on the SAME model at early layers (cos 0.26 at L0 on
Qwen3-4B), so cos(base, Instruct) = 0.69 must be shown under the other estimator before it
is a claim about the model rather than about the J-lens.

What this fits: the pooled LRP Jacobian -- Blank, Bhatia & Nanda's R-lens -- through the
safety-research/jpp_lens code (Ayonrinde & Lindsey 2026, Apache-2.0): ``LensTrainer`` (their
copy of Anthropic's estimator: one-hot cotangent at every kept position, mean over kept
positions, mean over prompts) with the forward pass run under ``apply_lrp_rules_for_mode``
(LN-rule on the residual RMSNorms, identity + half rules on the gated MLP). No routers, no
expert weights, no labelled items. ``lrp_mode="none"`` through the same class is a plain
J-lens and is used as the equivalence check against our own pipeline.

Protocol held fixed to the ladder: same 616 WikiText-103 prompts (``_prompts``), layers
0,3,...,30, target = final block, 128 rows per backward pass, max_seq_len 128, skip 16,
bf16, B200. transformers is pinned to 5.14.1 (jpp_lens's pin; has the OLMo-3 per-layer
RoPE fix; the version their Olmo3MLP/Olmo3RMSNorm patchers were verified on).

Stages (each a local entrypoint):
  precheck   base model, 8 prompts, lrp_mode none vs rlens -> timing, memory, and the
             equivalence numbers (prompt-0 J30 identity distance must be 0.2429 under
             "none", as in results/rope_refit_tf513/rope_emulation.json)
  run_rlens  8 shards x 2 arms under rlens -> combine -> analyze
"""

from __future__ import annotations

import modal

APP = "olmo-rlens-estimator-control"
app = modal.App(APP)

JPP_COMMIT = "93f02cd6c27db765abb127f20eff656bb5d27340"   # safety-research/jpp_lens main, 2026-10-08

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.6",
        "transformers==5.14.1",     # jpp_lens's pin; OLMo-3 RoPE fix present (>=5.13)
        "accelerate", "datasets", "huggingface_hub", "numpy", "pandas", "pyarrow",
    )
    .run_commands(
        # --no-deps: the pyproject pins exact versions of everything and a torch index extra.
        f"pip install --no-deps git+https://github.com/safety-research/jpp_lens@{JPP_COMMIT}",
    )
)

cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("jlens-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"}

BASE = "allenai/Olmo-3-1025-7B"
INSTRUCT = "allenai/Olmo-3-7B-Instruct"
ARMS = [BASE, INSTRUCT]

N_PROMPTS = 616
N_SHARDS = 8
LAYER_STEP = 3
DIM_BATCH = 128          # jacobian_rows_per_pass; the ladder's dim_batch
MAX_SEQ_LEN = 128
SKIP_FIRST = 16
OUT_ROOT = "/out/olmo_rlens"
JLENS_ROOT = "/out/olmo_ladder"      # the July J-lenses (transformers 5.14.1, fixed RoPE)


def _slug(arm: str) -> str:
    return arm.split("/")[-1].lower()


def _prompts(tok, n: int) -> list[str]:
    """Verbatim from modal_olmo_ladder.py / modal_olmo_refit_tf513.py: the ladder's prompt set."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    prompts, it = [], iter(ds)
    while len(prompts) < n:
        t = next(it)["text"].strip()[:2000]
        if (len(t) >= 400 and not t.startswith("=")
                and len(tok(t, add_special_tokens=False)["input_ids"]) >= MAX_SEQ_LEN):
            prompts.append(t)
    return prompts


def _make_trainer_class():
    """The pooled-LRP trainer: jpp_lens's LensTrainer with the forward run under the LRP
    surgery (what ExpertJacobianRelPTrainer does for the expert trainer). Defined inside a
    function because the imports only exist in the image."""
    from workspace_lens.fitting.jacobian_fitting import LensTrainer
    from workspace_lens.lrp import apply_lrp_rules_for_mode
    from workspace_lens.lrp.lrp import hf_module_to_patch

    class RelPLensTrainer(LensTrainer):
        applies_lrp_rules = True

        def __init__(self, config, model, prompts, **kw):
            super().__init__(config, model, prompts, **kw)
            hf_module_to_patch(model)        # fail at construction, not at the first prompt
            self.last_patch_report = None

        def _run_fit_forward(self, prompt):
            with apply_lrp_rules_for_mode(self.model, self.config.lrp_mode) as report:
                state = super()._run_fit_forward(prompt)
            if report is not None:
                self.last_patch_report = report
            return state

        def write_checkpoint(self, *, final: bool = False) -> None:
            return None                      # shards are saved by the caller, to the volume

    return RelPLensTrainer


def _config(arm: str, layers: list[int], lrp_mode: str, name: str):
    from workspace_lens.config import LensConfig
    return LensConfig(
        hf_model_name=arm, checkpoint_name=name, source_layers=layers,
        relative_end_transport_layer=-1, jacobian_rows_per_pass=DIM_BATCH,
        max_seq_len=MAX_SEQ_LEN, skip_first_n_positions=SKIP_FIRST,
        checkpoint_every_n_prompts=None, lrp_mode=lrp_mode,
    )


def _provenance(model, lrp_mode: str, trainer, n: int, layers: list[int]) -> dict:
    import datetime as dt
    import platform
    import torch
    import transformers
    from workspace_lens.lrp import lrp_rule_config_for_mode

    rep = getattr(trainer, "last_patch_report", None)
    patched = rep.patched_names() if rep is not None else None
    hf = getattr(model, "_hf_model", None)
    return {
        "fitted_by": "m9h/jacobian-lens modal_olmo_rlens.py (jpp_lens LensTrainer + LRP surgery)",
        "estimator": "pooled Jacobian, mean over kept positions then over prompts (jpp_lens LensTrainer)",
        "lrp_mode": lrp_mode,
        "lrp_rules": lrp_rule_config_for_mode(lrp_mode).to_dict(),
        "lrp_patched_module_counts": ({k: len(v) for k, v in patched.items()} if patched else None),
        "jpp_lens_commit": JPP_COMMIT,
        "transformers": str(transformers.__version__),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_id": getattr(model, "hf_model_name", None) or getattr(getattr(hf, "config", None), "_name_or_path", None),
        "model_commit": getattr(getattr(hf, "config", None), "_commit_hash", None),
        "model_dtype": "bfloat16",
        "n_prompts": n,
        "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train, streamed; filter: len>=400 chars, "
                   "not a heading, >=128 tokens (modal_olmo_ladder.py _prompts)",
        "source_layers": layers,
        "target_layer": "final block",
        "jacobian_rows_per_pass": DIM_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "skip_first": SKIP_FIRST,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


@app.function(image=image, gpu="B200", volumes={"/cache": cache, "/out": out},
              timeout=60 * 60, env=ENV)
def precheck(n_prompts: int = 8) -> dict:
    """Base model, the first n prompts, both modes. Returns the equivalence numbers and timing."""
    import json, pathlib, time, torch
    from workspace_lens import get_hf_model

    Trainer = _make_trainer_class()
    model = get_hf_model(BASE)
    tok = model.tokenizer
    prompts = _prompts(tok, n_prompts)
    layers = list(range(0, model.n_layers - 1, LAYER_STEP))
    res = {"n_prompts": n_prompts, "layers": layers, "modes": {}}
    means = {}
    for mode in ("none", "rlens"):
        trainer = Trainer(_config(BASE, layers, mode, f"precheck_{mode}"), model, prompts)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        per_prompt_id30 = []
        acc = None
        for i, p in enumerate(prompts):
            J, seq_len, n_pos = trainer.fit_step(p, prompt_idx=i)
            I = torch.eye(J[30].shape[-1], dtype=torch.float64)
            per_prompt_id30.append(((J[30].double() - I).norm() / I.norm()).item())
            if acc is None:
                acc = {l: J[l].double().clone() for l in layers}
            else:
                for l in layers:
                    acc[l] += J[l].double()
        secs = (time.perf_counter() - t0) / n_prompts
        means[mode] = {l: acc[l] / n_prompts for l in layers}
        I = torch.eye(means[mode][0].shape[-1], dtype=torch.float64)
        rep = trainer.last_patch_report
        res["modes"][mode] = {
            "sec_per_prompt": secs,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "seq_len": seq_len, "n_source_positions": n_pos,
            "per_prompt_id30": per_prompt_id30,
            "identity_distance_mean8": {str(l): ((means[mode][l] - I).norm() / I.norm()).item() for l in layers},
            "lrp_patched": ({k: len(v) for k, v in rep.patched_names().items()} if rep is not None else None),
        }
    I = torch.eye(means["none"][0].shape[-1], dtype=torch.float64)
    res["none_vs_rlens"] = {}
    for l in layers:
        a, b = means["none"][l], means["rlens"][l]
        res["none_vs_rlens"][str(l)] = {
            "cos64": ((a * b).sum() / (a.norm() * b.norm())).item(),
            "rel64": ((a - b).norm() / a.norm()).item(),
            "cos_offidentity": (((a - I) * (b - I)).sum() / ((a - I).norm() * (b - I).norm())).item(),
        }
    res["expected_from_our_pipeline"] = {
        "prompt0_id30_fixed": 0.24290379623893954,     # modal_loadpath_test / rope_emulation
        "mean8_id30_fixed": 0.2368,                     # rope_emulation.json 'fixed'
    }
    p = pathlib.Path(f"{OUT_ROOT}/precheck.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    out.commit()
    return res


@app.function(image=image, gpu="B200", volumes={"/cache": cache, "/out": out},
              timeout=120 * 60, env=ENV, retries=1)
def fit_shard(arm: str, shard: int, lrp_mode: str = "rlens") -> dict:
    """One shard's mean Jacobian under ``lrp_mode`` on prompts[shard::N_SHARDS]."""
    import logging, pathlib, torch
    from workspace_lens import get_hf_model

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    Trainer = _make_trainer_class()
    model = get_hf_model(arm)
    tok = model.tokenizer
    layers = list(range(0, model.n_layers - 1, LAYER_STEP))
    mine = _prompts(tok, N_PROMPTS)[shard::N_SHARDS]
    trainer = Trainer(_config(arm, layers, lrp_mode, f"{lrp_mode}_{_slug(arm)}_s{shard}"), model, mine)
    lens = trainer.fit()                              # mean over this shard's prompts
    n = trainer.completed_prompt_count
    if n != len(mine):
        raise RuntimeError(f"{arm} shard {shard}: fitted {n} of {len(mine)} prompts")
    d = pathlib.Path(f"{OUT_ROOT}/{lrp_mode}/{_slug(arm)}")
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"jac": {l: lens.jacobians_L_dict_FN[l].float().cpu() for l in layers}, "n": n,
                "layers": layers, "provenance": _provenance(model, lrp_mode, trainer, n, layers)},
               d / f"shard_{shard}.pt")
    out.commit()
    return {"arm": arm, "shard": shard, "n": n, "mode": lrp_mode}


@app.function(image=image, volumes={"/out": out}, timeout=30 * 60, env=ENV)
def combine(arm: str, lrp_mode: str = "rlens") -> dict:
    """Exact weighted mean of the shard means; saves lens.pt (fp16, Neuronpedia/jlens layout,
    with provenance) and lens_fp32.pt."""
    import pathlib, torch

    d = pathlib.Path(f"{OUT_ROOT}/{lrp_mode}/{_slug(arm)}")
    shards = sorted(d.glob("shard_*.pt"))
    if len(shards) != N_SHARDS:
        raise RuntimeError(f"{d}: {len(shards)} shards, expected {N_SHARDS}")
    acc, n_tot, layers, prov = None, 0, None, None
    for s in shards:
        st = torch.load(s, map_location="cpu", weights_only=False)
        layers = st["layers"]; prov = st["provenance"]
        if acc is None:
            acc = {l: st["jac"][l].double() * st["n"] for l in layers}
        else:
            for l in layers:
                acc[l] += st["jac"][l].double() * st["n"]
        n_tot += st["n"]
    J = {l: (acc[l] / n_tot) for l in layers}
    prov = dict(prov, n_prompts=n_tot, shards=len(shards))
    base = {"n_prompts": n_tot, "source_layers": layers, "d_model": J[layers[0]].shape[-1], "provenance": prov}
    torch.save(dict(base, J={l: J[l].to(torch.float16) for l in layers}), d / "lens.pt")
    torch.save(dict(base, J={l: J[l].to(torch.float32) for l in layers}), d / "lens_fp32.pt")
    I = torch.eye(J[layers[-1]].shape[-1], dtype=torch.float64)
    out.commit()
    return {"arm": arm, "n_prompts": n_tot, "layers": layers,
            "identity_distance_last": ((J[layers[-1]] - I).norm() / I.norm()).item()}


@app.function(image=image, volumes={"/out": out}, timeout=30 * 60, env=ENV)
def analyze(lrp_mode: str = "rlens") -> dict:
    """cos(base, Instruct) per layer under the J-lens (July ladder files) and under the R-lens,
    the movement profile by layer group, and the cross-estimator distance per arm. Float64."""
    import json, pathlib, torch

    def load(p):
        st = torch.load(p, map_location="cpu", weights_only=False)
        J = st.get("J") or st.get("jacobians")
        return {int(k): v.double() for k, v in J.items()}, st.get("n_prompts")

    jl_b, n_jb = load(f"{JLENS_ROOT}/{_slug(BASE)}/lens.pt")
    jl_i, n_ji = load(f"{JLENS_ROOT}/{_slug(INSTRUCT)}/lens.pt")
    rl_b, n_rb = load(f"{OUT_ROOT}/{lrp_mode}/{_slug(BASE)}/lens_fp32.pt")
    rl_i, n_ri = load(f"{OUT_ROOT}/{lrp_mode}/{_slug(INSTRUCT)}/lens_fp32.pt")
    layers = sorted(set(jl_b) & set(jl_i) & set(rl_b) & set(rl_i))
    I = torch.eye(jl_b[layers[0]].shape[-1], dtype=torch.float64)

    def cos(a, b):
        return ((a * b).sum() / (a.norm() * b.norm())).item()

    per = {}
    move = {"jlens": {}, lrp_mode: {}}
    for l in layers:
        row = {}
        for name, (b, i) in {"jlens": (jl_b, jl_i), lrp_mode: (rl_b, rl_i)}.items():
            d = b[l] - i[l]
            row[f"cos_base_instruct_{name}"] = cos(b[l], i[l])
            row[f"rel_move_{name}"] = (d.norm() / b[l].norm()).item()
            row[f"cos_offidentity_base_instruct_{name}"] = cos(b[l] - I, i[l] - I)
            row[f"id_base_{name}"] = ((b[l] - I).norm() / I.norm()).item()
            row[f"id_instruct_{name}"] = ((i[l] - I).norm() / I.norm()).item()
            move[name][l] = (d.norm() ** 2).item()
        row["cos_estimators_base"] = cos(jl_b[l], rl_b[l])
        row["cos_estimators_instruct"] = cos(jl_i[l], rl_i[l])
        row["cos_offidentity_estimators_base"] = cos(jl_b[l] - I, rl_b[l] - I)
        per[str(l)] = row

    groups = {"L0-9": [l for l in layers if l <= 9], "L12-18": [l for l in layers if 12 <= l <= 18],
              "L21-30": [l for l in layers if l >= 21]}
    profile = {}
    for name in ("jlens", lrp_mode):
        tot = sum(move[name].values())
        profile[name] = {g: sum(move[name][l] for l in ls) / tot for g, ls in groups.items()}
    summary = {
        "mean_cos_base_instruct_jlens": sum(per[str(l)]["cos_base_instruct_jlens"] for l in layers) / len(layers),
        f"mean_cos_base_instruct_{lrp_mode}": sum(per[str(l)][f"cos_base_instruct_{lrp_mode}"] for l in layers) / len(layers),
        "mean_cos_estimators_base": sum(per[str(l)]["cos_estimators_base"] for l in layers) / len(layers),
        "mean_cos_estimators_instruct": sum(per[str(l)]["cos_estimators_instruct"] for l in layers) / len(layers),
        "movement_share_by_group": profile,
        "n_prompts": {"jlens_base": n_jb, "jlens_instruct": n_ji, f"{lrp_mode}_base": n_rb, f"{lrp_mode}_instruct": n_ri},
    }
    res = {"layers": layers, "per_layer": per, "summary": summary}
    pathlib.Path(f"{OUT_ROOT}/analysis_{lrp_mode}.json").write_text(json.dumps(res, indent=1))
    out.commit()
    return res


@app.local_entrypoint()
def run_precheck(n_prompts: int = 8):
    import json
    print("PRECHECK", json.dumps(precheck.remote(n_prompts), indent=1))


@app.local_entrypoint()
def run_rlens(mode: str = "rlens"):
    import json
    for arm in ARMS:
        for r in fit_shard.starmap([(arm, s, mode) for s in range(N_SHARDS)]):
            print("SHARD", r)
        print("COMBINE", combine.remote(arm, mode))
    print("ANALYSIS", json.dumps(analyze.remote(mode), indent=1))
