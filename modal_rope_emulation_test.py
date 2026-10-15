"""RoPE-emulation test (2026-10-15): does the YaRN-on-sliding-layers bug move the Jacobian lens?

Self-contained copy of modal_olmo_refit_tf513.py's image/consts (Modal mounts only the entry
script). Original docstring follows.

RoPE-erratum refit of the OLMo-3 BASE lens under transformers>=5.13 (YaRN on the 8
full-attention layers only). See results/ROPE_ERRATUM.md and results/rope_damage_bound.md.

This is modal_olmo_ladder.py with three changes, so the comparison is protocol-matched:
  * transformers pinned >=5.13 (the fix, huggingface/transformers#46911)
  * every output path moved from /out/olmo_ladder_tf513/ to /out/olmo_ladder_tf513/ so the
    buggy fit is NOT overwritten -- it is the other half of the comparison
  * a `refit_base` entrypoint = anchor gate + env report + per-layer cosine between the
    buggy and corrected base lenses on identical prompts/layers/n (616, layer_step 3)

    modal run --detach modal_olmo_refit_tf513.py::refit_base
"""


from __future__ import annotations

import modal

APP = "olmo-rope-emulation-test"
app = modal.App(APP)

# OLMo-3 needs no mamba-ssm (that was Nemotron-H) and no arch pin, so this image is
# much lighter than modal_app.py's.
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

# All twelve arms. Ungated, all Olmo3ForCausalLM, 32 layers, d=4096, 14.6GB each.
ARMS = [
    BASE,
    "allenai/Olmo-3-7B-Instruct-SFT",
    "allenai/Olmo-3-7B-Instruct-DPO",
    "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-3-7B-Think-SFT",
    "allenai/Olmo-3-7B-Think-DPO",
    "allenai/Olmo-3-7B-Think",
    "allenai/Olmo-3-7B-RL-Zero-Math",
    "allenai/Olmo-3-7B-RL-Zero-Code",
    "allenai/Olmo-3-7B-RL-Zero-IF",
    "allenai/Olmo-3-7B-RL-Zero-General",
    # RL-Zero-Mix is DELIBERATELY EXCLUDED. Its config is model_type "olmo2-retrofit"
    # (Olmo2RetrofitForCausalLM), NOT olmo3 like every other arm -- a different base
    # architecture. Including it would confound the RL-Zero control (whose entire purpose
    # is to hold architecture + method constant and vary only DOMAIN) with an architecture
    # change, and cross-architecture J-space geometry is not comparable anyway. The four
    # olmo3 domain arms (Math/Code/IF/General) remain a valid domain-varied control.
    # Caught by the capability probe before the ladder spent on it.
]

# SECOND-FAMILY REPLICATION -- Ministral-3, a DIFFERENT base with a base->reasoning
# ladder. Tests whether the ladder's main effect reproduces outside OLMo. It is a
# replication, not a second copy of the study, and it comes with two hard caveats
# (see docs/anthropic_claims_scorecard.md, "Second-family replication"):
#
#   NO ANCHOR   Neuronpedia publishes no lens for any Mistral/Ministral, so
#               validate_fit cannot gate the base fit. Ministral lenses are
#               UNANCHORED; validation is (a) the pipeline proven on OLMo + (b)
#               the seed-split self-consistency check below.
#   NO CONTROL  base + instruct + reasoning only; no RL-Zero-by-domain arms, so
#               the capability confound is uncontrolled. Ministral answers "does
#               the shift reproduce", not "is it viewpoint vs capability".
#   MULTIMODAL  Mistral3ForConditionalGeneration wraps a vision encoder; the text
#               backbone is a submodule. _load_lm() below extracts it. UNTESTED
#               -- this is the one integration risk.
MINISTRAL_BASE = "mistralai/Ministral-3-8B-Base-2512"
MINISTRAL_ARMS = [
    MINISTRAL_BASE,
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-8B-Reasoning-2512",
]

N_PROMPTS = 616          # matches the published anchor lens
N_SHARDS = 8
LAYER_STEP = 3           # 11 of 32 layers; Anthropic's own figures subsample
MAX_SEQ_LEN = 128
# dim_batch is memory-only for a SINGLE fit but NOT throughput-neutral in practice:
# n_passes = ceil(d_model/dim_batch), and each pass carries fixed per-launch overhead.
# The first anchor attempt ran dim_batch=32 (128 backward passes/prompt) on an A100 and
# measured ~55 s/prompt; the H100+dim_batch=64 anchor that PASSED measured ~26 s/prompt --
# still ~2x Anthropic's published ~11.5 s/prompt (dim_batch=128 on a B200). The ladder now
# matches the published config exactly: dim_batch=128 on B200 (below), which fits
# comfortably (Anthropic ran it on a 178 GB B200 for this same model at all 31 layers; we
# use 11), restoring ~13 s/prompt and the ~27 GPU-hr budget. dim_batch is mathematically
# neutral for the RESULT, so the dim_batch=64 anchor validation carries over unchanged.
DIM_BATCH = 128
SKIP_FIRST = 16


def _slug(arm: str) -> str:
    return arm.split("/")[-1].lower()


def _load_lm(arm: str, torch):
    """Load an arm and wrap it as a jlens ``LensModel``, ready for jacobian_for_prompt.

    jacobian_for_prompt does NOT take a raw HF model -- it drives a LensModel
    (``.encode / .forward / .n_layers / .d_model / .layers``). ``jlens.from_hf`` builds
    that wrapper AND locates the text decoder itself: its layout registry already covers
    both the plain ``Olmo3ForCausalLM`` (``model``) and the multimodal
    ``Mistral3ForConditionalGeneration`` (``model.language_model`` / ``language_model``),
    so we do NOT hand-unwrap ``.language_model`` -- that gave jlens a bare nn.Module with
    none of the LensModel interface. We only choose the right AutoModel class for loading;
    from_hf does the rest and raises a clear error if a layout is genuinely unknown.

    Returns (lens_model, tokenizer).
    """
    import jlens
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(arm)
    cfg = AutoConfig.from_pretrained(arm)
    if getattr(cfg, "vision_config", None) is not None:
        # Multimodal wrapper: load the whole ForConditionalGeneration; from_hf finds the
        # text tower via its layout registry. UNTESTED path -- the anchor runs OLMo first.
        from transformers import AutoModelForImageTextToText

        full = AutoModelForImageTextToText.from_pretrained(
            arm, dtype=torch.bfloat16, device_map="cuda").eval()
    else:
        from transformers import AutoModelForCausalLM

        full = AutoModelForCausalLM.from_pretrained(
            arm, dtype=torch.bfloat16, device_map="cuda").eval()
    return jlens.from_hf(full, tok), tok


def _prompts(tok, n: int) -> list[str]:
    """Identical prompt set for every arm -- required for the comparison to mean
    anything. Deterministic given (n, max_seq_len) because the stream order is fixed."""
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




@app.function(image=image, gpu="B200", volumes={"/cache": cache, "/out": out},
              timeout=60 * 60, env=ENV)
def rope_emulation(n_prompts: int = 8) -> dict:
    """Fit a small-n Jacobian lens twice in THIS pipeline on the SAME prompts: once with the
    stock (per-layer-type) RoPE and once with YaRN forced onto the sliding-window layers,
    which is exactly what transformers 5.0-5.12 did (huggingface/transformers#46911).
    Reports the per-layer float64 cosine / relative error / identity distance between the
    two, and the final-hidden-state change on one prompt as a sanity check that the
    emulation actually altered the forward pass."""
    import json, pathlib, torch, jlens, transformers
    from jlens.fitting import jacobian_for_prompt
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(BASE)
    prompts = _prompts(tok, n_prompts)
    layers = [0, 9, 30]
    res = {"transformers": str(transformers.__version__), "torch": str(torch.__version__),
           "n_prompts": n_prompts, "layers": layers, "dim_batch": DIM_BATCH,
           "max_seq_len": MAX_SEQ_LEN, "skip_first": SKIP_FIRST, "conds": {}}
    J_by, hid = {}, {}
    for cond in ("fixed", "emulated_bug"):
        cfg = AutoConfig.from_pretrained(BASE)
        rp = getattr(cfg, "rope_parameters", None)
        before = json.loads(json.dumps(rp, default=str))
        if cond == "emulated_bug":
            cfg.rope_parameters["sliding_attention"] = dict(cfg.rope_parameters["full_attention"])
        m = AutoModelForCausalLM.from_pretrained(
            BASE, config=cfg, dtype=torch.bfloat16, device_map="cuda").eval()
        re_ = m.model.rotary_emb
        info = {"rope_parameters_before": before,
                "rope_parameters_used": json.loads(json.dumps(cfg.rope_parameters, default=str)),
                "rope_type": {k: str(v) for k, v in getattr(re_, "rope_type", {}).items()},
                "attention_scaling": {lt: float(getattr(re_, f"{lt}_attention_scaling"))
                                      for lt in getattr(re_, "layer_types", [])
                                      if hasattr(re_, f"{lt}_attention_scaling")}}
        try:
            info["inv_freq_full_eq_sliding"] = bool(torch.allclose(
                re_.full_attention_inv_freq, re_.sliding_attention_inv_freq))
        except Exception as e:  # noqa: BLE001
            info["inv_freq_full_eq_sliding"] = f"? {type(e).__name__}"
        ids = tok(prompts[0], return_tensors="pt", truncation=True,
                  max_length=MAX_SEQ_LEN).input_ids.to("cuda")
        with torch.no_grad():
            hid[cond] = m(ids, output_hidden_states=True).hidden_states[-1][0].float().cpu()
        lm = jlens.from_hf(m, tok)
        acc, per30 = None, []
        for p in prompts:
            J, _seq, _valid = jacobian_for_prompt(
                lm, p, layers, dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ_LEN,
                skip_first=SKIP_FIRST)
            if acc is None:
                acc = {l: torch.zeros(J[l].shape, dtype=torch.float64) for l in layers}
            for l in layers:
                acc[l] += J[l].double().cpu()
            I = torch.eye(J[30].shape[-1], dtype=torch.float64)
            per30.append(((J[30].double().cpu() - I).norm() / I.norm()).item())
        J_by[cond] = {l: acc[l] / len(prompts) for l in layers}
        I = torch.eye(J_by[cond][0].shape[-1], dtype=torch.float64)
        res["conds"][cond] = {
            "rotary": info, "per_prompt_id30": per30,
            "identity_distance": {str(l): ((J_by[cond][l] - I).norm() / I.norm()).item()
                                  for l in layers}}
        del m, lm
        torch.cuda.empty_cache()

    I = torch.eye(J_by["fixed"][0].shape[-1], dtype=torch.float64)
    per_layer = {}
    for l in layers:
        a, b = J_by["fixed"][l], J_by["emulated_bug"][l]
        per_layer[str(l)] = {
            "cos64": ((a * b).sum() / (a.norm() * b.norm())).item(),
            "rel64": ((a - b).norm() / a.norm()).item(),
            "cos_offidentity": (((a - I) * (b - I)).sum() / ((a - I).norm() * (b - I).norm())).item(),
        }
    res["per_layer"] = per_layer
    hf, hb = hid["fixed"], hid["emulated_bug"]
    res["final_hidden_rel_diff_prompt0"] = ((hf - hb).norm() / hf.norm()).item()
    res["final_hidden_cos_prompt0"] = ((hf * hb).sum() / (hf.norm() * hb.norm())).item()
    p = pathlib.Path("/out/olmo_ladder_tf513/rope_emulation.json")
    p.write_text(json.dumps(res, indent=1))
    out.commit()
    return res


@app.local_entrypoint()
def run_emulation(n_prompts: int = 8):
    import json
    r = rope_emulation.remote(n_prompts)
    print("ROPE-EMULATION", json.dumps(r, indent=1))
