"""Fit Jacobian lenses across the whole OLMo-3 post-training ladder, on Modal.

WHAT THIS TESTS
---------------
Anthropic claim 6 (see societies-of-thought/docs/anthropic_claims_scorecard.md):
post-training shaped the J-space to reflect a point of view rather than pure
next-token prediction. Untestable from outside a frontier lab -- you need base
AND post-trained checkpoints of one model. AI2 published twelve:

    Olmo-3-1025-7B                      <- published lens exists; our ANCHOR
      Instruct-SFT -> Instruct-DPO -> Instruct
      Think-SFT    -> Think-DPO    -> Think
      RL-Zero-{Math, Code, IF, General, Mix}

THE CONFOUND, AND WHY BOTH HALVES ARE NEEDED
--------------------------------------------
Every rung of the ladder changes CAPABILITY as well as training objective, so
"post-training shaped the viewpoint" and "the model got better" are not separable
from the ladder alone. The RL-Zero arms are the control: one base, one RLVR
method, five domains, capability roughly held. Geometry that moves across DOMAIN
at matched method cannot be explained as "more capable".

Agreed with the GWT agent 2026-07-19, after each of us found this confound in the
other's design and missed it in our own.

WHY FIXED PROMPT COUNT, NOT FIT-TO-CONVERGENCE
----------------------------------------------
A convergence check is inherently sequential -- it stops when mean relative
change stays under threshold for N consecutive prompts -- so it cannot be
sharded. Worse, arms converging at different counts are not strictly comparable.
So: 616 prompts for every arm, identical prompts, and convergence inherited from
the anchor via validate_fit rather than re-derived twelve times.

WHY SHARDING IS EXACT
---------------------
The Jacobian accumulates as a plain sum over prompts divided by the count
(modal_app.py: `jac[l] += J[l]; n += 1` ... `jac[l] / n`). So partial sums from
K workers combine as sum(partials) / sum(counts) with no approximation --
bit-identical modulo float associativity. This is what makes 30+ GPU-hours
collapse to ~20 minutes of wall clock.

It also removes the failure that already cost a run here: modal_app.py notes a
full fit "blew Modal's 6h function timeout at 650 prompts -- with NOTHING saved".
At 616/8 = 77 prompts per shard the timeout stops being reachable.

ORDER OF OPERATIONS -- the anchor gate is not optional
-------------------------------------------------------
    1. fit the BASE arm only
    2. validate against the published olmo-3-1025-7b lens (mean cosine >= 0.95)
    3. only if that passes, fan out the other eleven

Rationale: this project has already found a silent wrong-RoPE bug in the HF Flax
path that would corrupt a Jacobian fit invisibly -- J stays dense, the lens stays
readable, nothing downstream flags it. OLMo-3 is not affected by THAT bug (it is
Olmo3ForCausalLM with YaRN RoPE, not Llama with llama3), but the class of failure
is live and YaRN carries an attention_factor term that an incomplete
implementation would silently drop. Reproducing a published number first is the
only cheap defence, and skipping it has already cost this project ~10 GPU-hours
once.

USAGE
-----
    modal run modal_olmo_ladder.py::anchor          # step 1+2, ~3 min
    modal run modal_olmo_ladder.py::ladder          # step 3, ~20 min, 88 containers

Costs are dominated by GPU-seconds, which sharding does not change (~27 GPU-hours
at layer_step=3). Sharding buys wall clock only.
"""

from __future__ import annotations

import modal

APP = "olmo-ladder-jlens"
app = modal.App(APP)

# OLMo-3 needs no mamba-ssm (that was Nemotron-H) and no arch pin, so this image is
# much lighter than modal_app.py's.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.57,<5",   # 5.x removed the Flax classes and moved rope config
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
    "allenai/Olmo-3-7B-RL-Zero-Mix",
]

N_PROMPTS = 616          # matches the published anchor lens
N_SHARDS = 8
LAYER_STEP = 3           # 11 of 32 layers; Anthropic's own figures subsample
MAX_SEQ_LEN = 128
DIM_BATCH = 32
SKIP_FIRST = 16


def _slug(arm: str) -> str:
    return arm.split("/")[-1].lower()


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


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=60 * 60, env=ENV, retries=1)
def fit_shard(arm: str, shard: int) -> dict:
    """Accumulate a PARTIAL Jacobian sum over this shard's prompts.

    Returns metadata only; the partial sum itself goes to the volume because
    11 layers x 4096^2 float32 is ~740MB and does not belong in a return value.
    """
    import pathlib, torch, jlens
    from jlens.fitting import jacobian_for_prompt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(arm)
    model = AutoModelForCausalLM.from_pretrained(
        arm, dtype=torch.bfloat16, device_map="cuda").eval()

    n_layers = model.config.num_hidden_layers
    # EXCLUDE the target (final) layer: a Jacobian from the target to itself is
    # undefined and jacobian_for_prompt raises on every prompt.
    layers = list(range(0, n_layers - 1, LAYER_STEP))

    mine = _prompts(tok, N_PROMPTS)[shard::N_SHARDS]
    jac, n, skipped = None, 0, 0

    for i, p in enumerate(mine):
        try:
            J, _seq, _valid = jacobian_for_prompt(
                model, p, layers, dim_batch=DIM_BATCH,
                max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST)
        except Exception as e:
            # Fail fast on the FIRST prompt. A bare `except: continue` here once hid a
            # source_layers bug behind 600 useless iterations in this codebase.
            if i == 0:
                raise RuntimeError(
                    f"jacobian_for_prompt failed on the FIRST prompt of "
                    f"{arm} shard {shard}: {type(e).__name__}: {e}") from e
            skipped += 1
            continue
        if jac is None:
            jac = {l: torch.zeros_like(J[l]) for l in layers}
        for l in layers:
            jac[l] += J[l]
        n += 1

    if n == 0:
        raise RuntimeError(f"{arm} shard {shard}: no prompt produced a Jacobian")

    d = pathlib.Path(f"/out/olmo_ladder/{_slug(arm)}")
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"jac": {l: v.cpu() for l, v in jac.items()}, "n": n, "layers": layers},
               d / f"shard_{shard}.pt")
    out.commit()
    return {"arm": arm, "shard": shard, "n": n, "skipped": skipped, "layers": len(layers)}


@app.function(image=image, volumes={"/out": out}, timeout=30 * 60, env=ENV)
def combine(arm: str) -> dict:
    """sum(partials) / sum(counts) -- exact, because the fit is a plain mean."""
    import pathlib, torch, jlens

    d = pathlib.Path(f"/out/olmo_ladder/{_slug(arm)}")
    shards = sorted(d.glob("shard_*.pt"))
    if len(shards) != N_SHARDS:
        raise RuntimeError(f"{arm}: expected {N_SHARDS} shards, found {len(shards)}. "
                           "Refusing to build a lens from a partial fan-out.")

    total, n_total, layers = None, 0, None
    for s in shards:
        st = torch.load(s, map_location="cpu")
        layers = st["layers"]
        if total is None:
            total = {l: torch.zeros_like(v) for l, v in st["jac"].items()}
        for l in layers:
            total[l] += st["jac"][l]
        n_total += st["n"]

    lens = jlens.JacobianLens(
        jacobians={l: total[l] / n_total for l in layers},
        n_prompts=n_total, d_model=next(iter(total.values())).shape[-1])
    lens.save(str(d / "lens.pt"))
    for s in shards:      # partials are transient and large
        s.unlink()
    out.commit()
    return {"arm": arm, "n_prompts": n_total, "layers": len(layers)}


@app.function(image=image, volumes={"/out": out}, timeout=30 * 60, env=ENV)
def validate_anchor() -> dict:
    """Gate: does our BASE fit match the published olmo-3-1025-7b lens?

    Nothing else runs until this passes. See module docstring for why.
    """
    import jlens
    from jlens_lab import artifacts

    lens = jlens.JacobianLens.load(f"/out/olmo_ladder/{_slug(BASE)}/lens.pt")
    gate = artifacts.validate_fit(lens, ANCHOR_ID)
    return dict(gate)


@app.local_entrypoint()
def anchor():
    """Step 1+2: fit the base arm and validate it against the published lens."""
    res = list(fit_shard.starmap([(BASE, s) for s in range(N_SHARDS)]))
    print(f"  shards done: {sum(r['n'] for r in res)} prompts, "
          f"{sum(r['skipped'] for r in res)} skipped")
    print(" ", combine.remote(BASE))
    gate = validate_anchor.remote()
    print("  anchor gate:", gate)
    if not gate.get("pass"):
        raise SystemExit(
            "ANCHOR GATE FAILED -- our fit does not match the published lens. "
            "Do NOT fit the other eleven arms; the discrepancy is upstream of "
            "everything the ladder would conclude."
        )
    print("  anchor OK -- safe to run `modal run modal_olmo_ladder.py::ladder`")


@app.local_entrypoint()
def ladder():
    """Step 3: the other eleven arms, all shards in parallel."""
    rest = [a for a in ARMS if a != BASE]
    jobs = [(a, s) for a in rest for s in range(N_SHARDS)]
    print(f"  launching {len(jobs)} containers ({len(rest)} arms x {N_SHARDS} shards)")
    res = list(fit_shard.starmap(jobs))
    bad = [r for r in res if r["skipped"] > 0]
    if bad:
        print(f"  NOTE: {len(bad)} shards skipped prompts -- "
              f"arms will differ in n_prompts: {[(r['arm'], r['skipped']) for r in bad]}")
    for r in combine.map(rest):
        print("  ", r)
