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
        "torch>=2.6",
        # OLMo-3 (Olmo3ForCausalLM, YaRN RoPE) and Ministral-3
        # (Mistral3ForConditionalGeneration) only exist in transformers 5.x -- 4.57
        # predates the OLMo-3 release. This app is pure torch (no Flax), so the old
        # "<5 keeps the Flax classes" rationale does not apply. Verified locally on
        # transformers 5.9.0 / torch 2.12: both classes import. The anchor gate is what
        # confirms the weights (and YaRN RoPE) actually load correctly under this pin.
        "transformers>=5.5,<6",
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
# measured ~55 s/prompt -- ~4x Anthropic's published ~11.5 s/prompt (dim_batch=128 on a
# B200), which would blow both the 60-min shard timeout and the ~27 GPU-hr budget. 64
# halves the passes and fits an 80 GB card comfortably for an 11-layer 7B fit; paired
# with H100 (below) this restores published-class throughput. (128 fits memory too but
# has OOM'd before on larger configs, so 64 is the safe step.)
DIM_BATCH = 64
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


# =====================================================================================
# CAPABILITY PROBE -- run BEFORE the ladder (agreed 2026-07-19; the user's chosen flow)
# =====================================================================================
# The ladder's whole logic is that geometry moving across the RL-Zero DOMAIN arms
# (Math/Code/IF/General/Mix, matched method) is viewpoint, not capability -- because
# capability is "held roughly constant". That premise is an assumption until measured.
# This probe measures capability per arm with FORWARD PASSES ONLY (no Jacobian), so if
# capability turns out to be collinear with the domain axis we learn it for ~one cheap
# pass instead of after ~27 GPU-hours of fitting.
#
# Two axes, both generation-free:
#   ppl  -- mean token NLL on the SAME neutral wikitext prompts the lens is fit on.
#           A domain-neutral scalar. (Instruction/Think arms shift the text distribution,
#           so read ppl RELATIVELY, not as a clean capability number -- hence also:)
#   mmlu -- length-normalised answer-logprob accuracy on a domain-balanced MMLU battery,
#           bucketed into math / code / general. The KEY output is the arm x domain
#           matrix: if each RL-Zero-{Math,Code,General} arm is best at ITS OWN domain
#           (a dominant diagonal), capability is domain-shaped and the RL-Zero control
#           is confounded -- exactly the finding worth having before the ladder.
MMLU_DOMAINS = {
    "math": ["abstract_algebra", "college_mathematics",
             "high_school_mathematics", "elementary_mathematics"],
    "code": ["college_computer_science", "high_school_computer_science",
             "machine_learning", "computer_security"],
    "general": ["miscellaneous", "philosophy", "world_religions", "prehistory"],
}
N_PER_SUBJECT = 25          # 25 x 4 subjects x 3 domains = 300 MC items, generation-free
N_PPL_PROMPTS = 50


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=40 * 60, env=ENV, retries=1)
def capability(arm: str) -> dict:
    """Forward-pass capability for one arm: neutral perplexity + MMLU-by-domain accuracy.

    No Jacobian, no LensModel -- a plain HF model. MC items are scored by the
    length-normalised log-probability of each choice's text (the standard base-model
    protocol; letter-scoring is unreliable for base checkpoints).
    """
    import json, math, pathlib, torch
    from transformers import AutoConfig, AutoTokenizer
    from datasets import load_dataset

    # Degrade gracefully: a single unloadable arm (e.g. a non-standard model_type) must
    # not crash the whole .map -- return an error marker so the others still report.
    try:
        tok = AutoTokenizer.from_pretrained(arm)
        cfg = AutoConfig.from_pretrained(arm)
        if getattr(cfg, "vision_config", None) is not None:
            from transformers import AutoModelForImageTextToText as _Cls
        else:
            from transformers import AutoModelForCausalLM as _Cls
        model = _Cls.from_pretrained(arm, dtype=torch.bfloat16, device_map="cuda").eval()
    except Exception as e:
        return {"arm": arm, "error": f"{type(e).__name__}: {str(e)[:120]}"}

    # --- neutral perplexity (same wikitext the lens is fit on) ---
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for p in _prompts(tok, N_PPL_PROMPTS):
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=MAX_SEQ_LEN).input_ids.to("cuda")
            if ids.shape[1] < 2:
                continue
            nll = model(ids, labels=ids).loss.item()
            tot_nll += nll * (ids.shape[1] - 1)
            tot_tok += ids.shape[1] - 1
    ppl = math.exp(tot_nll / tot_tok) if tot_tok else float("nan")

    # --- MMLU by domain, length-normalised answer logprob ---
    def score(ctx_ids, choice_text):
        cids = tok(choice_text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to("cuda")
        full = torch.cat([ctx_ids, cids], dim=1)
        with torch.no_grad():
            logits = model(full).logits[0]
        lp = torch.log_softmax(logits[ctx_ids.shape[1] - 1: full.shape[1] - 1], dim=-1)
        return lp.gather(1, cids[0, :, None]).mean().item()   # mean = length-normalised

    dom_acc, dom_n = {}, {}
    for dom, subjects in MMLU_DOMAINS.items():
        correct = total = 0
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split=f"test[:{N_PER_SUBJECT}]")
            except Exception:
                continue
            for q in ds:
                ctx = tok(f"{q['question'].strip()}\nAnswer:",
                          return_tensors="pt").input_ids.to("cuda")
                scores = [score(ctx, " " + c) for c in q["choices"]]
                correct += int(scores.index(max(scores)) == int(q["answer"]))
                total += 1
        dom_acc[dom] = (correct / total) if total else None
        dom_n[dom] = total

    accs = [a for a in dom_acc.values() if a is not None]
    res = {"arm": arm, "ppl": ppl,
           "mmlu_overall": (sum(accs) / len(accs)) if accs else None,
           "mmlu_by_domain": dom_acc, "n_by_domain": dom_n}
    d = pathlib.Path(f"/out/olmo_ladder/{_slug(arm)}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "capability.json").write_text(json.dumps(res, indent=2))
    out.commit()
    return res


@app.local_entrypoint()
def capability_all():
    """Measure capability across all twelve OLMo arms, then flag the confound directly.

    The decision this informs: if capability is collinear with the domain axis (each
    RL-Zero domain arm best at its own domain, or capability monotone along the method
    ladder), the ladder geometry cannot be read as viewpoint-vs-capability and the design
    must change BEFORE ~27 GPU-hours are spent.
    """
    allrows = [r for r in capability.map(ARMS) if r]
    failed = [r for r in allrows if r.get("error")]
    rows = [r for r in allrows if not r.get("error")]
    rows.sort(key=lambda r: r["arm"])
    for r in failed:
        print(f"  SKIPPED {_slug(r['arm']):38s} {r['error']}")
    print(f"\n  {'arm':38s} {'ppl':>7s} {'mmlu':>6s}  {'math':>6s} {'code':>6s} {'genl':>6s}")
    for r in rows:
        d = r["mmlu_by_domain"]
        def f(x): return f"{x:.3f}" if isinstance(x, float) else "  -  "
        print(f"  {_slug(r['arm']):38s} {r['ppl']:7.2f} {f(r['mmlu_overall']):>6s}  "
              f"{f(d['math']):>6s} {f(d['code']):>6s} {f(d['general']):>6s}")

    # Confound check 1: does each RL-Zero domain arm top its OWN domain column?
    rl = {r["arm"].split("RL-Zero-")[-1]: r for r in rows if "RL-Zero-" in r["arm"]}
    print("\n  RL-Zero domain-diagonal check (is each arm best at its own domain?):")
    for dom, arm_key in (("math", "Math"), ("code", "Code"), ("general", "General")):
        if arm_key not in rl:
            continue
        col = {k: v["mmlu_by_domain"].get(dom) for k, v in rl.items()
               if v["mmlu_by_domain"].get(dom) is not None}
        if not col:
            continue
        best = max(col, key=col.get)      # best is the arm KEY, not a value
        own = col.get(arm_key)
        flag = "DIAGONAL (own-domain best -> capability is domain-shaped)" if best == arm_key \
            else f"off-diagonal (best={best}={col[best]:.3f})"
        print(f"    {dom:8s}: RL-Zero-{arm_key} = {own:.3f}, argmax = {col[best]:.3f}  {flag}")

    # Confound check 2: capability spread along the method ladder.
    order = ["7B", "Instruct-SFT", "Instruct-DPO", "Instruct",
             "Think-SFT", "Think-DPO", "Think"]
    print("\n  Method-ladder capability (mmlu_overall along base -> post-trained):")
    for name in order:
        m = next((r for r in rows if r["arm"].endswith(name)
                  or r["arm"].endswith("Olmo-3-1025-7B") and name == "7B"), None)
        if m and m["mmlu_overall"] is not None:
            print(f"    {name:16s} {m['mmlu_overall']:.3f}")
    print("\n  Read: a dominant diagonal or a steep monotone ladder = capability is "
          "collinear with the axis; the ladder needs capability AS A COVARIATE, not "
          "as an assumed-away constant.")


@app.function(image=image, gpu="H100", volumes={"/cache": cache, "/out": out},
              timeout=90 * 60, env=ENV, retries=1)
def fit_shard(arm: str, shard: int) -> dict:
    """Accumulate a PARTIAL Jacobian sum over this shard's prompts.

    Returns metadata only; the partial sum itself goes to the volume because
    11 layers x 4096^2 float32 is ~740MB and does not belong in a return value.
    """
    import pathlib, torch, jlens
    from jlens.fitting import jacobian_for_prompt

    model, tok = _load_lm(arm, torch)

    n_layers = model.n_layers          # LensModel exposes n_layers; it has no .config
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
def combine(arm: str, half: int | None = None) -> dict:
    """sum(partials) / sum(counts) -- exact, because the fit is a plain mean.

    half=None (default): combine ALL shards into lens.pt -- the OLMo path.
    half in {0,1}: combine only that half's shards into lens_half{h}.pt, for the
    Ministral seed-split self-consistency check. In the half case shards are NOT
    deleted, since the other half is combined in a separate call.
    """
    import pathlib, torch, jlens

    d = pathlib.Path(f"/out/olmo_ladder/{_slug(arm)}")
    if half is None:
        shards = sorted(d.glob("shard_*.pt"))
        expected, out_name, delete = N_SHARDS, "lens.pt", True
    else:
        want = set(range(half, N_SHARDS, 2))
        shards = sorted(s for s in d.glob("shard_*.pt")
                        if int(s.stem.split("_")[1]) in want)
        expected, out_name, delete = len(want), f"lens_half{half}.pt", False
    if len(shards) != expected:
        raise RuntimeError(f"{arm}: expected {expected} shards, found {len(shards)}. "
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
    lens.save(str(d / out_name))
    if delete:
        for s in shards:      # partials are transient and large
            s.unlink()
    out.commit()
    return {"arm": arm, "n_prompts": n_total, "layers": len(layers), "half": half}


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


@app.function(image=image, volumes={"/out": out}, timeout=30 * 60, env=ENV)
def seed_split_consistency(arm: str) -> dict:
    """The anchor SUBSTITUTE for a family with no published lens.

    Ministral has no reference lens, so we cannot check the fit against ground
    truth. What we CAN check is that the fit is stable: two lenses built from
    disjoint halves of the prompt set must agree. A high self-similarity does not
    prove correctness (a consistently-wrong fit would also pass), which is why
    Ministral rides alongside OLMo rather than standing alone -- OLMo proves the
    pipeline against a real anchor, this proves Ministral's fit is not noise.
    """
    import pathlib, jlens, torch
    import torch.nn.functional as F

    d = pathlib.Path(f"/out/olmo_ladder/{_slug(arm)}")
    a = jlens.JacobianLens.load(str(d / "lens_half0.pt"))
    b = jlens.JacobianLens.load(str(d / "lens_half1.pt"))
    per_layer = {}
    for l in a.jacobians:
        va, vb = a.jacobians[l].flatten().float(), b.jacobians[l].flatten().float()
        per_layer[l] = float(F.cosine_similarity(va, vb, dim=0))
    mean_cos = sum(per_layer.values()) / len(per_layer)
    return {"arm": arm, "mean_self_cosine": mean_cos,
            "pass": mean_cos >= 0.95, "per_layer": per_layer}


@app.local_entrypoint()
def ministral():
    """Second-family replication. UNANCHORED -- run OLMo `anchor` first so the
    pipeline is proven against a real reference before trusting this.

    Each arm is fit twice, on disjoint prompt halves, and the two lenses must
    agree (seed_split_consistency) -- the substitute for the missing anchor.
    """
    print("  Ministral-3 second-family replication (UNANCHORED -- see scorecard)")
    for arm in MINISTRAL_ARMS:
        # fit two half-lenses per arm using the even/odd shards as the split
        for half, shard_set in ((0, range(0, N_SHARDS, 2)), (1, range(1, N_SHARDS, 2))):
            list(fit_shard.starmap([(arm, s) for s in shard_set]))
            combine.remote(arm, half=half)
        check = seed_split_consistency.remote(arm)
        flag = "OK" if check["pass"] else "INCONSISTENT -- fit is unstable, do not trust"
        print(f"  {arm}: self-cosine {check['mean_self_cosine']:.3f}  [{flag}]")
