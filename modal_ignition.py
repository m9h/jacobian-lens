"""Ignition test — the all-or-none threshold entry Dehaene & Naccache say the paper
did NOT establish, run on fully open OLMo-3.

WHAT DEHAENE ASKED FOR (commentary on Gurnee & Lindsey, June 2026, verbatim)
---------------------------------------------------------------------------
"Ignition remains to be fully demonstrated. The J-space is shown to be limited in
capacity, but the paper does not establish the nonlinear, competitive, all-or-none entry
into the workspace which ... is a reliable signature of conscious access. ... The decisive
experiment is feasible: present a stimulus at graded strengths ... and ask whether J-space
representations switch on with a threshold-like nonlinearity, while earlier, non-J-space
layers rise monotonically with input strength. Better still, present stimuli exactly at
threshold and look for a bifurcation across runs."

Nobody has run it: Claude's activations are closed, and Anthropic did not. OLMo-3 makes it
runnable in the open. This is a pure READOUT through the published olmo-3-1025-7b Jacobian
lens (31 layers) -- no fitting, no training.

DESIGN (text analog of "graded stimulus strength")
---------------------------------------------------
For each of several concepts, a neutral base sentence is followed by 0..K implying clues
(France: croissants, Louvre, Eiffel Tower, ...). Evidence strength = number of clues k.
The concept is NEVER named in the prompt; we ask whether it is present in the workspace
covertly. At the final token we read the lens logit for the concept's tokens at EVERY
layer -- the J-space activation of the concept as a function of evidence.

GNW prediction (ignition):
  * early / pre-workspace layers: concept activation rises ~monotonically with k
  * workspace band (mid layers): activation switches on with a THRESHOLD nonlinearity
    (near-flat, then a sudden jump, then saturated) -- all-or-none.
So the per-layer "nonlinearity index" should be LOW early and PEAK in the workspace band.

Bifurcation (the "at threshold, across runs" version): at each concept's threshold k*, we
generate many variants (shuffled clue order + neutral distractors) and record the spread of
workspace activation -- GNW predicts a bimodal (ignited / not) distribution near threshold.

CONTROLS baked in: a matched OFF-concept that no clue implies (must stay flat), and the
plain logit lens (use_jacobian=False) as a non-workspace baseline.
"""

from __future__ import annotations

import modal

app = modal.App("olmo-ignition-jlens")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.6", "transformers>=5.5,<6",
        "accelerate", "safetensors", "numpy", "huggingface_hub",
    )
    .run_commands(
        "pip install git+https://github.com/m9h/jacobian-lens.git",
        "pip install git+https://github.com/m9h/jlens-lab.git",
    )
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)   # OLMo already cached
out = modal.Volume.from_name("jlens-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"}

BASE = "allenai/Olmo-3-1025-7B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt"

# Each item: a concept implied by a graded pile of clues but NEVER named, plus a matched
# OFF concept that the same clues do NOT imply (the flat control).
ITEMS = [
    {"concept": "France", "off": "Brazil",
     "base": "The travelers filled a notebook with memories of their vacation.",
     "clues": [" In the morning they ate warm croissants.",
               " They spent an afternoon inside the Louvre.",
               " They climbed to the top of the Eiffel Tower.",
               " They took an evening walk along the Seine.",
               " They bought fresh baguettes from a corner bakery.",
               " They strolled the length of the Champs-Élysées."]},
    {"concept": "Japan", "off": "Egypt",
     "base": "The travelers filled a notebook with memories of their vacation.",
     "clues": [" They slept on tatami mats in a small inn.",
               " They ate sushi at a counter near the market.",
               " They rode the bullet train between cities.",
               " They visited a quiet Shinto shrine.",
               " They soaked in a hot spring called an onsen.",
               " They watched the cherry blossoms fall."]},
    {"concept": "chess", "off": "tennis",
     "base": "The two friends sat down at the table in the club.",
     "clues": [" One of them opened by advancing a pawn.",
               " The other developed a knight toward the center.",
               " They castled their kings to safety.",
               " A bishop pinned the opposing knight.",
               " She announced check with her queen.",
               " He resigned rather than face checkmate."]},
    {"concept": "hospital", "off": "library",
     "base": "The building was busy from the early hours of the morning.",
     "clues": [" Nurses moved quickly between the rooms.",
               " A surgeon scrubbed in before the operation.",
               " Monitors beeped beside each bed.",
               " An ambulance pulled up to the entrance.",
               " A patient was wheeled toward the ward.",
               " The doctor reviewed the latest x-rays."]},
    {"concept": "winter", "off": "summer",
     "base": "She looked out of the window before leaving the house.",
     "clues": [" Snow had piled against the fence overnight.",
               " Her breath fogged in the freezing air.",
               " She pulled on a heavy wool scarf.",
               " Icicles hung from the gutter above.",
               " The pond had frozen solid enough to skate.",
               " She shoveled the driveway before dawn."]},
]


def _first_ids(tok, words):
    """First-subword ids for a concept's surface variants (space/case)."""
    ids = set()
    for w in words:
        t = tok(w, add_special_tokens=False)["input_ids"]
        if t:
            ids.add(t[0])
    return sorted(ids)


def _variants(concept):
    c = concept
    return [f" {c}", c, f" {c.lower()}", c.lower(), f" {c.capitalize()}"]


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=40 * 60, env=ENV, retries=1)
def ignition() -> dict:
    import json, pathlib, torch, jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import JacobianLens

    tok = AutoTokenizer.from_pretrained(BASE)
    hf = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                              device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    layers = lens.source_layers                      # 0..30

    def concept_activation(prompt, ids, use_jac):
        """Max lens logit over the concept's first-subword ids, per layer, at last token."""
        ll, _, _ = lens.apply(model, prompt, layers=layers, positions=[-1],
                              max_seq_len=256, use_jacobian=use_jac)
        idx = torch.tensor(ids)
        return {l: ll[l][0][idx].max().item() for l in layers}

    # ---- graded-evidence sweep ----
    per_item = []
    for it in ITEMS:
        on_ids = _first_ids(tok, _variants(it["concept"]))
        off_ids = _first_ids(tok, _variants(it["off"]))
        curves_on, curves_off, curves_logit = [], [], []
        for k in range(len(it["clues"]) + 1):
            prompt = it["base"] + "".join(it["clues"][:k])
            curves_on.append(concept_activation(prompt, on_ids, True))
            curves_off.append(concept_activation(prompt, off_ids, True))
            curves_logit.append(concept_activation(prompt, on_ids, False))  # logit-lens baseline
        per_item.append({"concept": it["concept"], "off": it["off"],
                         "on": curves_on, "off_curve": curves_off, "logit": curves_logit})

    # ---- bifurcation at threshold ----
    # threshold k* = first k where the workspace-band on-activation exceeds the midpoint.
    ws_band = [l for l in layers if 0.45 * layers[-1] <= l <= 0.75 * layers[-1]]
    def ws_mean(curve): return sum(curve[l] for l in ws_band) / len(ws_band)

    import random
    bif = []
    for it in ITEMS:
        on_ids = _first_ids(tok, _variants(it["concept"]))
        full = [ws_mean(concept_activation(it["base"] + "".join(it["clues"][:k]), on_ids, True))
                for k in range(len(it["clues"]) + 1)]
        lo, hi = min(full), max(full)
        kstar = next((k for k in range(len(full)) if full[k] >= lo + 0.5 * (hi - lo)), len(full) // 2)
        # 24 variants at k*: shuffle clue order (rng seeded by index for resumability)
        clues = it["clues"][:max(kstar, 1)]
        vals = []
        rng = random.Random(len(it["concept"]))
        for _ in range(24):
            order = clues[:]
            rng.shuffle(order)
            vals.append(ws_mean(concept_activation(it["base"] + "".join(order), on_ids, True)))
        bif.append({"concept": it["concept"], "kstar": kstar,
                    "full_ws_curve": full, "variant_ws": vals})

    res = {"layers": layers, "n_layers_model": model.n_layers,
           "workspace_band": ws_band, "items": per_item, "bifurcation": bif}
    d = pathlib.Path("/out/ignition"); d.mkdir(parents=True, exist_ok=True)
    (d / "ignition_raw.json").write_text(json.dumps(res))
    out.commit()
    return {"n_items": len(per_item), "layers": len(layers),
            "workspace_band": [ws_band[0], ws_band[-1]], "saved": "/out/ignition/ignition_raw.json"}


@app.local_entrypoint()
def run():
    print("  Ignition test (Dehaene's 'decisive experiment') on OLMo-3, published lens...")
    print("  ", ignition.remote())
    print("  raw curves saved to the jlens-out volume: ignition/ignition_raw.json")
    print("  pull with: modal volume get jlens-out ignition/ignition_raw.json <local>")
