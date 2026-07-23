# An introductory mech-interp curriculum, built from the OpenAI/Distill and Anthropic corpora

*How many of the published mechanistic-interpretability articles fit our framework —
open-weight reproduction + a control + a runnable technique — as the basis for
introductory material that leads a newcomer up to the recent (J-space, introspection,
societies-of-thought) papers.*

## Fit criteria

An article fits if it is (1) **reproducible on open weights** (no frontier access), (2) a
**concrete runnable technique** (feature, circuit, lens, probe, SAE, intervention), (3)
**able to carry a control/null** (the discipline the program teaches), and (4) a
**stepping-stone** toward the recent papers.

## The count

Of ~40 items across the two threads, **~21 fit**, in two tiers:

- **Tier 1 — foundational, directly open-reproducible (14).** Ideal introductory rungs; the
  technique runs today on an open model or a toy model.
- **Tier 2 — recent "destination" papers whose *method* is open (7).** The subject model is
  closed (Claude), but the technique reproduces on open weights, so each becomes a capstone
  "reproduce-it-with-a-control" lesson. Several we have already reproduced.

**Do not fit** (not lessons): position/agenda essays (*Interpretability Dreams*, *Reflections
on Qualitative Research*), tooling pages (Microscope, PySvelte, HeadVis, Garcon), the monthly
*Circuits Updates* grab-bags, and architecture-change notes (SoLU) — plus the six extra Distill
vision case studies (Curve Detectors, Early Vision, Equivariance, High-Low Frequency, Branch
Specialization, Weight Banding), which reproduce fine but are redundant once one worked example
is taught.

## The curriculum — eight rungs, each: concept → run it on an open model → run its control

| # | rung (concept) | source article(s) | open model to run on | the control it teaches |
|---|---|---|---|---|
| 1 | What a **feature / circuit** is | *Zoom In*; *Feature Visualization*; *Curve Circuits*; *Visualizing Weights* (Distill) | InceptionV1 (open vision) | randomize the net → the "feature" should vanish (Adebayo sanity check) |
| 2 | How a **transformer computes** (residual stream, QK/OV) | *A Mathematical Framework for Transformer Circuits* | GPT-2 small | QK/OV attribution vs a shuffled-head baseline |
| 3 | **Superposition** — why features are hard to read | *Toy Models of Superposition* (+ *Superposition, Memorization & Double Descent*) | toy model, on a laptop | recovered features vs a random-dictionary null |
| 4 | A concrete **LM circuit: induction heads** | *In-Context Learning and Induction Heads* | GPT-2 / Pythia | ablate the head → in-context accuracy drops (causal null) |
| 5 | **Extracting features: sparse autoencoders** | *Towards Monosemanticity* → *Scaling Monosemanticity* | released 1-layer model; **Gemma Scope** open SAEs | feature-steering effect vs a random-direction null |
| 6 | **Reading the stream into words: lenses** | J-space paper's **Jacobian lens** (+ logit/tuned lens lineage) | **OLMo-3 / Qwen** (our `jlens` / `jlens-lab`) | randomization control, distance null, logit-lens baseline — *already built* |
| 7 | **Following computation: attribution graphs** | *Circuit Tracing* + *On the Biology of an LLM* | Gemma-2-2B / Llama via open `circuit-tracer` | faithfulness check (cf. *Toy Model of Mechanistic Unfaithfulness*) |
| 8 | **The recent property claims (capstones)** | *Manifolds*; *Introspection*; *Societies of Thought*; *Global Workspace / J-space*; *Emotions*; metacognition | OLMo-3 / Ministral (our work) | each reproduced on open weights **with its null** — the Scorecard cells |

Optional bridge between rungs 1 and 5: *Multimodal Neurons in CLIP* — "features that mean
things," on open CLIP weights, motivating the leap from vision features to concept features in
language models.

## What we already have vs. what the on-ramp needs

- **Rungs 6 and 8 are largely written** — our lens, controls, randomization result, and the
  metacognition/introspection/SoT reproductions *are* those lessons, each with a control.
- **Rungs 1–5 and 7 are the on-ramp** — the foundational material (features, superposition,
  induction, SAEs, attribution) that makes the recent papers legible to a newcomer. All are
  open-reproducible today; several have canonical open targets (Toy Models code, Gemma Scope,
  `circuit-tracer`).

The curriculum is therefore not a from-scratch build: it is our existing capstone work (the
hard end) plus a well-defined, fully-open foundational on-ramp assembled from published,
reproducible articles. This is the concrete form of the proposal's tutorial deliverable — and
each rung's "run the control" step is the same discipline the Scorecard enforces, taught by
hand.
