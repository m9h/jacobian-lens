# The Consciousness-Indicator Scorecard

*The core deliverable: an ARC-AGI-style, controls-first, open-model benchmark for
machine-consciousness claims — the adjudication layer the field lacks, built on top of the
open interpretability artifacts that already exist.*

---

## What Neuronpedia doesn't have

Neuronpedia is the **artifact layer**: it hosts SAEs, the Jacobian lenses, circuit graphs —
and lets you *explore* them. What it has no notion of:

1. **Controls.** It hosts the lens; it doesn't ship the *null* beside it. There's nowhere
   that says "this readout beats a distance-only / randomization / logit-lens baseline by X."
   (We even found one of its hosted lenses is silently broken — there is no QA/adjudication
   layer.)
2. **Claims, not features.** It's feature-level ("here's SAE feature #30939"). It has no
   object for a *claim* — "this model has a global workspace," "post-training installs a
   viewpoint" — and no way to run a standardized, controlled test of one.
3. **Emergence.** It's per-model snapshots. Nothing tracks how a *property* appears across a
   training ladder or checkpoints, or compares it across families/scales.
4. **Negatives on equal footing.** It's a gallery of things that fire. "This claim *failed*
   its control" is not a first-class entry anywhere.

Neuronpedia is the substrate. **The missing thing is the adjudication layer that sits on top
of it.**

## The ARC-AGI-shaped product

ARC-AGI is a *fixed, versioned battery + a public leaderboard + reproducible eval*, designed
so that hype fails and only real capability scores. The analog here:

**A standardized, controlled benchmark of consciousness-indicator tests for open models —
with a null required for every test, emergence tracked across training, and negative results
first-class.**

What a visitor sees: a grid — **open models (rows) × indicators (columns)** (GWT bottleneck,
global broadcast, ignition/all-or-none, metacognition/self-monitoring, …), every cell a
**score-vs-its-null** (green/amber/red), each clicking through to (a) the exact controlled
test, (b) a one-command reproduction on open weights — *no frontier access* — and (c) the
**emergence curve** across that model's training. Plus a **claims registry**: when a paper
asserts an indicator, it gets an entry with an adjudication status — *reproduced under
control / refuted under control / inconclusive* — and the artifact to rerun.

Why it's genuinely ARC-AGI-shaped, not just a repo:

- **Adversarial by construction.** A "pass" *requires* beating the control; the design goal
  is to make overclaims fail. That's the anti-hype benchmark this field has never had.
- **Canonical response.** When a lab says "our model shows property X," the answer becomes
  "what's its row on the Scorecard?" — the way ARC-AGI is the answer to reasoning claims.
- **Reproducible + open.** Every cell reruns on a laptop from published lenses. That's the
  property that made "$250 produced all this" true, turned into a public good.
- **Versioned + living.** New indicators added as the field proposes them; new models as they
  ship; the checkpoint dimension makes it developmental.

## Why this is the deliverable

- The individual experiments are cheap — that's why they can't be the deliverable.
  **Assembling them into a standardized, validated, maintained, canonical benchmark that the
  field adopts is the year of work**, and it's durable in a way a script never is.
- It **builds on Neuronpedia rather than duplicating it** — consumes its hosted lenses,
  contributes the controls + fitter + fixes upstream, and adds the adjudication layer above.
  Complementary, not competitive. The clean line: *"Neuronpedia is where you look at a model's
  internals; the Scorecard is where you check a claim about them."*
- It's the concrete artifact that makes "standing external referee" and "open emergence
  science" the *same object*: the referee is the benchmark; the science is what populating it
  produces.

## The honest risks

A benchmark lives or dies on **test quality and adoption**. A sloppy indicator test is worse
than none — which is exactly why the *controls-first, negatives-first, kill-your-own-headline*
discipline already demonstrated is the thing that makes this version credible where a hype
leaderboard wouldn't be. Adoption is uncertain and is the real risk the grant is buying down.

## Scoping (12 months, one researcher + OREL)

If the full benchmark is too much for the grant period, scope it to a **seed** and pitch the
benchmark as the trajectory:

- **three indicators** (e.g. global workspace / bottleneck, ignition, metacognition),
- **five open models** across scale and architecture,
- **the emergence dimension on OLMo-3** (the checkpoint sweep as the first live, developmental
  cells).

In this framing the proposal's spine becomes "build and populate the Scorecard," the
checkpoint-sweep result becomes the first live cell rather than a standalone finding, and the
title/pitch are built around the Scorecard as the durable, canonical product.
