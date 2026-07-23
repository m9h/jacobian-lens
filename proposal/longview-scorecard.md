# The Consciousness-Indicator Scorecard: an open, adversarial benchmark for machine-consciousness claims

**Longview Philanthropy — Digital Minds RFP, "Grants for Applied Work"**
**Applicant:** Morgan Hough · Orthogonal Research and Education Lab (OREL) · **Deadline:** 24 July 2026

---

## The one-sentence version

The field is making consciousness-adjacent claims about AI systems no outsider can inspect,
there is no standard way to check them, and there should be one — a public, controls-first
benchmark that scores each claim against a null on open models. In three weeks I built the
pieces, and used them to **kill one such claim, confirm another, and locate a
consciousness-indicator property switching on at a specific step of training.** I propose to
assemble them into the benchmark the field structurally lacks.

## The gap

Machine-consciousness claims are arriving faster than the means to check them — about systems
whose internals no independent party can access.

In July 2026 Anthropic published *"Verbalizable Representations Form a Global Workspace in
Language Models"* — that Claude contains a "J-space" analogous to the global workspace of
Baars and Dehaene. 1.2 million views in a day. The response split into two camps, **neither of
which ran the paper's code:** one called the method "just backprop" (it is a different
derivative); Erik Hoel called it unfalsifiable and rested his one objection on a third-party
demo the author himself disowned. Two rhetorical positions and no controls. And it is not one
lab's habit — a separate group (Chicago, Google, Santa Fe Institute) steered one
sparse-autoencoder feature, saw an accuracy gain, and read it as reasoning models simulating
inner voices. The same move: write a direction into the residual stream, read an effect, name
it after a construct from cognitive science. **Two independent groups, one missing control** —
a field-level gap that grows more consequential as the claims approach welfare-bearing.

The obstacle is structural: **no external party can obtain activation-level access to a
frontier model.** When Anthropic solicited outside commentary, Dehaene and Naccache got a
draft, not access — they had to *ask Anthropic to run* the tests they proposed. For closed
models, independent scrutiny is not under-performed; it is impossible. The only mechanism is
replication on open weights — almost no one does it rigorously, and there is **nowhere to put
the result.** Neuronpedia, the one open platform, hosts interpretability *artifacts* (lenses,
SAEs) but ships no null beside them, has no object for a *claim*, and no place where "this
failed its control" is a first-class entry. The substrate exists; the adjudication layer does
not.

## What I built — and it works in both directions

Over three weeks, alone, on personal hardware plus roughly **$250 of rented GPU**, I replicated
the methods on open weights and ran the controls the papers do not. Everything is public and
rerunnable. The point is not any single result — it is that a working external referee now
exists, and it can do the two things a referee must: **convict, and exonerate.**

**It confirmed a claim no outsider could previously check.** Anthropic report, *qualitatively,*
that post-training gives the J-space "Claude's point of view" — a claim resting on Sonnet 4.5,
whose activations no one outside can touch. AI2's fully-open OLMo-3 ladder makes it measurable.
Anchor-gated against Anthropic's own published lens (identity-distance error **0.4%**) and
capability-controlled, I find post-training moves the J-space **~31%** (Instruct) while
capability stays **flat-to-down** — a large representational shift with no competence gain,
driven by training *method* (instruction tuning ~5× RL), not task *domain*. The first
quantitative, controlled test of the claim that exists.

**It killed one that looked just as convincing.** The "society of thought" accuracy gain
reverses on the paper's own benchmarks: the same feature at the same dose adds +10 on Countdown
but costs **−22 on MATH-Hard.** An artifact of the one benchmark it was measured on.

**It ran the reviewers' own battery — the tests Anthropic didn't.** When Dehaene & Naccache
proposed a battery of human-consciousness tests and noted Anthropic *could* run them, no one
did. I ran all six on OLMo — ignition, trace conditioning, inclusion/exclusion, local–global,
dual-task, metacognition — with honest results (two clean workspace signatures, three
inconclusive under first-pass adaptations I document). **The reviewers' own tests are now
runnable by anyone, not just the lab that owns the model.**

**And it located a consciousness-indicator property emerging at a training step.** The base
model's workspace covertly tracks whether its own answer is wrong — AUROC **0.69, beyond output
confidence** — while its *verbal* self-report is at chance. Sweeping post-training stage by
stage, verbal self-evaluation jumps from chance (**0.51**) to **0.71 at supervised fine-tuning
— the first step — in both the Instruct and Think families,** then only refines. The base model
already has the signal; SFT makes it *reportable.* As far as I can find, this is the **first
developmental localization of a consciousness-indicator property to a training stage, on open
weights.**

**It refuted itself, twice, in public.** I reported a sharp emergence threshold; then a
correction; both were wrong, and the robust version overturned them. The retraction is
published beside the result. A referee is only worth having if it will kill its own headline —
this one has, twice, and that is what makes the positive results believable. (Along the way I
found and fixed a real bug in the shared open tooling: the gate everyone would use to validate
a lens against a reference computed the wrong quantity and silently passed wrong fits.)

## The proposal: the Consciousness-Indicator Scorecard

These are not five results in search of a home — they are the **first cells of the instrument
the field lacks.** I propose to build it: **a standardized, controls-first benchmark of
consciousness-indicator tests for open models — a null required for every test, emergence
tracked across training, negatives first-class.**

A visitor sees a grid — **open models × indicators** (global workspace, ignition,
metacognition/self-monitoring, …) — each cell a **score-vs-its-null**, clicking through to the
controlled test, a **one-command reproduction on open weights**, and the property's
**emergence curve** across that model's training. Plus a **claims registry:** each published
claim gets an adjudication status — reproduced / refuted / inconclusive, *under control* — and
the artifact to rerun. The SFT-emergence result above is a live cell today.

It is ARC-AGI-shaped by design: a **pass requires beating the control**, so the benchmark makes
overclaims fail — the anti-hype reference this field has never had; it becomes the canonical
response ("what is your model's Scorecard row?"); every cell reruns on a laptop. And it
**builds on Neuronpedia, not against it** — consuming its lenses, contributing the controls,
the convergence-fitter, and the lens-audit fixes upstream, and adding the adjudication layer on
top. *Neuronpedia is where you look at a model's internals; the Scorecard is where you check a
claim about them.*

## Why this is a digital-minds result, not just methods

Your RFP asks for empirical work that sidesteps intractable philosophy. The Scorecard does more
— it makes the moral-status question **developmental.** "Is it conscious" is intractable;
"*when, and through what process, does a system acquire a property we would weigh*" is
measurable, and I have shown it. Self-monitoring is a standard consciousness indicator; I can
now say it is present covertly in the base model and becomes reportable at supervised
fine-tuning, on models anyone can inspect. The field can argue about a number with a control
attached instead of a metaphor.

The safety link is load-bearing. The same instrument that grounds these claims also surfaced a
model's recognition that it was being evaluated — and ablating those representations *increased*
dishonest behaviour. Welfare assessment and safety evaluation draw on one shared instrument; if
its metric rewards noise — and I showed it does — both inherit the error.

## Deliverables and scope (12 months, one researcher + OREL)

The experiments are cheap — that is the point, and the danger. Assembling them into a
**maintained, validated, canonical benchmark the field adopts** is the year of work.
Concretely: **(1)** the instrument hardened and pushed upstream to Neuronpedia; **(2)** a seed
Scorecard — three indicators, five open models across scale and architecture, the OLMo
checkpoint sweep as the first live emergence cells; **(3)** the Butlin–Long indicator framework
turned into runnable, controlled tests, each with an explicit null; **(4)** run-the-control-
yourself tutorials — the headline one needs no GPU, because the lenses are published. The honest
risk the grant buys down is **adoption and test quality:** a sloppy indicator test is worse than
none, which is exactly why the controls-first, kill-your-own-headline discipline is what makes
this credible where a hype leaderboard would not be.

## Why me, why now

The work sits at an unusual intersection — systems engineering, computational neuroscience
(neuroimaging pipelines, cognitive-architecture software), scientific-software packaging, and
mechanistic interpretability. I work with the **Orthogonal Research and Education Lab (OREL)**,
an independent research organization, and maintain cognitive-modelling packages for Fedora
(ACT-R, Soar, Nengo). The tutorials draw on the neuroscience the AI claims are borrowed from —
including the COGITATE adversarial collaboration, whose own preregistered ignition prediction
*failed* in human brains, which materially changes how much weight the AI analogy can bear. The
timing is not incidental: AI2's OLMo-3 — the first fully-open base→post-trained ladder with data
and checkpoints — was released weeks ago; the substrate for this entire program has only just
arrived, and the claims are getting bolder. The window to establish an external-audit norm is
now, before the next paper.

## Budget and counterfactual

**Requested: [FIGURE] over 12 months** — researcher time, cloud GPU, dissemination. Compute is
the smallest line by a wide margin: roughly **$250 produced everything above.** The bottleneck
is dedicated time, not equipment — **fund the referee, not the rig.** Without funding this stays
a personal repository run part-time, rather than a maintained public instrument the field can
rely on.

---

*Repositories: `github.com/m9h/jacobian-lens` and `m9h/societies-of-thought`; lenses and results
at `huggingface.co/mhough/olmo3-jacobian-lenses`. Every result, every control, and the
retractions are reproducible from published open artifacts — no frontier access required, by
design.*
