# An Open Adjudication Layer for Interpretability Claims

**Foresight Institute — AI for Science & Safety Nodes** · focus area: **AI for Science &
Epistemics** · Morgan Hough, Orthogonal Research and Education Lab (OREL) · San Francisco hub,
**weekly in person** · **DRAFT — deadline not yet announced**

---

## The problem

Interpretability claims about AI systems are published faster than anyone can check them, on
systems no outside party can open. When Anthropic reported in 2026 that its model contains a
subspace analogous to a global workspace, its own invited commentators — Dehaene and Naccache,
whose theory the claim borrows — could not test it. They proposed six experiments and noted
Anthropic could run them. None were run, because the model is closed.

The response divided into two rhetorical positions and produced no controls. This is not one
laboratory's practice: a second group reported that steering a single sparse-autoencoder feature
raised reasoning accuracy and read it as interacting internal voices. Same procedure both times
— write a direction into the residual stream, observe an effect, name it after a construct from
cognitive science — and both times the confirming control is absent.

**There is no venue that records whether a claim survived a null.** The substrate exists;
the evaluation layer does not.

## What has been built, on roughly $250 of compute

Working alone, on open weights only, every artifact public:

**A previously uncheckable claim, quantified.** Anthropic states qualitatively that
post-training gives the workspace a "point of view." On AI2's fully open OLMo-3 family it is
measurable. Post-training moves the representation while task capability stays flat or declines
— a representational change with no competence gain — and **method sets the magnitude, not
domain**: instruction/CoT tuning moves it **~10× more than RLVR**, while varying the RL domain
at matched capability adds ~1%, and varying the *quantity* of tuning data across a 300× sweep
adds nothing measurable. The viewpoint installs early and saturates.

**Convergent validity neither source paper could claim alone.** A *linear* readout (Jacobian
lens) and a *trained neural verbaliser* (Anthropic's Natural Language Autoencoder), sharing no
machinery, recover the same content from the same activation — **42× above a mismatch null**,
with 79% of items beating every cross-prompt mispairing against a 2.4% chance rate. Made causal
by injecting a known feature: both instruments report it, and both stay silent under the
negation, a random direction, and no injection.

**A property localised to a training stage.** The base model's workspace covertly encodes
whether its own answer is wrong — beyond what its output distribution reveals — while its
verbal self-assessment is at chance. Supervised fine-tuning, not RL, makes that signal
reportable. To our knowledge the first developmental localisation of such a property on open
weights, and it answers two open questions in the metacognition survey literature.

**The reviewers' battery, executed.** All six tests Dehaene and Naccache proposed, run on open
weights, reported in full: two clean signatures, one partial, three inconclusive under
first-pass adaptations whose limitations are documented.

**Public artifacts:** 4 repositories and 4 Hugging Face datasets, Apache-2.0 — including the
first published per-head induction scores across a full training checkpoint sequence (the
flagship 2022 result whose own 34 models were internal), and a quantitative tuning atlas for the
vision model the field's founding papers used, whose numbers were never published and whose
visualisation layer has been offline since 2025.

## Why this record, and not a cleaner one

**Four of the results above are corrections or negatives, and that is the point.**

- We reported that Anthropic's NLA fails a mismatch null. **It was two bugs of ours.** Verified
  against the authors' own published worked example, the pipeline reproduces their number. The
  retraction is published.
- We reported a 4–6× mixture-of-experts penalty on lens convergence. **Widening the comparison
  set showed it was mostly model family.** A within-family control reversed it. Published.
- We found our own headline pooled layers whose measurement floors differ by an order of
  magnitude. **Correcting it halved one number and doubled another** — the method ratio went
  from ~5× to ~10×.
- We tested whether our metacognition signal improves answer selection. **It does not** (+0.008
  over baseline). Published as a bound on our own claim.

An adjudication layer that cannot adjudicate itself is worthless. The evidence that this one can
is that it has, four times, in public, against its own results.

## The proposal: a Scorecard, and the discipline that makes it credible

A standardised, controlled benchmark for open-weight models in which **every cell carries an
explicit null**, property emergence is tracked across training, and negative results are
first-class. A companion registry records each published claim as reproduced, refuted, or
inconclusive-under-control. The design follows ARC-AGI: a result registers only when it exceeds
its control.

It is complementary to Neuronpedia, which hosts interpretability artifacts but attaches no null
to them — verified: their entire feature export contains no steering measurement, Gemma Scope
lists comparing steering methods as an open problem in its own paper, and the best existing
course material states the core control as an optional bonus exercise with no code.

**Milestones (12 months).** (1) Harden the measurement tooling and contribute it upstream —
Neuronpedia is MIT-licensed, actively developed, and has publicly solicited exactly the
artifacts we produce. (2) A seed Scorecard: three consciousness indicators and three
metacognitive properties across five open-weight model families plus a checkpoint sequence.
(3) An executable implementation of the Butlin–Long indicator framework, currently applied only
in prose. (4) The public curriculum, already begun and running.

## Why it reduces risk

Metacognition is the load-bearing case. A model whose internal error signal is not reportable is
one whose stated confidence cannot be trusted — that is a reliability property before it is a
philosophical one. The same instrument surfaced a model's representation *that it was under
evaluation*, whose ablation increased dishonest behaviour: an evaluation-integrity result, and
evaluation integrity is upstream of every safety case built on evals.

The consciousness framing is a secondary implication, not the thesis. The instrument, the
controls, and the open substrate are identical either way.

## Budget

**$48,000 over 12 months** ($4,000/month).

| line | amount | note |
|---|---|---|
| Researcher time | $43,000 | the binding constraint; this work is time-limited, not resource-limited |
| Cloud compute | $5,000 | ~$250 covered everything above; this is 20× headroom for larger models and the agentic evaluation harness |

Hardware is already in place (a local GPU workstation and a DGX Spark), so the request is not
for capital. **Weekly in-person participation at the San Francisco hub**, with hub compute used
where it fits.

## Qualifications

Systems engineering, computational neuroscience, and scientific-software maintenance (I maintain
the cognitive-modelling packages — ACT-R, Soar, Nengo — for Fedora). The work engages the
neuroscience the AI claims are borrowed from, including the COGITATE adversarial collaboration,
whose own preregistered ignition prediction was *not* confirmed in humans — which constrains how
much weight the analogy can bear, and is the sort of thing an adjudication layer should say out
loud.

---

*Repositories: github.com/m9h/{jacobian-lens, spinning-up-in-mech-interp, tri-lens,
controls-and-trajectories} · datasets at huggingface.co/mhough. Every result, control and
retraction reproducible from published open artifacts, without access to any closed model.*
