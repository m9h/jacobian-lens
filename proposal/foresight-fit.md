# Foresight "AI for Science & Safety Nodes" — fit assessment

**2026-08-02.** [foresight.org/grants/grants-ai-for-science-safety](https://foresight.org/grants/grants-ai-for-science-safety/)
· $10k–$100k (higher for safety-oriented areas) · ~$3M/yr · `grants@foresight.org`
· Process being updated; **deadlines announced "in the coming days"** — so there is time to
prepare, and a reason to prepare now.

## Why this fits better than the Longview RFP did

Longview's Digital Minds call was **moral-status framed**, which pushed the proposal toward
consciousness indicators — the half of the Scorecard that, as the RewardBench analysis showed,
*cannot* demonstrate predictive validity. Foresight's stated criteria point the other way, and
they favour what we have actually built:

| their criterion | what we can put in front of them |
|---|---|
| **Preference for open-source** | 4 public repos, 4 HF datasets, Apache-2.0, every result reproducible from published artifacts — shipped, not promised |
| **AI for Science & Epistemics** | the Scorecard *is* an epistemics instrument: claims adjudicated against explicit nulls |
| **AI safety** | metacognition → calibration/selective prediction; the evaluation-awareness result whose ablation increased dishonest behaviour |
| **1–3 year milestones** | the Scorecard is naturally milestoned (seed cells → registry → adoption) |
| **High-risk, high-reward** | an adjudication layer only pays off if the field adopts it |
| Individuals and small orgs eligible | OREL qualifies; both non-profit and for-profit accepted |

**Target focus area: "AI for Science & Epistemics."** Not the Neuro/BCI area despite the
neuroscience content — that area is about mapping and simulating biological intelligence, not
about importing neuroscience methods to evaluate AI.

## The pitch, and why our failures are the strongest part of it

For an *epistemics* funder, the track record to lead with is not "we got results" but "we
built a process that catches wrong results, including our own":

- **Refuted** a published claim (society-of-thought accuracy reverses across benchmarks).
- **Confirmed and quantified** a claim its authors could only state qualitatively, on open
  weights — then **corrected our own headline** when a per-layer floor showed RLVR's effect was
  half noise, which *strengthened* the method ratio from ~5× to ~10×.
- **Retracted two of our own results**: an NLA false negative traced to two bugs of ours, and a
  MoE sparsity claim that was mostly model family. Both retractions are published.
- **Established a convergent-validity result** neither source paper could claim alone: a linear
  readout and a trained verbaliser recover the same content 42× above a mismatch null.
- **Published the negative** that bounds our own metacognition claim (no best-of-N gain).

Most grant applications cannot show a retraction. For a programme funding *epistemics*, the
ability to demonstrate self-correction under public artifact is the differentiating evidence.

## ⚠ The one condition that could disqualify us

> "We prioritize AI-first projects that want to be active, **in-person** members of one of our
> hubs" (San Francisco or Berlin). "We accept 'funding-only' projects only in exceptional cases."

This is the binding constraint, not the science. **It needs an answer before drafting.** Options,
in order of strength:

1. **In-person at the SF hub** if that is geographically workable — strongest, and unlocks their
   local compute.
2. **Partial residency** — the work is compute-portable (everything already runs on Modal from a
   laptop), so a defined in-person cadence is more credible here than for a wet-lab project.
3. **Argue the exception** — weakest, and their wording suggests it is rare.

## Framing risk

Their evaluation criterion is *"impact on reducing AI existential risks"*. The consciousness /
moral-status framing does not map onto that; the **reliability** framing does — a model whose
internal error signal is not reportable is one whose stated confidence cannot be trusted, and
an evaluation-awareness representation is directly an eval-integrity problem. Lead with
epistemics and reliability, keep moral status as a secondary implication rather than the thesis.

## Next actions

1. **Resolve the hub question** — everything else is downstream of it.
2. Watch for the deadline announcement; the application is an Airtable form plus an itemised
   budget and project plan.
3. Reuse `longview-scorecard.md` as the substrate but **re-frame the opening** from moral status
   to epistemics, and add the artifact list and the self-correction record, neither of which
   existed when that draft was written.
