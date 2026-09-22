# Fund the Layer Nobody Funds: Adjudication for Open Mechanistic Interpretability

**Foresight Institute — AI for Science & Safety Nodes** · focus: **AI for Science & Epistemics**
· Morgan Hough, Orthogonal Research and Education Lab (OREL) · San Francisco hub, weekly in person
· **$48,000 / 12 months**

*Supporting detail, full landscape survey and evidence tiering:
[foresight-application-v3.md](foresight-application-v3.md) · claim-by-claim source audit:
[CITATION_AUDIT.md](CITATION_AUDIT.md)*

---

## The problem

In 2009 a dead salmon was placed in an fMRI scanner and shown photographs of humans in social
situations. Standard analysis found significant brain activation. The fish was dead. That negative
result changed how an entire discipline handles multiple comparisons.

[*The Dead Salmons of AI Interpretability*](https://arxiv.org/abs/2512.18792) (Méloux et al., 2025)
shows the AI analogue is not hypothetical: **feature attribution, probing, sparse autoencoding and
even causal analyses all produce plausible explanations of randomly initialized networks** — models
that cannot contain what is being reported.

**I am a neuroscientist and computational psychiatrist. The salmon is my field's scandal, not a
metaphor I found useful.** Neuroimaging answered it with cluster correction, preregistration,
adversarial collaboration and multi-site replication, and it worked. **I propose to import the
reform, not the neuroscience.**

Meanwhile MIT Technology Review named **"Mechanistic interpretability"** a 2026 Breakthrough
Technology, while the field's own results fail their controls — Anthropic reports attribution
graphs giving *"satisfying insight for about a quarter of the prompts we've tried"*; Google
DeepMind's interpretability team publicly pivoted *"from ambitious reverse-engineering to…
pragmatic interpretability"*, naming their own 2024 sparse-autoencoder programme as the failure;
and **SAEs are not beating neurons**, found twice by unrelated routes.

## The gap, stated narrowly

The open stack is **funded at every layer but one**: NDIF/`nnsight` holds access (**$9M NSF**),
Decode Research holds tooling (Neuronpedia, SAELens, circuit-tracer, and now the inference engine),
Goodfire holds the commercial layer (**$50M Series A**), Timaeus and EleutherAI hold theory.

Adjudication *is* done — but episodically, by the interested party, and only where someone happens
to care. Anthropic published 22k words of solicited critique alongside its global-workspace paper;
a lone researcher ran a preregistered reliability campaign on Gemma-4-12B. **Nobody's actual job is
checking your results.**

This has been formally requested. ***Make Mechanistic Interpretability Auditable*** ([arXiv
2606.00033](https://arxiv.org/abs/2606.00033), **ACL 2026**) calls for a continuous collaborative
reviewing platform for critiques, negative findings and reproductions; expert-verified guidelines;
and source-based auditing. **It is a position paper. Nothing is implemented, including by its
authors** — several of whom are at Martian, which runs a **$1M interpretability prize** awarding
completed work.

## Why me: the function has already run on my own work, twice, in two months

- **A stranger reading our lockfile found a real error in our published artifacts.** Our 11 OLMo-3
  Jacobian lenses encoded a library bug applying rope-scaling to **24 of 32 layers that should not
  have had it**. We confirmed it, published an erratum, annotated ten result files, pinned the
  dependency — and then **bounded the damage for free** against a corrected lens published by
  another group, showing the effect sits *inside our own refit noise* and the conclusions stand.
- **Someone else's control invalidated a specific claim of ours.** An answer-readout confound
  applies exactly to our unanswerable-detection control; our main results are clean and sit inside
  their honest band. **That claim is now marked as unsupported.**

Critically: **our own cross-validation could not have caught the first one**, because both lenses
compared shared the same wrong convention. That is the core methodological finding — *a comparison
between two implementations tests correctness only if they differ in the way that could be wrong* —
and it is now a public teaching entry.

**An adjudication group that has never published a retraction of its own is not doing
adjudication.** We have published four, on ~**$250 of compute**.

## The work: 12 months, model scale

1. **Tool gates** — hold the open stack to the standard it holds papers to. One exists and
   immediately found a live 28× calibration bug in our own curriculum. Contributed upstream.
2. **A public register of claims and their controls** — prospective, so targets are nominated
   before results are known. Seeded with ~20 claims we have already adjudicated, our own reversals
   included.
3. **Reproductions the field asked for and did not get**, including unclaimed projects on Timaeus's
   own board that our existing artifacts already address.
4. **A red–green testbed with synthetic ground truth.** The salmon gives only the *null*. Tracr
   compiles a known program *into* transformer weights and gives the **positive control** — bracket
   a method between them and you can state its error rate instead of its plausibility.
5. **The reproduction group at the hub**, weekly and in person, which produces 1–4.

**Why model scale, deliberately.** Agents are where the field is going and where its toolkit does
not reach — every method in the open stack assumes a single forward pass. But **an adjudication
method must be shown to catch real errors somewhere cheap before it is trusted somewhere
expensive**, and ours has. Phase 2 is named, not asked for.

## Why it is cheap

Access needed $9M and commercial $50M because they are compute- and product-intensive.
**Adjudication is neither** — it is judgment, controls, and willingness to publish a negative.
Everything above cost ~$250 of compute.

**$48,000 over 12 months**: $43,000 researcher time, $5,000 compute — 20× headroom on current burn;
hardware in place. **The ask is time, not resources.**

## Qualifications

Neuroscientist and computational psychiatrist — mechanistic inference on neuroimaging data, the
discipline that had this exact crisis and fixed it. **Executive Director and President of Société
BCI Montréal (NeuroTechX)**, so running a distributed technical community is my existing job, not a
skill this grant would fund me to acquire. Co-run a **San Francisco cognitive-science reading group
with a thirty-session, three-year record**. Maintain ACT-R, Soar, Nengo, pyactr and pyDDM for
Fedora — keeping other people's research code running is the same skill the tool gates require.

The field is importing constructs from cognitive science and statistics — *representation*,
*mechanism*, *report*, *metacognition* — mostly without anyone in the room who has had to defend
one to a reviewer. **I have, and the fields I come from already paid for the lesson this one is
about to learn.**

---

*github.com/m9h/{jacobian-lens, spinning-up-in-mech-interp, tri-lens, controls-and-trajectories} ·
huggingface.co/mhough. Every result, control and retraction reproducible from published open
artifacts.*
