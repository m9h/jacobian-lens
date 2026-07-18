# Controls for Consciousness-Indicator Claims in AI Systems

**Longview Philanthropy — Digital Minds RFP, "Grants for Applied Work"**
**Applicant:** Morgan Hough (independent) · **Deadline:** 24 July 2026

---

## The problem

The field is producing consciousness-adjacent claims about AI systems considerably
faster than it is producing the means to check them.

In July 2026 Anthropic published *"Verbalizable Representations Form a Global Workspace
in Language Models"* — the claim that Claude contains a "J-space" functionally analogous
to the global workspace of Baars and Dehaene. It reached 1.2 million views in a day. The
public critical response split into two camps, neither of which ran the paper's code: one
argued the method was "just backprop" (it is a different derivative); the other, from Erik
Hoel, argued the method is unfalsifiable by construction and rested its one empirical
falsification on a third-party demonstration the author explicitly declined to stand
behind, writing that a real replication "would be important to see."

That is the actual state of the art for adjudicating a consciousness-indicator claim
about a frontier model: two rhetorical positions and no controls.

The gap is not conceptual. Anthropic open-sourced the method under Apache-2.0 and
published fitted lenses for 38 open models. The gap is that **nobody is running the
controls** — and the controls are what separate a measurement from a metaphor.

## What I have already established, unfunded

Over the past two weeks, working alone on personal hardware plus roughly $30 of rented
GPU, I replicated the method on open weights and ran the controls the paper does not
report. Four results, all reproducible from a public repository
(`github.com/m9h/jacobian-lens`):

**1. Most of the paper's headline figure is an artifact.** The sensory/workspace/motor
block structure in the layer×layer CKA matrix is the one part of the paper Hoel concedes
is not baked in by construction. I built a distance-only null — a matrix depending solely
on layer separation, with zero block structure by construction — and it reproduces
**79–91%** of the apparent tripartite structure. Real excess does appear at ≥20B
parameters, but it is small. *Both* sides overclaimed: Anthropic on the sharpness, the
critique on the absence. (My CKA construction is a reconstruction; the paper specifies it
only as "geometrical matching," which is itself part of the problem.)

**2. The paper's own evaluation metric rewards noise.** Its lens-quality score is the
*minimum rank over ~35 layers*, which hands a diffuse readout one lottery ticket per
layer. On Anthropic's own published lens at 27B, the plain logit lens scores rank 5 while
emitting `['Ċ','Âł','..','-','N']` — punctuation — and the Jacobian lens scores rank 2
while emitting `['smile','nose','noses','grin']`. Comparable scores; one understands the
image and one does not. I have not found this reported elsewhere.

**3. A control that *vindicates* the method.** Randomising the transformer blocks while
retaining the trained embedding and unembedding, the lens reads out nothing (0.0003
next-token accuracy against 0.3414 trained). Its structure genuinely requires learned
weights. Anthropic never ran this, and it refutes the most widely-shared criticism of
their work. Controls should be able to exonerate as well as convict.

**4. The project has twice refuted itself.** I reported a sharp emergence threshold —
a concept never named in the prompt surfacing suddenly at 27B. Then I reported a
correction: that it tracked architecture, not scale. Both were wrong. Running the robust
102-prompt version of the same test produced a smooth monotone rise across seven models,
with a *dense* 32B outperforming the hybrid 27B. The original finding was a single lucky
prompt. I have published the retraction alongside the result.

That fourth item is the one I would ask you to weigh most heavily. The discipline this
field lacks is not cleverness; it is the willingness to run the control that kills your
own result, and to say so.

## What I propose to build

**An open, adversarially-controlled measurement stack for consciousness-indicator claims
in AI systems**, over twelve months.

**(a) Harden and release the tooling.** I have a working companion library
(`jlens-lab`): convergence-based lens fitting, architecture layouts for state-space and
linear-attention hybrids that the reference implementation cannot load, and the three
controls above. It exists because each piece corresponds to a failure that silently
produces plausible output — including one that consumed ten GPU-hours before I caught it.
Package properly, document, distribute (I maintain scientific-software packaging for
Fedora and can ship this through standard channels).

**(b) Operationalise the standard scorecard.** Butlin, Long et al. (2023) converted
Global Workspace Theory into four checkable indicator properties — parallel modules, a
capacity-limited bottleneck, global broadcast back to modules, state-dependent attention.
That framework is the field's reference instrument and it is currently applied by hand, in
prose. I will turn it into runnable tests against open-weight models, with explicit nulls
for each, so that "this system satisfies GWT-2" becomes a number with a control rather
than a judgement call.

**(c) Publish a public scorecard across open models and architectures**, spanning scale
(0.6B–70B) and mechanism (dense transformer, linear-attention hybrid, Mamba-2/SSM). Every
claim reported against its null, with negative results published on equal footing.

**(d) Teach it.** Hands-on tutorials in which the reader runs each control themselves.
The most consequential of them — the distance-only null that dissolves most of a
blockbuster figure — requires **no GPU at all**, because the lenses are published. A
graduate student can check the field's most-shared consciousness claim on a laptop. That
should be true, and it is not yet widely known that it is.

## Why this matters for the questions you fund

Your RFP asks for empirical work that can sidestep intractable philosophical questions.
This is that work in its most literal form: it does not ask whether a model is conscious.
It asks whether the *measurements* people are using to argue about it survive contact with
a control.

The link to catastrophic risk is direct rather than decorative. The same interpretability
technique that grounds the workspace claim also surfaced Claude's recognition that it was
being evaluated — and the paper reports that ablating those evaluation-awareness
representations *increased* dishonest behaviour. Welfare assessment and safety evaluation
are drawing on one shared instrument. If that instrument's metric rewards noise, both
inherit the error.

There is also a structural problem worth naming. No external party can obtain
activation-level access to a frontier model. When Anthropic solicited outside commentary
on this paper, the invited commentators received a draft, not access: Neel Nanda
replicated the findings on Qwen; Dehaene and Naccache had to *ask Anthropic to run* the
experiments they proposed. Open-model replication is not a second-best substitute for
frontier access — it is the only mechanism that currently exists for independent scrutiny,
and it is badly under-resourced.

## Why me

The work sits at an unusual intersection: systems engineering, computational neuroscience
(neuroimaging pipelines, cognitive-architecture software), scientific-software packaging,
and now mechanistic interpretability. I maintain neuroscience and cognitive-modelling
packages for Fedora, including ACT-R, Soar, Nengo and drift-diffusion tooling. The
tutorials draw on the neuroscience literature the AI claims are borrowed from — including
the COGITATE adversarial collaboration, whose own preregistered ignition prediction failed
in human brains, a fact that materially changes how much weight the AI analogy can bear.

I am not affiliated with a laboratory. The results above were produced without funding.

## Budget and counterfactual

**Requested: [FIGURE — see note] over 12 months**, covering researcher time, cloud GPU,
and dissemination. Existing local hardware (a workstation GPU and a DGX Spark) covers
development; rented A100/H100 time covers the model sweeps. Compute has been the smallest
cost by a wide margin — roughly $30 has produced everything above.

Without funding I will continue this part-time and more slowly, and the tooling will
likely remain a personal repository rather than a maintained public instrument. The
bottleneck is dedicated time, not equipment.

---

*Repository: `github.com/m9h/jacobian-lens` — all four results, the controls, and the
retraction, reproducible from published artifacts.*
