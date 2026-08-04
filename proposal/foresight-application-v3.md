# Fund the Layer Nobody Funds: Adjudication for Open Mechanistic Interpretability

**Foresight Institute — AI for Science & Safety Nodes** · focus: **AI for Science & Epistemics**
· Morgan Hough, Orthogonal Research and Education Lab (OREL) · San Francisco hub, weekly in
person · **DRAFT v3 — reframed from benchmark attribution to the open mech-interp stack**

---

## The field went mainstream on foundations that are shakier than the coverage

In 2009 a dead Atlantic salmon was placed in an fMRI scanner and shown photographs of humans in
social situations. Standard analysis found significant activation in its brain. The fish was
dead. The study won an Ig Nobel and permanently changed how an entire discipline handles multiple
comparisons — **a negative result that improved a field more than most positive ones.**

[*The Dead Salmons of AI Interpretability*](https://arxiv.org/abs/2512.18792) (Méloux, Dirupo,
Portet & Peyrard, 2025) shows the AI analogue is not hypothetical: **feature attribution, probing,
sparse autoencoding and even causal analyses all produce plausible-looking explanations of
randomly initialized networks** — models that cannot contain the structure being reported. Its
prescription is that an interpretability result is a **parameter of a statistical model inferred
from computational traces**, and needs alternative hypotheses, identifiability and quantified
uncertainty rather than a picture and a name.

That is the whole of this proposal, and it is not our idea. **Neuroimaging needed a dead fish and
roughly a decade. Interpretability now has the paper; what it does not have is anyone whose job is
to run the test.**

Meanwhile MIT Technology Review named mechanistic interpretability a **2026 Breakthrough
Technology** — niche to essential in about two years, with Anthropic, Google DeepMind and
well-capitalised startups all investing heavily. In the same window its central results have been
failing their controls:

- **Anthropic's circuit tracing on Claude 3.5 Haiku produced satisfying insight for about a
  quarter of tested prompts.** DeepMind's months-long Chinchilla circuit analysis produced a
  brittle, partial explanation.
- **Sparse autoencoders — the field's flagship method — are not beating neurons.**
  [MIB](https://arxiv.org/abs/2504.13151) (ICML 2025) found it by benchmarking against a private
  test set; [Transluce](https://arxiv.org/abs/2601.22594) (2026) found it by building better
  neuron circuits and reproducing three of Anthropic's own case studies without any learned
  dictionary. **Two unrelated methods, same negative.**
- DeepMind has publicly shifted from *"ambitious reverse-engineering"* to *"pragmatic
  interpretability"* — useful-but-imperfect understanding. That is a retreat, correctly made, and
  it went largely unremarked outside the field.

None of this is a reason to stop funding interpretability. It is a reason to fund the part of it
that establishes which results survive.

## The open stack is real, and it is funded at two of its three layers

| layer | who | scale |
|---|---|---|
| **Access** — run experiments on model internals at scale | [NDIF / `nnsight`](https://github.com/ndif-team), Bau Lab @ Northeastern | **$9M NSF**; 110+ papers at ICLR/NeurIPS/ICML/EMNLP; a European counterpart (eDIF) now exists |
| **Tooling** — the libraries and platform everyone uses | [Decode Research](https://www.decoderesearch.org/) — Neuronpedia, SAELens, circuit-tracer, SAEDashboard | funded by Open Philanthropy, LTFF, AISTOF, Anthropic, Manifund |
| **Commercial** | Goodfire (Ember), Tilde, Apollo | Goodfire raised a **$50M Series A** |
| **Theory & development** | [Timaeus](https://timaeus.co/) (SLT, `devinterp`), EleutherAI ("Interpreting Across Time") | nonprofit, Discord, open project boards |
| **★ Adjudication** — does a published claim survive a control? | **nobody** | **—** |

That last row is not my editorialising. It is the field's own finding.

## The gap has been formally named, and left unbuilt

***Make Mechanistic Interpretability Auditable: A Call to Develop Guidelines via Continuous
Collaborative Reviewing*** ([arXiv 2606.00033](https://arxiv.org/abs/2606.00033), Lan et al.,
**ACL 2026**) argues that MI lacks any standardised auditing system, and that stakeholders in
safety-critical settings cannot verify findings. Its motivating example is precise: **conflicting
papers on the same behaviour that were "partially correct but incomparable due to methodological
inconsistencies."**

It proposes three mechanisms:

1. a **continuous collaborative reviewing platform** hosting the meta-science that does not fit in
   papers — critiques, **negative findings**, reproductions, partial results;
2. **expert-verified guidelines** generalised from what accumulates there;
3. **source-based auditing** — dependency chains showing which claims hold up other claims.

It is a **position paper**. Nothing is implemented; the authors explicitly invite "constructive
debate over the necessity, design and implementation."

**So the need is peer-reviewed, published at a top venue, and unmet.** This proposal is the
implementation, run as a working group rather than a website — because the hard part is not
hosting critiques, it is *producing* them, and that requires people who will do the reproduction.

## We have been doing this unfunded, and the receipts are the negatives

Every artifact public, open weights only, on **~$250 of compute**.

**A published claim, reproduced, that reverses.** A 2026 paper reported that steering a single
feature produced a "society of thought" improving reasoning accuracy. On the paper's own
benchmarks the gain **inverts** — +10 points on one, −22 on another. The original had no public
code or data. Ours does.

**Two independent instruments made comparable.** We put the Jacobian lens and a natural-language
activation reader on the same footing and showed they converge **42× above a mismatch null**, both
detecting known injected content. That is exactly the "partially correct but incomparable" problem
the ACL paper names, resolved by construction.

**Method separated from scale.** Across the fully open OLMo-3 family, instruction tuning moves the
model's internal representation ~10× more than RLVR while task capability stays flat — and varying
tuning-data *quantity* across a 300× sweep changes nothing measurable.

**Three corrections and retractions of our own, published.** A false negative about another lab's
method, traced to two bugs of ours. A sparsity claim that turned out to be model family. A
headline effect that pooled measurements whose noise floors differed by an order of magnitude —
correcting it halved one number and doubled another.

**We have already run a dead-salmon test, and published the result.** Applying the Adebayo
randomization control to the Jacobian lens — replace trained blocks with random ones and see
whether the instrument still reads out content — the lens **passes**: random blocks read out
nothing. That is a *positive* data point in a literature that is mostly negative, and it is
exactly the kind of entry the register below would hold.

**A gate that caught us.** Last week we built a tool gate asking whether an external line of
evidence recovers a signal we had measured. Within an hour it found a **live bug in our own
published curriculum**: a steering coefficient calibrated on GPT-2's attention sink, over-steering
**28×** while the code printed the words *"at the feature's natural activation scale."* The
control logic survived; the spectacular demo did not. It is now
[PITFALLS #24](https://github.com/m9h/spinning-up-in-mech-interp/blob/master/PITFALLS.md), one of
24 entries each recording a wrong result we produced and caught.

That last item is the qualification that matters. **An adjudication group that has never published
a retraction of its own is not doing adjudication.**

## Proposed work

**1. Tool gates — hold the tooling to the standard we hold papers to.** The open stack is
recommended on *maintenance* evidence: the repo is alive, it has stars. That is not evidence a
tool works. A gate holds an external line of evidence against a number someone already measured
and fails loudly when they disagree. One exists and has already paid for itself. Deliverable: a
gate suite across the stack Decode Research maintains, contributed upstream. *This is the concrete
answer to the ACL paper's mechanism (2).*

**2. A public register of interpretability claims and their controls** — what was claimed, what
control would discriminate it, whether anyone has run one, what happened. Prospective, so targets
are nominated before results are known. Seeded with the ~20 claims we have already adjudicated,
including our own reversals. *Mechanisms (1) and (3).*

**3. Reproductions the field has asked for and not received.** Two are already scoped and
unclaimed: Timaeus's own project board lists **vision-circuit development** (our rung 1 measures
InceptionV1 tuning with two nulls; they want the training axis) and a **review of
complexity measures** — where we have a worked confound to contribute, having watched a difference
matrix look strikingly low-rank until we checked that the matrix it came from already was. We
would also run the side-by-side nobody has: our behavioural induction phase-change across Pythia
checkpoints, seeds and sizes against Timaeus's weight-space rLLC detector.

**4. Benchmark attribution as one worked case.** Reasoning benchmarks are where interpretability
claims meet forecasting. ARC-AGI's API and Kaggle tracks run **identical tasks** and differ by
**~68 points** on compute budget and harness alone — the cleanest published capability-vs-scaffold
decomposition in evaluation, and nobody analyses it as one. ARC Prize publishes per-task pass/fail
with costs for **77 models × 400 tasks**; most of that analysis is undone and needs no GPU.

**5. ★ The gap nobody's toolkit covers: agents.** Every method in the open stack — attribution
graphs, SAEs, lenses, probes — was built for **a single forward pass**, while deployment has moved
to agents running for hours over many decisions. Martian's [ARES](https://github.com/withmartian/ares)
(open-source, actively developed, Terminal-Bench 2.0 and 36+ Harbor task packs) exists partly to
support interpretability of *sequential decision-making*, and its central design choice — "the LLM
itself is the agent, not the scaffolding" — **is the model-versus-harness boundary made into an
architecture**, which is item 4's question with an intervention handle on it. We already author
scientific tasks in the Harbor format
([m9h/terminal-bench-science](https://github.com/m9h/terminal-bench-science): the NODDI diffusion
and transcranial-ultrasound challenges), and Martian solicits exactly that. Contributing a task
pack is a first deliverable that costs almost nothing and opens the harder question: what does a
*control* even look like for a claim about an agent's reasoning over a hundred steps?

**6. The reproduction group at the hub** — weekly, in person, which is what produces 1–5.

## Why this reduces risk, and why it is cheap

Safety cases rest on evaluations, and increasingly on interpretability evidence. If a method's
apparent success is an artifact of its own intervention — as ours was, by 28× — then an audit
built on it is unfounded. **We have already surfaced a model's internal representation *that it
was being evaluated*, whose ablation increased dishonest behaviour.** Evaluation integrity is
upstream of every safety argument; adjudication is upstream of evaluation integrity.

And the economics are unusual in a way that favours this ask. **The access layer needed $9M and
the commercial layer $50M because they are compute- and product-intensive. Adjudication is
neither.** It is judgment, controls, and the willingness to publish a negative. Everything above
cost ~$250 of compute. **$48,000 over 12 months** ($4,000/month: **$43,000** researcher time,
**$5,000** compute — 20× headroom on the current burn; hardware is in place). **The ask is time,
not resources.**

## The in-person activity

A **reproduction group** at the San Francisco hub: papers with incomplete artifacts get a public
open-source implementation, training documentation, and the control the original omitted. It
continues an existing SF cognitive-science reading group with a thirty-session, three-year record,
organised on the 1978 cognitive-science hexagon — because most of these claims import a construct
from a discipline that is not in the room.

We expect to hand work upstream rather than accumulate it: Decode Research states it is "always
looking for new partners," Timaeus runs a public project board with a stated protocol for claiming
work, and EleutherAI runs an open mentorship programme. **The collaborators exist and are asking.
What is missing is somebody whose actual job is to check.**

## Who does the work

An adjudication group is only as good as the people willing to run someone else's experiment and
publish a negative. Three channels, in order of how targeted they are:

- **★ The authors of the call, who are also a funder.** Lan et al. (ACL 2026) invite debate on
  the design and implementation of exactly this — and Oozeer, Quirke and Abdullah are at
  **[Martian](https://withmartian.com/prize)**, which runs a **$1M interpretability prize**
  awarding both promising directions and *completed work*, plus hackathons with Apart Research.
  Their commercial product is LLM routing, so their stake in interpretability being *auditable*
  is structural rather than rhetorical. Approaching the people who published the call, with the
  implementation, is a better first move than any broadcast.
- **Named unclaimed work, not open invitations.** Timaeus runs a public project board — its
  *vision-circuit development* project is in progress **with no lead listed**, and our rung 1
  already measures InceptionV1 tuning against two nulls. Decode Research states it is "always
  looking for new partners." MIB will score a submission against a private test set. Every one of
  these is a specific thing a specific person can own, which is what MedARC got right and what
  reading groups without projects get wrong.
- **[BuzzRobot](https://buzzrobot.substack.com/about)** — a community founded by an OpenAI alum
  that runs talks with researchers from DeepMind, OpenAI, Meta and NVIDIA, and has published on
  interpretability. Reach into the right population. The talk we would give is not a recruitment
  pitch but a failure report: *we built a gate to check interpretability tooling and it
  immediately found a live bug in our own published curriculum.* That self-selects for people who
  find checking interesting — the one trait this work requires and cannot instil.

## On the NeuroAI framing, honestly

Foresight's reference is Mineault et al., *NeuroAI for AI Safety* (arXiv:2411.18526). Four of its
five paths are architectural bets that brain-likeness confers safety. **We are not making that
bet.** Only path 4 — interpretability using neuroscience methods — is load-bearing here, and even
there the import is *measurement discipline*, not brain structure: construct validity from
psychometrics, meta-d′ and type-2 ROC from signal detection theory, within-item controls from
psychophysics, and adversarial collaboration from fields that built it after a replication crisis.
AI evaluation is pre-crisis. That is the whole argument.

## Qualifications

Systems engineering, computational neuroscience, and scientific-software maintenance (I maintain
ACT-R, Soar and Nengo packages for Fedora). The work engages the neuroscience these claims borrow
from — including the COGITATE adversarial collaboration, whose own preregistered prediction was
*not* confirmed in humans, which is the sort of thing an adjudication project should be willing to
say out loud about its own source disciplines.

---

*github.com/m9h/{jacobian-lens, spinning-up-in-mech-interp, tri-lens, controls-and-trajectories}
· huggingface.co/mhough. Every result, control and retraction reproducible from published open
artifacts.*
