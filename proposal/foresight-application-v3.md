# Fund the Layer Nobody Funds: Adjudication for Open Mechanistic Interpretability

**Foresight Institute — AI for Science & Safety Nodes** · focus: **AI for Science & Epistemics**
· Morgan Hough, Orthogonal Research and Education Lab (OREL) · San Francisco hub, weekly in
person · **DRAFT v3 — reframed from benchmark attribution to the open mech-interp stack**

---

## The field went mainstream on foundations that are shakier than the coverage

In 2009 a dead Atlantic salmon was placed in an fMRI scanner and shown photographs of humans in
social situations. Standard analysis found significant activation in its brain. The fish was dead.
The study won an Ig Nobel and permanently changed how a discipline handles multiple comparisons —
**a negative result that improved a field more than most positive ones.**

[*The Dead Salmons of AI Interpretability*](https://arxiv.org/abs/2512.18792) (Méloux et al., 2025)
shows the AI analogue is not hypothetical: **feature attribution, probing, sparse autoencoding and
even causal analyses all produce plausible-looking explanations of randomly initialized networks** —
models that cannot contain what is being reported. Its prescription is that an interpretability
result is a parameter of a statistical model inferred from computational traces, needing
alternative hypotheses, identifiability and quantified uncertainty rather than a picture and a name.

That is this proposal, and it is not our idea. **Neuroimaging needed a dead fish and a decade.
Interpretability already has the paper; what it lacks is anyone whose job is to run the test.**

For me this is not a borrowed metaphor. **I am a neuroscientist and computational psychiatrist; the
salmon is my field's scandal, not an analogy I found useful.** I have spent my career doing
mechanistic inference on neuroimaging data — the original setting where a plausible mechanism story,
a compelling picture and an uncorrected statistic combined to produce findings that were not there.
Neuroimaging responded with cluster correction, preregistration, adversarial collaboration and
multi-site replication, and it worked. **AI interpretability is currently pre-reform, running the
same failure mode on a substrate where the ground truth is more accessible, not less.** I am
proposing to import the reform, not the neuroscience.

Meanwhile MIT Technology Review named mechanistic interpretability a **2026 Breakthrough
Technology**. In the same window its central results have been failing their controls: Anthropic's
circuit tracing on Claude 3.5 Haiku gave satisfying insight for **about a quarter** of tested
prompts; DeepMind's months-long Chinchilla analysis was brittle and partial, and DeepMind has
publicly retreated from *"ambitious reverse-engineering"* to *"pragmatic interpretability"*; and
**sparse autoencoders are not beating neurons**, found twice by unrelated routes —
[MIB](https://arxiv.org/abs/2504.13151) by benchmarking against a private test set (ICML 2025) and
[Transluce](https://arxiv.org/abs/2601.22594) by building better neuron circuits and reproducing
three of Anthropic's own case studies without any learned dictionary (2026).

None of this argues for funding interpretability less. It argues for funding the part that
establishes which results survive.

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

- **A published claim that reverses on reproduction.** A 2026 paper reported that steering one
  feature produced a "society of thought" improving reasoning. On the paper's own benchmarks the
  gain **inverts** — +10 on one, −22 on another. The original had no public code; ours does.
- **Two instruments made comparable.** The Jacobian lens and a natural-language activation reader,
  put on the same footing, converge **42× above a mismatch null**. That is precisely the
  "partially correct but incomparable" problem the ACL paper names, resolved by construction.
- **Method separated from scale.** Across OLMo-3, instruction tuning moves internal representation
  ~10× more than RLVR while capability stays flat; varying tuning-data *quantity* across a 300×
  sweep changes nothing measurable.
- **A dead-salmon test, passed and published.** Under the Adebayo randomization control the
  Jacobian lens reads out **nothing** from random blocks — a *positive* data point in a
  mostly-negative literature, and exactly what the register below would hold.
- **Three corrections and retractions of our own.** A false negative about another lab's method,
  traced to two bugs of ours; a sparsity claim that was really model family; a headline effect
  that pooled noise floors differing by an order of magnitude.
- **A gate that caught us.** Built last week; within an hour it found a live bug in our own
  published curriculum — a steering coefficient calibrated on GPT-2's attention sink,
  over-steering **28×** while the code printed *"at the feature's natural activation scale."* Now
  [PITFALLS #24](https://github.com/m9h/spinning-up-in-mech-interp/blob/master/PITFALLS.md), one of
  24 entries each recording a wrong result we produced and caught.

**An adjudication group that has never published a retraction of its own is not doing
adjudication.**

## Proposed work: cheap now, and the only route to the expensive thing

**The scoping is deliberate.** Model-level interpretability is where the methods exist and compute
is nearly free. Agents are where the field is going and where its toolkit does not reach. This
grant funds the first **because it is the positive control for the second** — an adjudication
method must be shown to catch real errors somewhere cheap before it is trusted somewhere
expensive. Ours has: a gate built last week found a live 28× bug in our own curriculum, on a
laptop, in under an hour. Auditing agent traces with an unvalidated gate would repeat exactly the
mistake we have already published a post-mortem about.

### Phase 1 — this grant, 12 months, model scale

1. **Tool gates.** The open stack is recommended on *maintenance* evidence — the repo is alive, it
   has stars. That is not evidence a tool works. A gate holds an external line of evidence against
   a number someone already measured and fails loudly when they disagree. One exists and has
   already paid for itself; deliverable is a suite across the stack Decode Research maintains,
   contributed upstream. *ACL mechanism (2).*
2. **A public register of claims and their controls** — what was claimed, what control would
   discriminate it, whether anyone ran one, what happened. Prospective, so targets are nominated
   before results are known. Seeded with the ~20 claims we have adjudicated, our reversals
   included. *Mechanisms (1) and (3).*
3. **Reproductions the field asked for and did not get.** Timaeus's board lists **vision-circuit
   development with no lead** — our rung 1 already measures InceptionV1 tuning against two nulls;
   they want the training axis — and a **review of complexity measures**, where we can contribute
   a worked confound: a difference matrix that looked strikingly low-rank until we checked the
   matrix it came from already was. Plus the side-by-side nobody has run, our behavioural
   induction phase-change across Pythia checkpoints and seeds against Timaeus's weight-space rLLC.
4. **The harness, measured.** ARC-AGI's API and Kaggle tracks run **identical tasks** and differ
   by **~68 points** on compute budget and scaffold alone — the cleanest published
   capability-vs-scaffold decomposition in evaluation, and nobody analyses it as one. ARC Prize
   publishes per-task results for **77 models × 400 tasks**; that analysis needs no GPU and is the
   empirical base for §Phase 2.
5. **★ A red–green testbed with synthetic ground truth.** The dead-salmon argument supplies only
   the *null* — a randomized network, where a sound method must find nothing. The positive control
   already exists and is barely used: **[Tracr](https://arxiv.org/abs/2301.05062)** compiles a
   human-written RASP program *into* transformer weights, so the ground truth is known by
   construction ([TracrBench](https://arxiv.org/abs/2409.13714) is a dataset of them). Bracket a
   method between the two — *find nothing on the random net, find exactly the program on the
   compiled one* — and you can finally state a method's error rate rather than its plausibility.
   This is question zero of our own reading checklist, answered with material the field has had
   since 2023. Timaeus lists a Tracr project as **unclaimed**.
6. **The reproduction group at the hub**, weekly and in person, which is what produces 1–5.

### Phase 2 — the harness is where the frontier is

**This is not a guess about the future; ARC-AGI has already run the experiment.**

| | what moved | what it measures |
|---|---|---|
| **ARC-AGI-1** | saturated at **98%**, $0.52/task against a $17 human baseline | harness on a *frozen* 8B: **+53.3 pts**. Base-model swap 3B→8B: **+3.4** |
| **ARC-AGI-2** | API **92.5%** vs Kaggle-constrained **~24%** | **68 points on identical tasks**, from compute budget and scaffold alone |
| **ARC-AGI-3** | **<1% in March 2026 → ~30% by July** | the benchmark itself went **interactive and agentic** |

The tell is in the name: **ARC-AGI-3's first milestone was won by Tufa Labs' "Duck Harness"** — a
small open LLM writing Python in a live REPL. The winning entry is a harness, and is called one.

So the harness and the agent are the same object seen twice. ARES's design choice — *"the LLM
itself is the agent, not the scaffolding"* — is a claim about where that boundary sits, and where
you draw it determines what an interpretability result is even *about*.

Which yields the uncomfortable statement this proposal exists to act on:

> **Mechanistic interpretability studies the model. The measured evidence says the model
> contributes the minority of the capability. The field's entire toolkit is pointed at the
> smaller term.**

**Every method in the open stack assumes a single forward pass** — attribution graphs, SAEs,
lenses, probes, activation oracles. Deployment has moved to agents running for hours across
hundreds of decisions and the toolkit did not follow. That is the largest uncovered surface in the
landscape, and no rung of our own curriculum touches it either.

Stated honestly: **what does a control even look like for a claim about an agent's reasoning over
a hundred steps?** A matched-norm random direction is a clean null for a steering claim; there is
no accepted analogue when the behaviour is a *trajectory*. We cannot answer that yet and will not
promise a method we have not scoped.

What this grant buys is the position to attack it. Martian's
[ARES](https://github.com/withmartian/ares) — open source, actively developed, Terminal-Bench 2.0
and 36+ Harbor task packs — exists partly to support interpretability of sequential decision-making,
and its core choice, *"the LLM itself is the agent, not the scaffolding,"* is the
model-versus-harness boundary turned into an architecture. We already author scientific tasks in
that format ([m9h/terminal-bench-science](https://github.com/m9h/terminal-bench-science): NODDI
diffusion, transcranial ultrasound), and Martian solicits them. **A task pack costs us almost
nothing and puts us inside the infrastructure where the agentic question gets settled.**

**Program synthesis is the bridge between the two phases, and deserves renewed emphasis.** What
actually moved ARC was not models that emit answers but **models that emit programs, selected by
execution** — Greenblatt's ~8,000 candidates filtered by whether they reproduce the training pairs
(50%), BARC's induction head run at t=0.8 with execution filtering against its transduction head at
t=0 (ensembling for +13.75, because they solve *different* tasks), SOAR weighting execution
accuracy over vote count by 1000. The common element is **verifiable selection**: keep what
executes, not what is popular.

That matters twice over. A program is **an interpretable artifact by construction** — read it, run
it, test it, no reverse-engineering required. And execution filtering carries something no
interpretability method currently has: **a measured false-discovery rate.** Greenblatt found ~**9%**
of programs that pass the training pairs are still wrong, and voting removes about half. Ask what
the equivalent number is for an attribution graph or an SAE feature label and nobody can answer.
**Producing that number for interpretability methods is what item 5 is for.**

Phase 2 needs real compute and more than one person. This proposal does not ask for it. It asks
for the twelve months that make asking credible.

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
implementation, training documentation, and the control the original omitted. It continues an
existing SF cognitive-science reading group with a thirty-session, three-year record, organised on
the 1978 cognitive-science hexagon — because most of these claims import a construct from a
discipline that is not in the room. Five papers have been through the process; two produced a
refutation or a correction.

## Who does the work

An adjudication group is only as good as the people willing to run someone else's experiment and
publish a negative. Three channels, most targeted first:

- **★ The authors of the call, who are also a funder.** Oozeer, Quirke and Abdullah are at
  **[Martian](https://withmartian.com/prize)**, which runs a **$1M interpretability prize**
  awarding completed work, ships ARES, and runs hackathons with Apart Research. Their product is
  LLM routing, so their stake in interpretability being *auditable* is structural. We would arrive
  with a task pack, not a request.
- **Named unclaimed work, not open invitations.** Timaeus's board has a project with no lead;
  Decode Research says it is "always looking for new partners"; MIB will score a submission
  against a private test set. Specific things a specific person can own — what MedARC got right
  and what reading groups without projects get wrong.
- **[BuzzRobot](https://buzzrobot.substack.com/about)** — an OpenAI-alum community running talks
  with researchers from DeepMind, OpenAI, Meta and NVIDIA. The talk we would give is a failure
  report, not a pitch, because that self-selects for people who find checking interesting.

## On the NeuroAI framing, honestly

Foresight's reference is Mineault et al., *NeuroAI for AI Safety* (arXiv:2411.18526). Four of its
five paths are architectural bets that brain-likeness confers safety. **We are not making that
bet.** Only path 4 — interpretability using neuroscience methods — is load-bearing here, and the
import is *measurement discipline*, not brain structure: construct validity from psychometrics,
meta-d′ and type-2 ROC from signal detection theory, within-item controls from psychophysics,
adversarial collaboration from fields that built it after a replication crisis. **AI evaluation is
pre-crisis. That is the whole argument.**

## Qualifications

**A neuroscientist and computational psychiatrist who does this work on brains.** Mechanistic
inference from neuroimaging — the discipline that had this exact crisis, diagnosed it, and fixed
it. I know what the reform cost and which parts of it transferred, including the uncomfortable
parts: I follow the **COGITATE** adversarial collaboration, whose own preregistered prediction was
*not* confirmed in humans. An adjudication project should be willing to say that out loud about
its own source disciplines.

**A community builder in exactly this space, already doing it at scale.** I am Executive Director
and President of **Société BCI Montréal (NeuroTechX)**, a Quebec non-profit — so running a
distributed technical community with governance, contributors and continuity is my existing job,
not a skill this grant would fund me to acquire. I co-run a **San Francisco cognitive-science
reading group with a thirty-session, three-year record**, and participate in the **Active Inference
Institute's** textbook group. The reproduction group proposed here is a continuation of things that
already meet, not a new thing hoping to attract people.

**A scientific-software maintainer.** I package and maintain **ACT-R, Soar, Nengo, pyactr and
pyDDM** for Fedora, plus neuroimaging and cognitive-modelling stacks. Reproducibility here is not
an aspiration — keeping other people's research code running on other people's machines is
something I do continuously, and it is the same skill the tool gates require.

The combination is the point. The AI interpretability field is importing constructs from cognitive
science and statistics — *representation*, *mechanism*, *report*, *metacognition* — mostly without
anyone in the room who has had to defend one to a reviewer. **I have, and the fields I come from
already paid for the lesson this one is about to learn.**

---

*github.com/m9h/{jacobian-lens, spinning-up-in-mech-interp, tri-lens, controls-and-trajectories}
· huggingface.co/mhough. Every result, control and retraction reproducible from published open
artifacts.*
