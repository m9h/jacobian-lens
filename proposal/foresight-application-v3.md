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
| **Adjudication as an academic agenda** | **[TSG Lab](https://tsglab.github.io/)** (Barez & Trager, Oxford) — the auditability call, an automated-auditing research agenda, *Chain-of-Thought Is Not Explainability*, contamination and benchmark critique | a lab: two PIs, ~12 people, papers |
| **Benchmark rigour** | **[ABC](https://github.com/uiuc-kang-lab/agentic-benchmarks)** — Zhu, Jin & Kang (UIUC) with Stanford, Berkeley, MIT, Transluce, ML Commons and **UK AISI** | a released checklist, and an audit that should alarm people |
| **★ Adjudication as a standing community function** | **nobody** | **—** |

**The distinction in those last two rows is the honest version of this proposal's claim, and it took
me three passes to get right.** Each time I mapped the landscape, a group turned out to be doing
part of what I thought was unoccupied — Transluce ran the neuron baseline the SAE literature
skipped; Timaeus has an open project board; and **TSG Lab is the closest thing to this agenda that
exists**, with a published research agenda for automated interpretability-driven auditing.

What remains genuinely unoccupied is narrower and more defensible:

- **A standing group whose output is other people's results, rechecked** — not papers advancing an
  agenda, but reproductions, negative findings and controls on claims *someone else nominated*.
  That is mechanism (1) of the ACL call, and it is the one nobody has built, including its authors.
- **Process validity, as distinct from task and outcome validity.** The **Agentic Benchmark
  Checklist** ([Zhu, Jin & Kang, UIUC, + 8 institutions](https://github.com/uiuc-kang-lab/agentic-benchmarks))
  is the state of the art here and its audit is alarming: a trivial agent returning **empty
  responses** passes **38%** of τ-bench airline tasks; SWE-Lancer agents can overwrite the test
  files and **score 100% without solving anything**; and **24% of SWE-bench-Verified's top-50
  leaderboard positions are incorrect**. Applied to CVE-Bench it removed **33% absolute**
  overestimation. But ABC is a checklist for whether a *score* means what it says — task validity
  (does success equal the capability?) and outcome validity (does the result equal success?). It
  contains a single subcheck gesturing at reasoning — *"correlate metrics with actual reasoning"* —
  and no method for it. **Whether a claim about an agent's reasoning is valid remains unaddressed**,
  which is precisely where the Diplomacy ablation and protocol analysis both point.
- **The teaching layer.** A curriculum where every technique ships with its null, so the next
  cohort arrives already able to check.

**TSG is therefore the collaborator, not the competitor** — and they explicitly invite sponsored
research directions. A lab producing automated auditing methods and a community running human
reproduction are complements; neither substitutes for the other.

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
no accepted analogue when the behaviour is a *trajectory*.

**But the discipline that owns this question is identifiable, and it is mine.** The gap is already
occupied by psychologists rather than by circuits people: *AI Agent Behavioral Science*
([Nature HSSC, 2026](https://www.nature.com/articles/s41599-026-07316-7)) and *Machine
Psychology* — the latter describing itself as "moving beyond architectural interpretability by
treating models as experimental subjects." And the classical method for long-horizon reasoning is
**Ericsson & Simon's protocol analysis**, *Verbal Reports as Data*, which agent tooling is
currently reinventing — [AgentTrace](https://arxiv.org/html/2602.10133v1) bills itself as "the
first open standard for structured agent logging," spanning cognitive, operational and contextual
traces — **without the validity conditions that took thirty years to establish.**

The sharpest statement of the problem I can offer:

> **Chain-of-thought is a concurrent verbal protocol, and "unfaithful chain-of-thought" is
> Nisbett & Wilson (1977) rediscovered** — *Telling More Than We Can Know*, in which people
> confidently report reasons that had no causal role in their behaviour.

Protocol analysis distinguishes concurrent from retrospective report, quantifies **reactivity**
(verbalizing alters the process being reported), and specifies when a report *tracks* a process
rather than *narrating* one. That is the missing validity theory for every interpretability claim
built on an agent's stated reasoning, it is a solved problem in another field, and think-aloud
protocol analysis is standard equipment in the clinical decision-making research I come from.
**And there is a worked case the interpretability field has not noticed.** CICERO reached
human-level play at *Diplomacy*, a game whose entire premise is negotiation.
[*More Victories, Less Cooperation*](https://aclanthology.org/2024.acl-long.672/) (ACL 2024, UMD /
Princeton / Sydney / USC ISI) then assessed the communication itself — 24 games, 200 human-player
hours, ~27,000 messages, Abstract Meaning Representation used to map stated intentions onto actual
moves, plus human annotation of perceived deception (318 messages annotated as lies, 1,167
perceived as lies). Their finding, in their words: Cicero's communication is **"more transactional,
relying on its optimal strategy rather than the alliance building which is the hallmark of top
human players,"** humans can reliably identify it, and it is *less* deceptive and persuasive than
they are. They state the attribution problem explicitly: **"it is unclear if Cicero's success is
due to its use of natural language or its strategic model."**

That is a verbal-report validity study with something single-prompt chain-of-thought work does not
have — an **independent behavioural record** (the moves) to check the talk against. Of the 13
papers citing it, **none** mention faithfulness, chain-of-thought, verbal reports or
interpretability; the two literatures are running in parallel. Connecting them is a cheap, concrete
deliverable, and one of the ACL auditability authors also co-wrote *Chain-of-Thought Is Not
Explainability*, so it contributes to work he is already doing.

⚠️ **A correction we are keeping visible.** An earlier draft described this as an *ablation* — that
communication was switched off and the score barely moved. **The paper contains no such
experiment.** That reading came from a university press page, and it was ours to check. Reading the
paper also produced a second lesson: an automated summariser of the same PDF asserted the *opposite*
(that removing communication sharply reduced the win rate), which is equally absent. **Two
secondary sources, two different wrong answers, one primary text.** This is recorded in
[CITATION_AUDIT.md](CITATION_AUDIT.md) rather than quietly fixed, because it is the failure mode
this proposal exists to address, committed by us, mid-drafting.

**This is the same argument as the dead salmon, arriving by a second route.** I am not promising
a finished method for agentic interpretability — but I can name which discipline has the controls,
which experiment already demonstrates the problem, and why nobody is importing either.

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

**A concrete first target already exists.** COS-PLAY ([arXiv 2604.20987](https://arxiv.org/abs/2604.20987),
UMD / USC / Good Start Labs / MBZUAI, **MIT-licensed code**) reports that an **8B base model** with
a co-evolving skill bank reaches **924.4** average reward across four single-player games against
GPT-5.4's **717.4** — the base Qwen3-8B alone scores 379.6. A harness moving an 8B from bottom to
top of a frontier leaderboard is the thesis in one table.

It is also exactly the kind of claim that should be adjudicated rather than repeated, and its own
ablation table says why: **every partial configuration lands at or below the base model** (SFT +
final skill 359.5, GRPO + 1st skill 305.2, base 379.6) while the full system doubles the best of
them. An interaction effect with no main effects is either real synergy or something the ablations
do not isolate. At 16 rollouts per condition with public code, it is cheap to check — and it is a
better first entry for the register (item 2) than anything we would invent.

The honest other half, which the write-ups lead less with: on **multi-player social reasoning the
same harness loses** — Avalon win rate 39.0 vs GPT-5.4's 65.0, Diplomacy 2.96 supply centres vs
4.70. So harness gains are large *and domain-bounded*: they transfer where skills are reusable
procedures and fail where the task is social and adversarial. **That boundary is a finding, and
nobody is measuring where it sits.**

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

*Source-based audit of this document's own claims — which rest on a full read, which on an
abstract, which on a search summary — is in [CITATION_AUDIT.md](CITATION_AUDIT.md). Several
claims are currently at the weakest tier and are marked not-for-submission until sourced. That
file is mechanism (3) of the ACL call applied to this proposal, and it exists because a document
arguing the field does not check its sources should say how well it checked its own.*

*github.com/m9h/{jacobian-lens, spinning-up-in-mech-interp, tri-lens, controls-and-trajectories}
· huggingface.co/mhough. Every result, control and retraction reproducible from published open
artifacts.*
