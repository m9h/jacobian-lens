# An Open, Controlled Benchmark for Consciousness-Indicator and Metacognitive Properties in AI

**Longview Philanthropy — Digital Minds RFP, "Grants for Applied Work"**
**Applicant:** Morgan Hough, Orthogonal Research and Education Lab (OREL) — Deadline: 24 July 2026

---

## Summary

Consciousness-adjacent and metacognitive claims about AI systems are being published faster
than they can be evaluated, on systems whose internals no independent party can access, and
there is no standard, controlled method for adjudicating them. I propose to build one: an open
benchmark that scores each claim against an explicit null on open-weight models. It spans two
property classes that share a single instrument — the consciousness indicators relevant to
moral status, and the metacognitive properties (self-monitoring, calibration, reportable
uncertainty) central to reliability and safety. Over three weeks I developed the methods and
applied them to several published claims, refuting one, confirming another, and locating a
property that is both a consciousness indicator and a metacognitive capacity emerging at an
identifiable stage of training.

## The problem

The rate of such claims now exceeds the field's capacity to check them, and the systems
concerned are inaccessible to outside researchers. In July 2026 Anthropic reported that its
model contains a subspace ("J-space") analogous to the global workspace of Baars and Dehaene.
The public response divided into two positions, neither of which ran the released code; the
exchange produced two rhetorical positions and no controls. This is not one laboratory's
practice: a separate group reported that steering one sparse-autoencoder feature raised
accuracy, and read it as evidence of interacting internal voices. The procedure is the same in
both cases — write a direction into the residual stream, observe an effect, and name it after a
construct from cognitive science — and in both the confirming control is absent.

The obstacle is structural: no external party can obtain activation-level access to a frontier
model. When Anthropic invited outside commentary, the reviewers received a draft rather than
access and had to request that Anthropic run the experiments they proposed. For closed models,
independent scrutiny is not under-resourced but impossible; the only mechanism is replication on
open-weight models, which is seldom done rigorously — and there is no venue in which to record
the result. Neuronpedia, the principal open platform, hosts interpretability artifacts but
attaches no null to them, has no representation of a claim, and gives no first-class status to a
result that fails its control. The substrate exists; the evaluation layer does not.

## Work completed

Working alone over three weeks, on personal hardware and roughly $250 of rented GPU, I
replicated the relevant methods on open-weight models and ran the omitted controls. The
materials are public and reproducible; the significance is that an external evaluation
capability now exists and can both refute unsupported claims and confirm supported ones.

**A previously uncheckable claim, confirmed.** Anthropic reports qualitatively that
post-training gives the J-space the assistant's "point of view," on a closed model. AI2's fully
open OLMo-3 family makes it measurable. Validating my procedure against Anthropic's published
lens (identity-distance error 0.4%) and controlling for capability, I find post-training moves
the J-space by ~31% while task capability is flat or declining — a large representational change
with no competence gain, set by training method rather than task domain. This is the first
quantitative, controlled test of the claim.

**An equally plausible claim, refuted.** The "society of thought" accuracy gain reverses on the
paper's own benchmarks: the same intervention adds ten points on one (Countdown) and removes
twenty-two on another (MATH-Hard). The effect is specific to the benchmark on which it was
measured.

**The reviewers' proposed tests, executed.** Dehaene and Naccache proposed a battery of tests
from human consciousness research and noted Anthropic could run them; none were, because the
model is closed. I implemented and ran all six on OLMo — ignition, trace conditioning,
inclusion/exclusion, local–global, dual-task, and metacognition — reporting outcomes in full:
two yield the predicted signatures, three are inconclusive under first-pass adaptations whose
limitations I document.

**A property localized to a training stage.** The base model's internal workspace covertly
encodes whether its own answer is incorrect (discrimination AUROC 0.69, exceeding what its
output distribution reveals), while its explicit self-assessment is at chance. Across
post-training stages, explicit self-assessment rises from chance (0.51) to 0.71 at supervised
fine-tuning — the first stage — in both post-training families, improving only marginally
thereafter. The base model already holds the internal signal; supervised fine-tuning renders it
reportable. To my knowledge this is the first developmental localization of such a property to a
training stage on open-weight models. It also answers two open questions in the metacognition
literature: Liu et al. (2026), surveying metacognition in LLMs, note that no prior work
establishes whether an internal signal predicts correctness *independently of output
probabilities*, or *when* such capacities emerge in training. This result does both.

I also reported, and then retracted, two results that a more robust version of the same
experiment overturned — both retractions are published — and corrected an error in the shared
tooling whose validation step computed the wrong quantity and would have accepted incorrect
fits.

## Proposed work: the Scorecard

These are the initial entries of the instrument I propose to build: a standardized, controlled
benchmark for open-weight models in which every test carries an explicit null, property
emergence is tracked across training, and negative results are recorded on equal terms. It
presents a matrix of models against properties — consciousness indicators (global workspace,
ignition) and metacognitive properties (self-monitoring, calibration, reportable uncertainty)
alike. Each cell reports a score relative to its null and links to the controlled test, a
single-command reproduction requiring no closed-model access, and the property's trajectory
across training. A companion registry records each published claim with its adjudication status
— reproduced, refuted, or inconclusive under control.

The design follows ARC-AGI: a result registers only when it exceeds its control. It is
complementary to Neuronpedia — consuming its lenses, contributing the controls, the
convergence-fitting procedure, and the tooling corrections upstream, and adding the evaluation
layer above. Neuronpedia enables inspection of a model's internals; the Scorecard evaluates
claims made about them.

## Relevance: two lines of impact from one instrument

**Digital minds.** The RFP seeks empirical work that does not depend on resolving intractable
philosophy. The Scorecard reframes moral status as *developmental*: whether a system is
conscious is not tractable, but when and by what process it acquires a property that would bear
on its moral status is measurable — and I have measured an instance, on models any researcher
can examine.

**Reliability and safety.** The same measurements bear directly on trustworthy AI. Metacognition
— a model's calibrated sense of what it knows and whether it is correct — is a determinant of
reliability that the survey literature treats as a cornerstone of transparent AI; a model whose
internal error signal is not reportable is one whose stated confidence cannot be trusted. The
same instrument also revealed a model's representation that it was under evaluation, whose
ablation increased dishonest behavior — an evaluation-integrity result. The controls, the open
substrate, and the emergence-across-training method apply identically to both classes, so the
benchmark advances moral-status research and AI-reliability research at once, and does not stand
or fall on the consciousness framing alone.

## Deliverables and scope (twelve months; one researcher with OREL)

The experiments are inexpensive, which is why they cannot be the deliverable and why unexamined
results in this field are so often wrong; the work of the grant is to assemble them into a
maintained, validated, adopted benchmark. Specifically: (1) harden the measurement tooling and
contribute it to Neuronpedia; (2) a seed Scorecard of three indicators and three metacognitive
properties across five open-weight models and the OLMo checkpoint sequence, providing the first
emergence entries; (3) an executable, controlled implementation of the Butlin–Long indicator
framework, currently applied only in prose; and (4) tutorials in which the reader runs each
control — the principal one requiring no GPU. The chief risk is adoption and test quality: a
poorly constructed test is worse than none, which is why the controls-first, self-correcting
methodology already demonstrated is essential.

## Qualifications, timing, and budget

The work draws on systems engineering, computational neuroscience, scientific-software packaging
(I maintain cognitive-modeling packages for Fedora — ACT-R, Soar, Nengo), and mechanistic
interpretability, at OREL. It engages the neuroscience the AI claims are borrowed from,
including the COGITATE adversarial collaboration, whose own preregistered ignition prediction
was not confirmed in humans — which constrains the weight the analogy can bear. The timing is
consequential: AI2's OLMo-3, released weeks ago, is the first fully open family to ship a base
model, its post-trained variants, the training data, and intermediate checkpoints together.

Requested: **[FIGURE]** over twelve months for researcher time, cloud GPU, and dissemination.
Compute is the smallest line — approximately $250 produced the results above; the binding
constraint is dedicated time. Without funding, the work continues part-time and the tooling
remains a personal repository rather than a maintained public instrument.

---

*Repositories: github.com/m9h/jacobian-lens and m9h/societies-of-thought; lenses and results at
huggingface.co/mhough/olmo3-jacobian-lenses. Every result, control, and retraction is
reproducible from published open artifacts, without access to closed models.*
