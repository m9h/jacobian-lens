# The Consciousness-Indicator Scorecard: An Open, Controlled Benchmark for Machine-Consciousness Claims

**Longview Philanthropy — Digital Minds RFP, "Grants for Applied Work"**
**Applicant:** Morgan Hough, Orthogonal Research and Education Lab (OREL)
**Deadline:** 24 July 2026

---

## Summary

Claims that AI systems exhibit consciousness-adjacent properties are being published faster
than the field can evaluate them, and they concern systems whose internals no independent
party can access. There is at present no standard, controlled method for adjudicating such
claims. I propose to build one: a public benchmark that evaluates each claim against an
explicit null on open-weight models. Over the past three weeks I developed the constituent
methods and applied them to three published claims — refuting one, confirming a second, and,
in a third, locating a consciousness-indicator property that appears at an identifiable stage
of training. This proposal is to consolidate those methods into the standardized instrument
the field currently lacks.

## The problem

The rate of consciousness-adjacent claims about AI systems now exceeds the field's capacity to
check them, and the systems in question are ones whose internal states are inaccessible to
outside researchers.

In July 2026 Anthropic published *"Verbalizable Representations Form a Global Workspace in
Language Models,"* reporting that its model contains a representational subspace ("J-space")
functionally analogous to the global workspace of Baars and Dehaene. The public response
divided into two positions, neither of which executed the paper's released code: one held that
the method was a trivial variant of backpropagation (it is a distinct derivative); the other,
advanced by Erik Hoel, held that the method is unfalsifiable in principle and rested its single
empirical objection on a third-party demonstration the original author declined to endorse. The
exchange produced two rhetorical positions and no controls.

This is not the practice of a single laboratory. A separate group (Chicago, Google, and the
Santa Fe Institute) reported that steering one sparse-autoencoder feature produced an accuracy
gain, which they interpreted as evidence that reasoning models simulate interacting internal
voices. The underlying procedure is the same in both cases: write a direction into the residual
stream, observe an effect, and name it after a construct from cognitive science. Two independent
groups produced the same class of claim with the same missing control — a field-level
methodological gap that will grow more consequential as such claims approach questions of
welfare.

The obstacle is structural. No external party can obtain activation-level access to a frontier
model. When Anthropic invited outside commentary, the commentators — including Dehaene and
Naccache — received a draft rather than access, and had to request that Anthropic run the
experiments they proposed. For closed models, independent scrutiny is not merely
under-resourced; it is impossible. The only available mechanism is replication on open-weight
models, which is rarely done rigorously — and there is, at present, no venue in which to record
the result. Neuronpedia, the principal open platform, hosts interpretability artifacts (lenses,
sparse autoencoders) but attaches no null to them, provides no representation of a claim as
such, and offers no first-class status for a result that fails its control. The substrate for
evaluation exists; the evaluation layer does not.

## Work completed

Working alone over three weeks, on personal hardware supplemented by approximately $250 of
rented GPU time, I replicated the relevant methods on open-weight models and ran the controls
the original papers omit. The materials are public and reproducible. The significance is not
any individual result but the demonstration that an external evaluation capability now exists
and can perform both of its necessary functions: refuting unsupported claims and confirming
supported ones.

**Confirmation of a previously uncheckable claim.** Anthropic reports, qualitatively, that
post-training gives the J-space the assistant's "point of view" — a claim resting on a closed
model. AI2's fully open OLMo-3 model family makes it measurable. Validating my procedure against
Anthropic's own published lens (identity-distance error 0.4%) and controlling for capability, I
find that post-training moves the J-space by approximately 31% (for the instruction-tuned
variant) while task capability remains flat or declines — a substantial representational change
with no accompanying competence gain. The magnitude is determined by training method
(instruction tuning versus reinforcement learning) rather than task domain. This is the first
quantitative, controlled test of the claim.

**Refutation of an equally plausible claim.** The "society of thought" accuracy gain reverses on
the paper's own benchmarks: the same intervention at the same magnitude adds ten points on one
benchmark (Countdown) while removing twenty-two on another (MATH-Hard). The reported effect is
specific to the benchmark on which it was measured.

**Execution of the reviewers' proposed tests.** When Dehaene and Naccache proposed a battery of
tests drawn from human consciousness research and observed that Anthropic could run them, none
were run, because the model is closed. I implemented and ran all six on OLMo — ignition, trace
conditioning, inclusion/exclusion, local–global, dual-task, and metacognition — reporting the
outcomes in full: two yield the predicted signatures, three are inconclusive under first-pass
adaptations whose limitations I document. These tests are now available to any researcher.

**Localization of a consciousness-indicator property to a training stage.** The base model's
internal workspace covertly encodes whether its own answer is incorrect (discrimination
AUROC 0.69, exceeding what the model's output distribution reveals), while its explicit
self-assessment performs at chance. Examining each post-training stage in turn, explicit
self-assessment rises from chance (0.51) to 0.71 at supervised fine-tuning — the first stage —
in both post-training families, and improves only marginally thereafter. The base model already
possesses the internal signal; supervised fine-tuning renders it reportable. To my knowledge
this is the first developmental localization of a consciousness-indicator property to a specific
training stage on open-weight models.

**Self-correction.** On two occasions I reported a result — a sharp emergence threshold, and a
subsequent reinterpretation of it — that a more robust version of the same experiment
overturned; both retractions are published alongside the results. I also identified and
corrected an error in the shared open tooling: the validation procedure others would use to
check a lens against a reference computed the wrong quantity and would have accepted incorrect
fits.

## Proposed work: the Consciousness-Indicator Scorecard

These results are the initial entries of the instrument I propose to build: a standardized,
controlled benchmark of consciousness-indicator tests for open-weight models, in which every
test is accompanied by an explicit null, property emergence is tracked across training, and
negative results are recorded on equal terms.

The benchmark presents a matrix of open-weight models against consciousness indicators (global
workspace, ignition, metacognition, and others). Each cell reports a score relative to its null
and links to the controlled test, a single-command reproduction requiring no access to closed
models, and the property's trajectory across the model's training. A companion registry records
each published claim together with its adjudication status — reproduced, refuted, or
inconclusive under control — and the artifact required to reproduce it. The
supervised-fine-tuning emergence result described above constitutes a completed entry.

The design follows ARC-AGI: a result registers only when it exceeds its control, so the
benchmark does not reward unsupported claims; it provides a common reference for evaluating any
model; and every entry is independently reproducible. It is complementary to Neuronpedia rather
than duplicative — it consumes Neuronpedia's hosted lenses, contributes the controls, the
convergence-fitting procedure, and the tooling corrections back to it, and adds the evaluation
layer above. Neuronpedia enables inspection of a model's internals; the Scorecard evaluates
claims made about them.

## Relevance to digital minds

The RFP seeks empirical work that advances the field without depending on the resolution of
intractable philosophical questions. The Scorecard reframes the question of moral status as a
developmental one. Whether a system is conscious is not tractable; when, and by what process, a
system acquires a property that would bear on its moral status is measurable, and I have
measured an instance. Self-monitoring is a recognized consciousness indicator; I can now state
that it is present covertly in the base model and becomes reportable at supervised fine-tuning,
on models any researcher can examine. The result replaces a metaphor with a measured quantity
accompanied by a control.

The connection to safety is direct. The same instrument that grounds these claims also revealed
a model's representation of the fact that it was being evaluated, and ablating that
representation increased dishonest behavior. Welfare assessment and safety evaluation rely on a
single shared instrument; if that instrument's quality metric is sensitive to noise — which I
have shown it to be — both inherit the defect.

## Deliverables and scope (twelve months; one researcher with OREL)

The individual experiments are inexpensive, which is precisely why they cannot themselves
constitute the deliverable, and why unexamined results in this field are so often wrong. The
work of the grant is to assemble them into a maintained, validated, and adopted benchmark.
Specifically: (1) hardening the measurement tooling and contributing it to Neuronpedia; (2) a
seed Scorecard covering three indicators and five open-weight models across scale and
architecture, with the OLMo checkpoint sequence providing the first emergence entries; (3) an
executable, controlled implementation of the Butlin–Long indicator framework, currently applied
only in prose; and (4) tutorials in which the reader runs each control — the principal one
requiring no GPU, since the lenses are published. The principal risk the grant addresses is
adoption and test quality: a poorly constructed indicator test is worse than none, which is why
the controls-first, self-correcting methodology already demonstrated is essential to the
benchmark's credibility.

## Qualifications and timing

The work draws on an unusual combination of backgrounds: systems engineering, computational
neuroscience (neuroimaging pipelines and cognitive-architecture software), scientific-software
packaging, and mechanistic interpretability. I work with the Orthogonal Research and Education
Lab (OREL), an independent research organization, and maintain cognitive-modeling packages for
Fedora (ACT-R, Soar, Nengo). The tutorials draw directly on the neuroscience from which the AI
claims are borrowed, including the COGITATE adversarial collaboration, whose own preregistered
prediction of ignition was not confirmed in human subjects — a result that materially
constrains the weight the AI analogy can bear. The timing is consequential: AI2's OLMo-3,
released in the past weeks, is the first fully open model family to provide a base model, its
post-trained variants, the training data, and intermediate checkpoints together; the substrate
this program requires has only just become available, at a moment when the claims are becoming
more ambitious.

## Budget and counterfactual

Requested: **[FIGURE]** over twelve months, covering researcher time, cloud GPU, and
dissemination. Compute is the smallest component by a wide margin: approximately $250 produced
all of the results described. The binding constraint is dedicated researcher time rather than
equipment. Without funding, the work continues part-time and more slowly, and the tooling
remains a personal repository rather than a maintained instrument on which the field can rely.

---

*Repositories: github.com/m9h/jacobian-lens and m9h/societies-of-thought; lenses and results at
huggingface.co/mhough/olmo3-jacobian-lenses. Every result, every control, and the retractions
are reproducible from published open artifacts, without access to closed models.*
