# The Referee the Field Doesn't Have: External Controls for Machine-Consciousness Claims

**Longview Philanthropy — Digital Minds RFP, "Grants for Applied Work"**
**Applicant:** Morgan Hough (independent) · **Deadline:** 24 July 2026

---

## The one-sentence version

The field is making consciousness-adjacent claims about AI systems no outsider can
inspect, there is no external mechanism to check them, and in the last three weeks I built
one and used it to **kill one such claim and confirm another** — both on fully open models,
both reproducible from a laptop.

## The gap

Machine-consciousness claims are arriving faster than the means to check them, and they are
being made about systems whose internals no independent party can access.

In July 2026 Anthropic published *"Verbalizable Representations Form a Global Workspace in
Language Models"* — that Claude contains a "J-space" functionally analogous to the global
workspace of Baars and Dehaene. 1.2 million views in a day. The critical response split into
two camps, **neither of which ran the paper's code**: one called the method "just backprop"
(it is a different derivative); Erik Hoel called it unfalsifiable by construction and rested
his one empirical objection on a third-party demo the author himself declined to stand
behind, writing that a real replication "would be important to see."

Two rhetorical positions and no controls. That is the actual state of the art for
adjudicating a consciousness-indicator claim about a frontier model.

And it is not one lab's habit. In January 2026 a separate group — Chicago, Google, the Santa
Fe Institute, no authors in common with Anthropic — published *"Reasoning Models Generate
Societies of Thought,"* steering one sparse-autoencoder feature, observing an accuracy gain,
and reading it as evidence that reasoning models simulate interacting internal voices. The
same move: write a direction into the residual stream, observe an effect, name it after a
construct from cognitive science. **Two independent groups, two borrowed constructs, the same
missing control.** That is a field-level gap, not an isolated error — and it will only get
more consequential as the claims get closer to welfare-bearing.

The obstacle is structural, and it is the reason the gap persists: **no external party can
obtain activation-level access to a frontier model.** When Anthropic solicited outside
commentary, the commentators got a draft, not access — Dehaene and Naccache had to *ask
Anthropic to run* the experiments they proposed. Independent scrutiny of these claims is not
under-performed; for closed models it is impossible. The only mechanism that exists is
replication on open weights, and almost no one is doing it rigorously.

## I built the mechanism — and it works in both directions

Over three weeks, alone, on personal hardware plus roughly **$250 of rented GPU**, I
replicated the method on open weights and ran the controls the papers do not. The results are
public and rerunnable (`github.com/m9h/jacobian-lens`, `m9h/societies-of-thought`). The point
is not any single result — it is that a working external referee now exists, and it can do
the two things a referee must: **convict, and exonerate.**

**It confirmed a claim no outsider could previously check.** Anthropic's Claim 6 — that
post-training shaped the J-space "toward a point of view rather than pure prediction" — rests
on Sonnet 4.5, whose activations no external party can touch; the paper's own commentators
could not test it. AI2's fully-open OLMo-3 ladder (base → SFT → DPO → RLVR, Apache-2.0, with
training data) makes it checkable. Anchor-gated against Anthropic's own published lens
(identity-distance error **0.4%**) and capability-controlled, I find: post-training moves the
J-space **~31%** while capability (MMLU) stays **flat-to-down** — a large representational
shift with *no* competence gain, which is precisely a change of viewpoint rather than of
prediction. The magnitude is set by training *method* (instruction/CoT tuning ~5× RLVR), not
task *domain* (RL-Zero domains differ ~1% at matched capability). **Claim 6 supported, and
sharpened — the first external confirmation of a frontier consciousness-adjacent claim, on
artifacts anyone can rerun.**

**It killed a claim that looked just as convincing.** The "society of thought" accuracy gain
— steering one conversational feature — reverses on the paper's own benchmarks: the same
feature at the same dose adds +10 points on Countdown but costs **−22 on MATH-Hard** while
*increasing* dialogic markers. It is an artifact of the one benchmark it was measured on. A
second, independent method (Jacobian-lens geometry) reaches the same verdict as the steering
result: the "viewpoint" layer of a model is **decoupled from its capability**. Two unrelated
instruments, one structural conclusion.

**And it refuted itself, twice, in public.** I reported a sharp emergence threshold — a
concept never named in the prompt surfacing suddenly at 27B. Then a correction: it tracked
architecture, not scale. Both were wrong. The robust 102-prompt version produced a smooth
rise across seven models with a *dense* 32B beating the hybrid 27B; the original was a single
lucky prompt. The retraction is published beside the result.

That last item is the one I would ask you to weigh most heavily, because it is the whole
argument. **A referee is only worth having if it will kill its own headline** — and this one
has, twice. The discipline this field lacks is not cleverness; it is the willingness to run
the control that destroys your own result and say so. That willingness is exactly what makes
the one positive result — Claim 6 — believable. I earned the right to confirm a claim by
demonstrating I will kill one, including mine.

Three supporting controls round out the instrument, each already run: a distance-only null
that reproduces **79–91%** of the paper's headline block-structure figure (both sides
overclaimed — Anthropic on the sharpness, the critics on the absence); a demonstration that
the paper's own quality metric rewards noise (on Anthropic's published 27B lens, a plain
logit lens scores *better* while emitting punctuation); and a randomization control that
*vindicates* the method against its most-shared criticism (randomize the trained blocks and
the lens reads out nothing — 0.0003 vs 0.3414). I also found and fixed a real bug in the
shared open tooling: the gate everyone would use to validate a lens against a reference was
computing the wrong quantity and would have silently failed correct fits.

## Why this is a digital-minds result, not just a methods result

Your RFP asks for empirical work that sidesteps intractable philosophical questions. This is
that, in the most literal form — it does not ask whether a model is conscious; it asks
whether the *measurements* people use to argue about it survive a control.

But the Claim 6 result is more than methodological hygiene, and it is worth being direct
about why. A **stable point of view, installed by training, that is decoupled from task
competence** is not a capability — it is closer to the kind of thing welfare questions are
actually about. "The model got better at reasoning" is a competence claim; "the model
represents from a persistent standpoint that its competence does not explain" is not. That
distinction is exactly where consciousness-indicator arguments live, and I have now shown it
is **measurable on open models, method-driven, and separable from capability** — the first
time that structure has been demonstrated by anyone outside a frontier lab. Whatever one
concludes about its significance, the field can now argue about a number with a control
attached instead of a metaphor.

The safety link is direct, not decorative. The same technique that grounds the workspace
claim also surfaced Claude's recognition that it was being evaluated — and Anthropic reports
that ablating those evaluation-awareness representations *increased* dishonest behaviour.
Welfare assessment and safety evaluation are drawing on **one shared instrument.** If that
instrument's metric rewards noise — and I have shown it does — both inherit the error. The
referee is not a luxury; it is load-bearing for two of the questions you fund.

## What I propose to build

**A standing, open, adversarially-controlled audit layer for consciousness-indicator claims
in AI systems** — the external-referee function the field structurally lacks — over twelve
months.

**(a) Harden and release the tooling, and push it upstream where it has leverage.** I have a
working companion library (`jlens-lab`): convergence-based lens fitting, architecture layouts
for state-space and linear-attention hybrids the reference implementation cannot load, and
the controls above — each corresponding to a failure that silently produces plausible output.
I package scientific software for Fedora and can ship this through standard channels. More
consequentially, I will contribute it **upstream to Neuronpedia**, the single MIT-licensed
platform on which external scrutiny of this literature actually rests (hosting Anthropic's
lenses beside DeepMind's and OpenMOSS's SAEs, maintained by one person). Three offerings: the
convergence-fitting wrapper missing from the public chain, the fix for the defective artifact
I found auditing all 38 published lenses, and the controls — so that **a null ships with every
measurement by default.** Strengthening shared infrastructure beats another repository nobody
finds.

**(b) Turn the field's reference instrument into runnable tests.** Butlin, Long et al. (2023)
converted Global Workspace Theory into four checkable indicator properties — parallel
modules, a capacity-limited bottleneck, global broadcast, state-dependent attention. It is
the field's standard scorecard and it is currently applied *by hand, in prose.* I will make
it executable against open-weight models with an explicit null for each property, so that
"this system satisfies GWT-2" becomes a number with a control rather than a judgement. (I have
already begun: the paper's capacity-limitation claim gives checkable figures — "never more
than 10% of variance," "tens of concepts" — that a forward pass can test on published lenses,
and the broadcast property the paper itself concedes does not hold in the recurrent sense.)

**(c) Publish a public scorecard across open models and architectures** — scale (0.6B–70B)
and mechanism (dense, linear-attention hybrid, Mamba-2/SSM) — every claim against its null,
negatives on equal footing. I have proven this is now feasible across model *families*: the
OLMo-3 result already has a second-family replication wired (Mistral's Ministral-3 ladder),
so cross-family robustness is a scorecard column, not an aspiration.

**(d) Teach it so anyone can referee.** Hands-on tutorials in which the reader runs each
control. The most consequential — the distance-only null that dissolves most of a blockbuster
figure — needs **no GPU at all**, because the lenses are published. A graduate student can
check the field's most-shared consciousness claim on a laptop. That is true today and almost
nobody knows it.

## Why me, and why now

The work sits at an unusual intersection — systems engineering, computational neuroscience
(neuroimaging pipelines, cognitive-architecture software), scientific-software packaging, and
mechanistic interpretability. I maintain neuroscience and cognitive-modelling packages for
Fedora (ACT-R, Soar, Nengo, drift-diffusion tooling). The tutorials draw on the neuroscience
the AI claims are borrowed from — including the COGITATE adversarial collaboration, whose own
preregistered ignition prediction *failed* in human brains, which materially changes how much
weight the AI analogy can bear. I am not affiliated with a laboratory. Everything above was
produced without funding.

The timing is not incidental. AI2's OLMo-3 — released weeks ago — is the first fully-open
base→post-trained ladder with training data, and it is what made the Claim 6 test possible at
all; the open substrate for this entire program has only just arrived. The claims are getting
bolder and closer to welfare-bearing. The window to establish an external-audit norm is now,
before the next paper, not after.

## Budget and counterfactual

**Requested: [FIGURE] over 12 months**, covering researcher time, cloud GPU, and
dissemination. Local hardware (a workstation GPU and a DGX Spark) covers development; rented
A100/H100/B200 time covers the sweeps. Compute is the smallest line by a wide margin — roughly
**$250 has produced everything above.** The bottleneck is dedicated time, not equipment: fund
the referee, not the rig.

Without funding I continue this part-time and slower, and the tooling stays a personal
repository rather than a maintained public instrument the field can rely on.

---

*Repositories: `github.com/m9h/jacobian-lens` and `m9h/societies-of-thought` — every result,
every control, and the retractions, reproducible from published open artifacts. No frontier
access required, by design.*
