# Literature review and roadmap — 6 Oct 2026

**Status: partial.** Interrupted by a context compaction; §1 covers what was verified, §2 lists what
was *not* searched. Evidence tier marked per item (S1 full text · S2 abstract/page · S3 search
summary — unverified).

## 1. What changed in the literature

### ★ Our looped-architectures note was pre-empted — and the lesson is the one we keep relearning
**Wang & Reid, *Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive
Recurrence?*** ([arXiv 2609.01924](https://arxiv.org/abs/2609.01924), Fin AI Research, **1 Sep 2026**).
S1 — read. I wrote [`looped_architectures.md`](looped_architectures.md) on **7 Sep** without finding
it. Six days. The note's own §5 says *"apply the usual rule before building"* and I did not apply it
to §2–4. Filed against [`feedback-literature-review-before-design`](../../.claude) — again.

What they did, which is more than the note proposed: a **virtual-unrolling adapter** that gives the
J-lens `(layer, loop)` addressing (the exact gap §2 of the note said tooling lacked), then the full
workspace suite — lens fitting, readout, **eleven causal experiment families** — on **Ouro-2.6B**
(48 layers × 4, deeply supervised) and **Huginn-0125** (4-layer core × 16, latent reasoning), against
Qwen3.6-27B.

What they found:
- **A workspace forms in weight-tied models.** *"For the weight-tied special case… our functional
  answer is yes."* Recurrence changes the **interfaces**, not the existence: *"every route by which
  content enters, survives, or leaves the workspace follows the iteration structure rather than
  absolute depth."* — my §2 claim, now empirical.
- **Ouro is near-Markovian loop to loop**: deep supervision decodes each loop into a token-aligned
  state, destroying workspace content in transit; writes must span every remaining loop (38% vs 21%).
- **Huginn carries content across all 16 recurrences** but reads/writes/ablations act in a
  **~2-recurrence window**; the last window alone is as effective as all of them.
- ★ **Weight tying collapses the fitting cost**: transports repeat across iterations (**cos 0.9991**
  on Huginn), so one iteration's fit covers full depth. This *answers* my §3 SAE-pooling worry: on
  Huginn the per-loop distributions are near-identical.
- Verbalising **injected** content tracks explicit per-iteration supervision (Ouro 31%, Huginn 0/97,
  untied 27B scale does not substitute); steering **existing** content does not (Huginn 98%).
- The recurrence caveat I raised is **resolved against the strong reading**: persistence is
  *"continuous recomputation from a re-injected input encoding, not a memory buffer; content does
  not survive even one recurrence beyond its last refresh."*
- **Nanbeige is not tested** (0 mentions). Their stated limitation: *"one pre-trained checkpoint per
  family, so architecture is confounded with training recipe and scale. Following checkpoints
  throughout pretraining would test whether our conclusions are stable over training and reveal
  when J-space emerges."* **That is our developmental axis, named as the open problem.**

### The CoT-trace thread firmed up
**Bhambri, Biswas & Kambhampati, *Interpretable Traces, Unexpected Outcomes*** — **ACL 2026 long**
([2026.acl-long.1686](https://aclanthology.org/2026.acl-long.1686/)). S2. The QA replication of the
swapped-trace result on *pretrained* Llama/Qwen is peer-reviewed, not a preprint. Upgrades the
evidence tier assigned on 1 Sep and weakens my "it's just mazes" objection further. Still unread
in full.

### Adjudication-adjacent, unread (S3 — titles from search only)
- *The Story is Not the Science: Execution-Grounded Evaluation of Mechanistic Interpretability
  Research* ([2602.18458](https://arxiv.org/abs/2602.18458)) — "execution failures and weak
  reproducibility remain common." Directly on the proposal's thesis.
- *When Circuits Are Too Broad: Unit Tests for Mechanistic Interpretability* (ICML 2026) — unit tests
  for MI; adjacent to our tool gates.
- Martian's $1M prize round **closed 1 Feb 2026**; no new round found.

### ⚠️ One claim I could not source
A search summary stated *"the language-selectivity gate collapses to always-on on an SFT model."*
That bears on our post-training ladder. **Provenance unknown** — not in Wang & Reid by grep. Treat as
S3 and do not cite.

## 2. Not searched before compaction
Agentic-interpretability methods (Sep–Oct) · probe-validity / answer-readout follow-ups ·
**Foresight deadline** · Ouro/Huginn/Nanbeige-4.5 availability as substrates · source of the SFT-gate
claim. Resume from here.

## 3. Roadmap, ranked by leverage per unit effort

| # | item | why now | cost |
|---|---|---|---|
| 1 | **Reframe `looped_architectures.md` around Wang & Reid**; their open problem (checkpoints through pretraining × looped arch) is exactly our OLMo developmental axis — *if* a looped family publishes checkpoints | our niche, named by someone else | 1 hr + a substrate check |
| 2 | **SF-OSMI**: offer the crit methodology; run one session with the 29-s salmon opener | their stated weakness is our only real asset; fixes our zero-uptake problem | your call on the relationship |
| 3 | **Submit Foresight** — 1,042-word draft is ready | deadline unknown; **ask them** | 1 email |
| 4 | **RoPE refit, base lens only** (~3.8 GPU-h, Modal), matching Neuronpedia's protocol; recompute the n=1 floor against their corrected lens | not urgent (damage bounded inside refit noise) but the floor's baseline no longer exists | ~$15 |
| 5 | **Revive `self-report-validity`** around the decoupling hypothesis — Dehaene & Naccache now cite Ericsson & Simon on the J-space; Bhambri is ACL-citable | 1 commit, no remote since Aug | push a remote first |
| 6 | Read 2602.18458 and the ICML unit-tests paper; fold into ECOSYSTEM §3 | both may already be the register we propose | 1 hr |
| 7 | **Revoke the HF token** | outstanding since the thread began | you, 1 min |

**Process note for the roadmap itself:** items 1 and 5 both exist because I wrote design before
searching. Item 1 is the second instance this quarter. The fix is already in memory; the failure is
that memory is consulted at design time only when I remember to.
