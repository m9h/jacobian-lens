# Literature review and roadmap — 6–8 Oct 2026

**Status: complete** (8 Oct). The 6 Oct draft was interrupted by a context compaction; §2 is now
filled in, one claim in §1 is **corrected**, and the roadmap in §3 is re-ranked. Evidence tier marked
per item (S1 full text · S2 abstract/page/primary listing · S3 search summary — unverified).

## 1. What changed in the literature

### ★ Our looped-architectures note was pre-empted — and the lesson is the one we keep relearning
**Wang & Reid, *Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive
Recurrence?*** ([arXiv 2609.01924](https://arxiv.org/abs/2609.01924), Fin AI Research, **1 Sep 2026**).
S1 — read. I wrote [`looped_architectures.md`](looped_architectures.md) on **7 Sep** without finding
it. Six days. The note's own §5 says *"apply the usual rule before building"* and I did not apply it
to §2–4. Filed against `feedback-literature-review-before-design` — again.

What they did, which is more than the note proposed: a **virtual-unrolling adapter** that gives the
J-lens `(layer, loop)` addressing (the exact gap §2 of the note said tooling lacked), then the full
workspace suite — lens fitting, readout, **eleven causal experiment families** — on **Ouro-2.6B**
(48 layers × 4, deeply supervised) and **Huginn-0125** (4-layer core × 16, latent reasoning), against
Qwen3.6-27B. **No code release is linked anywhere in the text** (grep for URLs finds only
lesswrong.com and transformer-circuits.pub), so the adapter would have to be rebuilt.

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
- **An instrument blind spot they state precisely (§7.4):** `J = E[∂h_target/∂h_v]` is built from
  influence on the next token, so *held-but-idle* content maps to ~0. *"The lens sees the workspace's
  working content, not its inventory."* Every study using J-lens artefacts inherits this — including
  our capacity-style and metacognition readouts.
- **Nanbeige is not tested** (0 mentions). Their stated limitation: *"one pre-trained checkpoint per
  family, so architecture is confounded with training recipe and scale. Following checkpoints
  throughout pretraining would test whether our conclusions are stable over training and reveal
  when J-space emerges."* **That is our developmental axis, named as the open problem — and §2.2
  below finds a substrate for it.**

### ✅ CORRECTION — the "SFT collapses the language-selectivity gate" claim IS sourced
The 6 Oct draft said this claim had *"provenance unknown — not in Wang & Reid by grep"* and marked
it do-not-cite. **My grep was wrong.** It is Wang & Reid **§7.5 and Appendix C**, measured on
**Ouro-2.6B-Thinking** (reasoning SFT of their base checkpoint, ~8.3M examples), S1:

> *"the language-selectivity gate collapses to always-on: the passage's language label is readable
> at ceiling even when task-irrelevant (∆ +0.25 → +0.00) … consistent with reasoning SFT keeping
> more context features unconditionally active in the workspace. Second, ignition sharpens: the
> in-band 10–90 transition width drops 0.55 → 0.20 … **Both are single-condition observations; we
> report them as leads rather than conclusions.**"*

Everything else — writes spanning every loop, introspection at top-1 31% — reproduces on the SFT
model "within a few points." So: **their one post-training comparison agrees with our ladder's
direction** (post-training reshapes workspace *gating*, not workspace *existence*), at n=1 SFT
stage, flagged by the authors as a lead. Citable with that qualifier. Process note: a failed grep
nearly turned a sourced claim into an "unsourced" one — the mirror image of fabrication, and as
wrong. Re-grep with the hyphenated form before declaring provenance unknown.

### The CoT-trace thread firmed up
**Bhambri, Biswas & Kambhampati, *Interpretable Traces, Unexpected Outcomes*** — **ACL 2026 long**
([2026.acl-long.1686](https://aclanthology.org/2026.acl-long.1686/)). S2. The QA replication of the
swapped-trace result on *pretrained* Llama/Qwen is peer-reviewed, not a preprint. Still unread in
full.

## 2. The searches that were pending (completed 8 Oct)

### 2.1 Foresight — deadline found (S2, primary page)
[AI for Science & Safety Nodes RFP](https://foresight.org/grants/ai-science-safety-nodes-rfp/):
**31 October 2026, 23:59 PDT.** Grants "typically $30,000 to $100,000". Three calls; ours is
**Coordination and Accountability** — *"how independent institutions can assess AI risk."*
Required: *"work product (code, data, and outputs) … open-sourced."* Review ≈ 3 months after
deadline. Application is an Airtable form (JS-rendered; field list and any length limits **not
yet read** — open it in a browser before submitting). No word limit stated on the RFP page.

### 2.2 ★ Substrates — one looped family publishes the pretraining trajectory
Checked the Hub refs directly (S2, primary):

| family | intermediate pretraining checkpoints | note |
|---|---|---|
| **Huginn-0125** | **46 branches, `step_00001024` → `step_00047104`, uniform 1,024-step spacing**, in [`JonasGeiping/huginn-0125-checkpoints`](https://huggingface.co/JonasGeiping/huginn-0125-checkpoints); each **14.26 GB** (fp32, 3 shards) | README: *"In-training checkpoints only to be used for analysis."* Final run = 47,000 steps / 800B tokens; main card still says "8 … ~350 on request", the checkpoint repo has 46 |
| Ouro-1.4B / 2.6B | none found — final models only | 7.7T tokens, five-stage recipe |
| Nanbeige 4.2 | none; **4.5 not yet released** ("later in 2026") | |

**So Wang & Reid's named open problem — "when J-space emerges" through pretraining in a looped
model — has a public substrate, and it is the one of their two models where the workspace
survives recurrence.** 46 × 14 GB = 650 GB total; a 10-checkpoint log-spaced subsample is 140 GB.
Weight tying means one iteration's lens fit covers full depth (their cos 0.9991), so the per-
checkpoint fit is cheap. I did not find any paper running workspace or J-lens analysis across
these checkpoints. (A search summary mentioned "a Lyapunov exponent across 8 Huginn training
checkpoints"; **source not identified** — not in 2609.36585, 2609.19934, or 2610.02185 by abstract.
S3, do not cite.)

Other looped-model literature found, S2 (abstracts):
- Blayney, Arroyo, Obando-Ceron, Castro, Courville, Bronstein & Dong, *A Mechanistic Analysis of
  Looped Reasoning LMs* ([2604.11791](https://arxiv.org/abs/2604.11791)) — *"each layer in the cycle
  converges to a distinct fixed point"*; recurrent blocks learn stages mirroring feedforward
  models, repeated per iteration. 39 pp, 63 figs.
- Luo & Yu, *Prediction Dynamics in Depth-Recurrent LMs* ([2609.21383](https://arxiv.org/abs/2609.21383))
  — Huginn-3.5B and Ouro-1.4B; why predictions persist across depth while scores change.
- Dau, Khuat & Dung, *Beyond Depth Truncation* ([2609.19934](https://arxiv.org/abs/2609.19934)) —
  Depth Control Protocol with positive and negative controls for the depth-truncation evaluation.
- Liu et al., *Decoding Looped Transformers Better for (Almost) Free* ([2610.02185](https://arxiv.org/abs/2610.02185),
  1 Oct) — uses earlier-loop states as decoding guidance; four looped families.
- Marchenko et al., *Closing the Loop* ([2610.00673](https://arxiv.org/abs/2610.00673), 30 Sep) —
  looped training in 310B tokens; converts Qwen3-1.7B to looped. Release status not visible in
  abstract.

### 2.3 Probe validity — the answer-readout confound now has three named papers
All S2 (abstracts), all bearing on `results/posttrain/metacognition_result.md`:
- ★ **Diagnosing Correctness Probes under Self-Judgement Confounding** ([2607.16799](https://arxiv.org/abs/2607.16799)):
  objective correctness (OC) and the model's own self-judgement (SJ) usually agree, so a
  "correctness" probe cannot say which it tracks. On conflict cases, probes follow SJ. Across four
  instruction-tuned models ≤14B the **SJ direction transfers cross-domain; the OC direction has a
  below-chance point estimate.** Controls: answer-likelihood adjustment, length normalisation,
  null directions. **This is exactly the ambiguity in our covert error-monitoring result** — our
  AUROCs may be SJ, not OC, and only conflict items can tell.
- ★ **The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes**
  ([2609.10739](https://arxiv.org/abs/2609.10739)): on contexts where the true bit and the
  prescribed answer coincide, a probe *"cannot tell which of the two it measures"* — **this is
  solarkyle's constant-truth-slice confound, named and given a fix**: mixed fitting across
  contexts where the labels differ (0.006 → 1.000 AUROC in their game), or randomised output
  codebooks. That is the re-run design for our exposed unanswerable control.
- **When Do Internal Probes Beat Reading the Answer?** ([2609.04582](https://arxiv.org/abs/2609.04582),
  Villuri, Shaik & Doboli): three regimes — concealed (probe yes, margin no), **miscalibrated**
  (both yes, threshold kills behaviour; +4.6σ offset), undetected. Probe 0.96 AUC vs 50% behaviour
  on a 0.6B model. A control for "covert" claims: check the margin before calling it covert.

### 2.4 Adjudication — two methods now exist in print; the standing function still does not
- ★ **The Story is Not the Science: Execution-Grounded Evaluation of MI Research**
  ([2602.18458](https://arxiv.org/abs/2602.18458), Bai, Baumgartner, Sun, Holtzman & Tan, UChicago;
  code [ChicagoHAI/MechEvalAgent](https://github.com/ChicagoHAI/MechEvalAgent/)). S2 + partial HTML.
  30 outputs (10 agent replications, 10 agent open-ended, **10 human-written repos from ICLR/
  NeurIPS/Alignment Forum**). *"Over 90% of the tasks have at least one failure in
  reproducibility, largely driven by execution errors."* Catches 67 of 87 human-found issues and
  **surfaces 51 more**; >80% agreement with humans; 30 min vs 2.2 h per evaluation. Most common
  coherence failure: missing statistical significance (27.3%).
- ★ **When Circuits Are Too Broad: Unit Tests for MI** ([ICML 2026](https://icml.cc/virtual/2026/79409);
  no arXiv ID found). S2. *"a negative-control protocol for evaluating the specificity of circuit
  and feature claims"* — survives nuisance rewrites, fails on matched negatives, avoids off-target
  damage, beats same-budget baselines; **specificity frontiers** (target effect vs off-target
  damage). Self-described as *"a falsification layer for mechanistic claims, not a new discovery
  method."* That is our register, in the ICML proceedings.
- **Proposal consequence:** "Nobody's actual job is checking your results" survives — both are
  papers, not functions — but the proposal must cite them or it reads as unaware. Done (see
  `proposal/CITATION_AUDIT.md` round 4).

### 2.5 Agentic interpretability (Sep–Oct)
- **AgenticInterpBench / HyVE** ([2606.24026](https://arxiv.org/abs/2606.24026), Findings of EMNLP
  2026; rev. 2 Sep): **84 semi-synthetic transformer circuits with 163 component-level
  annotations** — ground truth by construction, the same move as our `positive_control.py`. Agents
  fail *"later in the validation loop, through incomplete validation plans, code execution
  errors, or unresolved hypotheses."* *"Reliable validation remains the key obstacle."* No
  shuffled/random-circuit null visible in the abstract.
- *Automated Interpretability and Feature Discovery with Agents* ([2605.01555](https://arxiv.org/abs/2605.01555))
  — Gemma-2 + weight-sparse MLP neurons; agent loops beat one-shot autointerp, produce auditable
  traces. *Agentic-imodels* ([2605.03808](https://arxiv.org/abs/2605.03808)). S3 both.
- **The Ignition Index** ([2608.05160](https://arxiv.org/abs/2608.05160), Rahbar, single author;
  code CC-BY-4.0): four-parameter sigmoid on per-layer probe accuracy → steepness β̂ as a GWT
  "ignition" scalar; 11 models incl. **Huginn**, Mamba, Pythia; **shuffled-label control** (9.6×
  selectivity). Does not use the J-lens or cite the workspace paper. Scorecard-adjacent and
  controls-first; worth a look as a candidate indicator implementation.
- Martian's $1M prize round closed 1 Feb 2026; no new round found.

## 3. Roadmap, re-ranked (8 Oct)

| # | item | why now | cost |
|---|---|---|---|
| 1 | ★ **Huginn developmental J-lens**: fit the lens on a log-spaced subsample of the 46 checkpoints; ask *when* the workspace signature (write/ablate along lens directions, the 2-recurrence window) appears. Controls: shuffled-weights null at each checkpoint, final-checkpoint positive. **Write the design *after* reading Blayney et al. and Luo & Yu in full** | Wang & Reid's stated open problem; public substrate; weight-tying makes it cheap; nobody found doing it | ~140 GB download; adapter must be rebuilt (no code release); est. 10–20 GPU-h |
| 2 | **Submit Foresight by 31 Oct 23:59 PDT**, Coordination & Accountability call; read the Airtable form first | deadline now known | 1 form |
| 3 | **Re-run the metacognition unanswerable control with mixed fitting** (2609.10739) and add OC/SJ conflict items (2607.16799) | our exposed control has a published fix; our clean result has a published alternative explanation | ~$5 |
| 4 | `looped_architectures.md` reframed — **done 8 Oct** | | |
| 5 | **SF-OSMI**: offer the crit methodology; one session with the salmon opener | your call on the relationship | — |
| 6 | RoPE refit, base lens — **done 8 Oct**: bug effect ≤0.2% cos (L0), 0 at L≥12; floor restored; **new open item: Neuronpedia's own refit moved 5% at L30, ours 0.003%** — raise with them | | ~$12 spent |
| 7 | `self-report-validity` revived + remote — **done 8 Oct** | | |
| 8 | **Revoke the HF token** | outstanding since the thread began | you, 1 min |

**Process note:** items 1 and 7 both exist because I wrote design before searching; item 1 is the
second instance this quarter. This pass added a third failure mode: a failed grep declared a
sourced claim unsourced. The rule generalises — *provenance unknown* is itself a claim that needs
a second look before it is written down.
