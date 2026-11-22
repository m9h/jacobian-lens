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

## 3. Roadmap, re-ranked (8 Oct; item 0 added and item 6 retracted 15 Oct)

| # | item | why now | cost |
|---|---|---|---|
| **0** | ★ **Estimator control on the ladder — DONE 15 Oct** (`results/posttrain/estimator_control_rlens.md`): R-lens refit of base + Instruct, same 616 prompts. cos(base, Instruct) 0.689 → **0.784**; early-layer concentration survives but 4.9× → 3.5×; the J-lens overstated layer-0 movement because it is noisier on the Instruct model (cross-estimator cos 0.34 vs 0.66 on base). **Open:** RL-Zero arms under the R-lens (~$70) before quoting the ~10× method ratio; no R-lens floor yet | the claim survives, smaller and flatter; quote 0.78 (R-lens) in Foresight | ~$35 spent |
| 1 | ★ **Huginn developmental J-lens**: fit the lens on a log-spaced subsample of the 46 checkpoints; ask *when* the workspace signature (write/ablate along lens directions, the 2-recurrence window) appears. Controls: shuffled-weights null at each checkpoint, final-checkpoint positive. **Write the design *after* reading Blayney et al. and Luo & Yu in full** | Wang & Reid's stated open problem; public substrate; weight-tying makes it cheap; nobody found doing it | ~140 GB download; adapter must be rebuilt (no code release); est. 10–20 GPU-h |
| 2 | **Submit Foresight by 31 Oct 23:59 PDT**, Coordination & Accountability call; read the Airtable form first | deadline now known | 1 form |
| 3 | **Re-run the metacognition unanswerable control with mixed fitting** (2609.10739) and add OC/SJ conflict items (2607.16799) | our exposed control has a published fix; our clean result has a published alternative explanation | ~$5 |
| 4 | `looped_architectures.md` reframed — **done 8 Oct** | | |
| 5 | **SF-OSMI**: offer the crit methodology; one session with the salmon opener | your call on the relationship | — |
| 6 | RoPE erratum — **retracted for our lenses 15 Oct**: the July Modal image was transformers **5.14.1** (fixed); the 8 Oct "refit" measured version drift (0.2% at L0). The bug's real effect, by emulation in our pipeline: L0 cos 0.886, L30 identity +4% — the reporter's number. Neuronpedia's Sept refit is correct (our prompt-0 reproduces their row 1 to 0.1%); the "5% discrepancy" was two effects cancelling. Nothing to raise; a confirmation note is drafted (`notes/neuronpedia_refit_discrepancy_draft.md`) — **your call** | `ROPE_ERRATUM.md` §Retraction; PITFALLS #30 | ~$12 + ~$2 spent |
| 7 | `self-report-validity` revived + remote — **done 8 Oct** | | |
| 8 | **Revoke the HF token** | outstanding since the thread began | you, 1 min |

**Process note:** items 1 and 7 both exist because I wrote design before searching; item 1 is the
second instance this quarter. This pass added a third failure mode: a failed grep declared a
sourced claim unsourced. The rule generalises — *provenance unknown* is itself a claim that needs
a second look before it is written down.

## 4. Upstream check, 22 Oct 2026

What moved since 15 Oct, in order of consequence for the ladder.

**Brauer, *J-lens shouldn't target the final layer by default*** (LessWrong `Adb55vLqt33LEyBfz`,
25 Sep; S1). On DeepSeek-V3 one output direction written by the last block (Chinese-vs-Latin
tokens) is the top Jacobian direction at every earlier layer under a final-layer target; 46% of
mid-depth readouts on English text go majority-Chinese, 3% with a penultimate target. 81% of 78
public lenses target the final layer; ours do. **This is a control we had not run on the
flagship pair**: if post-training rewrites the last block's dominant output direction, every
earlier layer inherits the change and "early layers move most" is partly a target-side artifact.
Lens-only, no GPU: `experiments/final_block_control.py` measures per layer the share of
‖ΔJ_l‖² lying in the top-k output directions of J₃₀ (both arms), whether u₁(J_l) is u₁(J₃₀), and
the cos(base, Instruct) that remains after projecting those directions out. Result:
`results/final_block_control/final_block_control.md`. **Result: survives.** The base lens has
the Brauer structure (47% of J₀'s energy in one direction, cos 0.81 to the last block's top
direction; it unembeds to the unigram-frequency axis), Instruct does not, but only 16–23% of the
L0 movement lies in those directions and projecting them out moves the cosine 0.25 → 0.29.
**Side finding, bigger than the control:** the depth profile is statistic-dependent — the
off-identity cosine is highest at L0 (0.84 R-lens) and lowest at L18–24 (0.72) under both
estimators, and Instruct's late transports sit 15–40% further from the identity than base's.
Absolute movement is early; relative restructuring is late. The write-ups carry a banner; the
Foresight text should use the two-clause version.

**anthropics/jacobian-lens#15** (dthinkr, 5 Sep; S1) is the RoPE report in its original form; it
triggered Neuronpedia's refit and its emulation numbers match ours to 0.1% (ROPE_ERRATUM.md
§Addendum 22 Oct). The reporter's Hub discussion on *our* repo is still unanswered; reply drafted
(`notes/hf_discussion_1_reply_draft.md`). Other open issues there worth knowing: #5 (input-copying
ceiling; `apply()` below the fitting floor of 16 positions), #6 (`save()` fp16 overflow → `inf` on
badly conditioned fits — ours are O(1), max |J| 1.14 at L30, checked), #4 (content digest for
lens files). The repo has had no commit since the 1 Jul release.

**WorkspaceBench** (Blank, Bhatia, Ong & Nanda, 23 Sep; S1 of the post). Method-agnostic reader
eval on Qwen3.6-27B, 3,356 items, ground truth by construction, LLM judges, no interventions.
Relevant to the Scorecard as the nearest existing adjudication instrument for this method
family; tailored to one model; MATS project.

**Foreword** (Kim, Feldt & Yoo, [2610.28187](https://arxiv.org/abs/2610.28187), 20 Oct; S2 +
partial S1). Software-testing paper that uses published J-lens readouts (Llama-3.1-8B,
Gemma-2-9B, Qwen3-8B, layers at 45–90% depth) as features to rank test inputs by failure risk
before generation; +0.14 ROC-AUC over Clotho. No lens-validity control; sensitivity to corpus,
vocab and layer range listed as untested. A downstream consumer of lens files, two hops from
anyone who could check them — the Foresight "who consumes an unvalidated probe" paragraph has a
second example beside Drift Calibrator.

**The post-training axis now has company.** `vincentoh/olmo-3-7b-jlens` (placeholder, nothing
uploaded) plans Instruct + SFT lenses with per-layer validated/provisional labels and a
known-answer test from `bigsnarfdude/jlens-toolkit` (Apache-2.0, Aug), whose README already
points readers at `mhough/olmo3-jacobian-lenses`. Our Hub repo is the reference artifact for
this axis and was last touched 2 Aug: it carries the 0.69 number without the estimator caveat,
no fit-environment statement, and an unanswered RoPE thread. **Refreshing it is the cheapest
high-value item on this list** (README + the eleven results files with banners + the R-lens
write-up; the R-lens `.pt` files follow when the Spark/Modal path is back — the SSH agent on
Legion refused the key after the 21 Oct reboot).

**No change:** jpp_lens and koayon/jpp-lenses only gained a citation block (17 Oct);
Neuronpedia's HF repo gained J++ files (15 Oct) and the monorepo only UI work on the lens
pages; the OLMo-3 model repos are unchanged (the Instruct repo's 25 Jun commit is README-only);
no new arXiv paper on the lens besides Foreword and *JLD* (2610.05967, unrelated: perceptual
distance).

### Roadmap deltas

| # | item | status 22 Oct |
|---|---|---|
| 0a | **Final-block inherited-direction control** on the J-lens pair | **done**, $0; survives; depth-profile caveat added to four write-ups |
| 0b | Same control on the R-lens pair; penultimate-target refit of base + Instruct if 0a flags inheritance | needs Modal (blocked on SSH agent); ~$35 if the refit is needed |
| 0c | RL-Zero arms under the R-lens before quoting the ~10× ratio | unchanged, ~$70, not approved |
| 9 | **Refresh the Hub repo** (README, banners, R-lens write-up, fit environment) | doing now |
| 10 | Reply to Hub discussion #1 | drafted; your call |

## 5. "Weight-space programming" (Takehi/Teoh/Zhang/Khattab, MIT CSAIL, posted 2 Nov 2026)

Blog post, no paper; logs CC BY 4.0 at `RikiyaT/lm_understand-logs`; edited Qwen3.5-0.8B on the Hub (Apache-2.0).
Lens-only check done the same day (`results/weight_space_programming/direction_identity.md`, $0): the experiment-8 edit
is Arditi-style weight orthogonalisation with α = 6 on 64 neurons at L15/L19, direction from an activation contrast
(productive vs repetitive contexts), cos(u₁₅,u₁₉) 0.85, not an unembedding/frequency artefact; reads as "close the
thought" vs "keep going". Gain is loop termination (576/579 rescues were truncations). The agent ran its own random-direction
control (+1.6 pp, SE 3.5) that the post does not report. Roadmap delta: item 0d, GPU controls 1–5 in the results file
(~$40–60, Modal), ranked after 0b; item 5 (J-lens sensitivity to a rank-1 behaviour-changing edit) is the one that
feeds our own Claim-6 interpretation.

**3 Nov addendum.** Simon Willison reproduced experiment 1 with Qwen 3.8 27B locally (3/4); Honnibal (framing conflates
no-tools weight reading with Python-and-forward-pass work) and Goedecke (a human would do the same; first-layer norms
give the game away) critiqued. Our readout of the four released networks (`results/weight_space_programming/exp1_firstlayer_maps.md`):
agent-channel energy 97% in one row (Breakout) / 96% in one column (Pong), maze walls visible as zeros (Pac-Man), crossing
lane + "up" bias (Freeway); the prompt supplies the channel legend. Proposed control: permute input and action indices
and drop the legend (four prompts per model). Resume point if interrupted: GPU controls 1–5 in `direction_identity.md`
once `ssh dgx` works; the permutation control needs only API access to any frontier model.

## 6. Marin / Open Athena as a second substrate (17 Nov 2026)

Morgan attended "Scaling Marin's Open LLM Pretraining" (Open Athena; Liang, Hall, Konwinski, Held)
and proposed Marin as an alternative to OLMo-3. Checked the same evening from the Hub API, the
model cards and the `marin-community/marin` issue tracker. **Nothing was run.**

What is public, by how it maps onto our axes:

| our axis | OLMo-3 (what we used) | Marin (verified on the Hub, 17 Nov) |
|---|---|---|
| post-training stages, dense, stock `transformers` | 7B base → SFT → DPO → RLVR; RL-Zero domain arms | **Marin 8B: SFT only** (`marin-8b-instruct`, 5.3B tokens, Tülu-3 mixture among the sets). `marin-8b-dpo-lora-…` exists but is an empty repo |
| post-training stages, any architecture | as above | **Snowball 67B-A2B** (Grug MoE: 26 layers, width 2560, 256 routed experts, 4 active + 1 shared; 134 GB bf16): base cooldown → chat SFT → thinking SFT → agentic SFT (`laion/snowball-67b-a2b-*`, Apache-2.0) → ~25 RLVR arms, merges, v0 / v0.1 / v0.1.1, identity-SFT arms (`open-athena/*`, OpenMDW-1.1). Far more arms than OLMo-3, negatives included |
| same, small enough to iterate | — | **Snowflake** d512 / d768 / d1024 base (0.75B params, 1.5 GB at d512) with Datakit-SFT partners at d512 and d768; same Grug MoE code path |
| pretraining trajectory | 1,487 checkpoint branches | Marin 8B: 44 `step-*` tags + 7 named milestones whose data mix and LR phase are documented (kestrel → deeper-starling, 2.7T → 12.7T tokens); 32B: 5 phase branches |
| scale with architecture held fixed | none | **Delphi**: 89 Qwen3-architecture models, 157M → 25B, one tokenizer, one data recipe, final checkpoints only |
| **seed replicates** | none | Delphi: 3 seeds at 3.4B and at 9.7B, 2 at 447M |
| licence | Apache-2.0 | Apache-2.0 (8B, 32B, Delphi, laion stage checkpoints); OpenMDW-1.1 on `open-athena/*` (not read by us yet) |

Assessment. Marin is **not a drop-in replacement** for the Claim 6 result: the clean dense
SFT/DPO/RLVR ladder that stock `transformers` loads is still OLMo-3's. It is the better substrate
for three things OLMo-3 cannot give:

1. **A noise floor.** We quote base→Instruct cos 0.78 with no idea what two lenses of the *same*
   recipe score. Delphi's seed triplets answer that. This is the control the result has always lacked.
2. **A second family for the SFT step.** Marin 8B base vs instruct, same pipeline, unchanged code.
3. **Many post-training arms on one base**, if J-lens is ported to Grug MoE. Open method question:
   top-4 routing makes the Jacobian piecewise; a corpus-averaged lens averages over routes.

They already do interpretability in the open, controls-first: issue #10696 (logit lens, direct
logit attribution, patching on Snowball's identity; a steering vector rejected because control
prompts moved as much as identity prompts). The Snowball SFT report lists "runaway thinking and
looping after SFT" as an open problem; our decoded loop-termination direction (§5) is a concrete
thing to offer. Marin trains in JAX (Levanter), so the unclaimed JAX J-lens port has a home.

Roadmap items (none started, none approved for spend):

- **M1** Delphi seed-pair lens noise floor at 447M, then 3.4B. Cheapest; gates how we quote 0.78.
- **M2** Marin 8B base vs instruct, J-lens and R-lens. About the cost of one OLMo pair.
- **M3** Port to Grug MoE on Snowflake d512 base vs Datakit-SFT; only then consider Snowball (B200-class, 134 GB per checkpoint).
- **M4** Marin 8B trajectory across the documented phase boundaries (does the lens move at the cooldown onto higher-quality data?).

## 7. Lambert's "open problems in post-training" deck and Trillium Labs (read 22 Nov 2026)

Source: Nathan Lambert, "Pushing Frontiers in Open Language Model Post-Training", Cornell talk,
slides dated 20 Nov 2026 (`natolambert.com/slides/2026/cornell-post-training/talk`, 51 slides + 3
extra). Read via a fetch-and-summarise tool, so slide wording below is paraphrase, not quotation.
The deck says it mainly states his goals at **Trillium Labs** (`trilliumlabs.org`, which today
serves only the name). Funders, compute and legal form are not stated anywhere we could read. A
third-party write-up calls him its executive director; the deck does not say what happened to
his AI2 role and we have not checked.

What the deck claims, in the parts that touch us:

- **Six open problems**: harness specialisation and design; character training; open-source RL
  environments; reasoning efficiency in RL; scaling laws for RL; multi-agent scaling laws.
- **Base models are disappearing.** Slide 15: a matrix of 12 open-weight labs, 2023 to 19 Nov
  2026; most moved from releasing base models to releasing post-trained weights only within the
  last 12 months. Slide 13 puts doing post-training well at roughly $50M+ a year.
- **Character training** is named as one of Trillium's first areas. The deck says open literature
  has not kept up with industry, and that in Open Character Training (Maiya, Bartsch, Lambert,
  Hubinger, arXiv 2511.01689) measuring performance was extremely hard. The measurement shown is
  behavioural (trait-preference Elo on Llama 3.1 8B). **Representations, activations, probes,
  steering and interpretability do not appear in the deck at all.**
- **Scaling laws for RL**: asks what the equivalent of a pretraining ladder is for RL, and how to
  predict a large post-training run from small ones (cites Khatri et al. 2025, ScaleRL).
- Trillium's stated goals: establish an area of study (example given: empirical character
  training in RL), establish shared language, build reusable datasets or models (example given:
  ones that enable AI-safety non-profits), change how people think about a problem. Hiring MTS,
  student researchers, interns. The LinkedIn post invites people working on these problems in
  the open to get in touch.

### How it bears on the program

1. **Character training is the measurement gap our Claim 6 result already sits in.** We measured
   what post-training does to the model's internal viewpoint on his own model family (OLMo-3):
   base→Instruct lens cosine 0.69 (J-lens) / 0.78 (R-lens) with MMLU flat to down, SFT+DPO moving
   the lens about 10× more than RLVR once the refit floor is removed (29.4% vs 2.9%; the README
   still prints the uncorrected "~5×"). That is an internal, capability-decoupled measure of the
   thing the deck says it cannot measure well. It is not a measure of *which* character; it says
   how far the viewpoint moved, not where to.
2. **The character-training artifacts are public and nobody has put a lens on them.** Verified
   on the Hub today: `maius/{llama-3.1-8b-it,qwen-2.5-7b-it,gemma-3-4b-it}-personas` (ten
   persona LoRAs each, rank 64, all attention + MLP projections), matching `-pt-distillation`
   (stage one) and `-pt-introspection` repos, and a `llama-3.1-8b-it-personas-no-cons` variant
   (name suggests a no-constitution control; card not read). The Qwen-2.5-7B set is Apache-2.0;
   Llama is under the Llama 3.1 licence and its misalignment adapter is gated. Prior work on
   these and their descendants, from abstracts and posts only:
   - *Persona Cartography* (Baines et al., arXiv 2607.07916): OCEAN-trait LoRAs on six models,
     behavioural evaluation; adapters scale monotonically and add roughly linearly.
   - *Persona Geology* (Ratnam & Africa, LessWrong 30 Oct 2026): 134 trait LoRAs on Qwen3.5-4B,
     **weight-space** factor analysis, released as `EternalRecursion/persona-lora-zoo-qwen35`
     (Apache-2.0), second seed for 40 traits. Their own caveat matters to us: two seeds of the
     same trait have weight cosine +0.018, and the signal only appears once directions are
     activation-weighted (0.66 same-trait vs 0.09 different-trait). The authors call it an
     orphaned direction and ask for collaborators.
   - Mitrani (LessWrong, 25 May 2026): OCT persona classifier macro-F1 falls from 0.86–0.94 in
     chat to 0.29–0.55 in an agentic email task; most of the drop traced to the scaffolding.
   - Anthropic's persona vectors (2025) are activation directions per trait; not a lens.
   None of these reports a logit-lens, tuned-lens or Jacobian-lens comparison. Not searched
   exhaustively; a proper pass is owed before any spend (PITFALLS: literature before design).
3. **"Base models are disappearing" raises the value of OLMo-3 and Marin and narrows our
   method.** The base→post lens comparison needs a base. Slide 15 is the strongest outside
   statement we have of why the fully open ladders are the only place this can be done. The
   deck's "post-post-training" examples (Harvey's Tenet on Kimi K3, Mercor/SkyRL on
   Qwen3.5-397B, TMax on Qwen 2B–27B) are post-trained→specialised pairs, which the lens can
   also compare, at sizes from trivial to out of reach.
4. **Scaling laws for RL.** Our numbers say RLVR barely moves the lens (2.9% beyond floor on
   OLMo-3; 2.5% for the RL step alone on OpenThinkerAgent-8B). If that holds, lens movement is a
   poor predictor of RL progress and a good detector of everything else a recipe does. The only
   open thing resembling an "RL ladder" we know of is Marin's ~25 Snowball RLVR arms (§6, M3).
5. **Reasoning efficiency in RL** (Never Give Up) touches §5 only loosely: the decoded edit's
   gain was loop termination, a compute-allocation effect. No experiment follows from this yet.
6. Harness design, open RL environments and multi-agent scaling laws are outside this program.

### Roadmap items (none started, none approved for spend)

- **C1** Lens on character adapters: Qwen-2.5-7B-Instruct vs each of the ten merged persona
  LoRAs, R-lens first, with the refit floor. Questions: is the movement above floor; is it the
  same size across personas (as RL-Zero domains were) or trait-dependent; where in depth.
  Add the distillation stage and the no-constitution variant as the within-method controls.
- **C2** Noise floor for C1 from *Persona Geology*'s 40 seed pairs on Qwen3.5-4B: lens cosine
  between two seeds of one trait vs two traits. Their weight-space result (+0.018) predicts
  weight geometry will not show it; does the lens.
- **C3** Does lens distance per persona predict Mitrani's chat→agentic drop per persona? Only
  worth doing if C1 shows trait-dependent movement.
- Housekeeping: update the README headline ratio to the floor-corrected figure.

Outreach to Lambert / Trillium is Morgan's call. The concrete thing to offer is the OLMo-3
result plus C1 as an internal measurement for character training, both on fully open weights.
