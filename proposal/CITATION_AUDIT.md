# Citation audit — what the proposal's claims actually rest on

The ACL 2026 auditability call proposes **source-based auditing**: dependency chains showing which
claims hold up which other claims. This applies it to our own proposal, because a document arguing
that the field does not check its sources should be able to say how well it checked its own.

**Evidence depth, worst to best:**

| tier | meaning |
|---|---|
| ⬛ **S3 — search summary** | I never opened the source. A search engine's paraphrase, or a subagent's report. |
| 🟨 **S2 — abstract / landing page** | I fetched the paper's abstract page or a lab's public page, not the paper. |
| 🟩 **S1 — full text** | I read the paper body. |
| ✅ **S0 — verified by us** | We ran it, or fetched the primary artifact directly (API, repo metadata). |

---

## ⬛ S3 — load-bearing claims resting on a search summary

**These are the ones to fix before submission.** Each is a specific quantitative or quoted claim
about someone else's work that I have only thirdhand.

| claim in the proposal | status |
|---|---|
| *"Anthropic's circuit tracing on Claude 3.5 Haiku produced satisfying insight for about a quarter of tested prompts"* | ⬛ **A specific quantitative claim about another lab's flagship result, taken from a search-result paraphrase.** I have not found the primary source. **Do not submit as stated.** |
| *"DeepMind's months-long Chinchilla circuit analysis produced a brittle, partial explanation"* | ⬛ same paraphrase, no primary source located |
| *"DeepMind has publicly shifted from 'ambitious reverse-engineering' to 'pragmatic interpretability'"* | ⬛ quoted phrases I never sourced to a document |
| MIB's *"SAE features are not better than neurons"* | ⬛ quoted from a search summary of the paper. The **claim** is corroborated independently by Transluce (S1), but **the quotation is not verified.** |
| MIT Tech Review "2026 Breakthrough Technology" | ⬛ trivially checkable, not checked |
| NDIF **$9M NSF**; Goodfire **$50M Series A**; Open Phil ~$336M / $46–50M yr | ⬛ search summaries; plausible, unverified |
| All ARC numbers — DiARC 0.63%, ARChitects 50.5 vs 51.5, Greenblatt 50%, BARC +13.75, SOAR c=1000, the 2026 leaderboard (98% / 92.5% / 30.2%), Duck Harness, the ~68-point gap | ⬛ **from our research subagent's report.** Well-sourced *in that report* with links, but I did not open the underlying papers. The report itself lists what it could not verify. |
| *"Anthropic's own report: no static transduction solution above 10%"* | ⬛ same |

## 🟨 S2 — abstract or landing page only

| claim | note |
|---|---|
| *The Dead Salmons of AI Interpretability* (2512.18792) | abstract page. **The framing device of the whole proposal.** The characterisation of which methods fail on random nets comes from the abstract; I have not seen its experiments or numbers. |
| *Make MI Auditable* (2606.00033) — the three mechanisms | abstract page. The mechanisms and "position paper, nothing implemented" are from the abstract. |
| Diplomacy — *More Victories, Less Cooperation* (2406.04643) | ⚠️ **search results + a USC ISI news article.** The key sentence — *"changing inputs related to communication did not significantly impact its high score"* — is **from the university's press page, not the paper.** My reading of it as an *ablation* is an inference. **Read the paper before this claim leaves the building.** |
| Tracr (2301.05062), TracrBench (2409.13714) | search summaries; the mechanism is well known but unread here |
| AI Agent Behavioral Science (Nature HSSC); Machine Psychology | search summaries; the "moving beyond architectural interpretability" quote is unverified |
| AgentTrace (2602.10133) | search summary |
| Model Organism Lottery; Activation Oracles blind-spots (2607.23379); Wang et al. rLLC (2410.02984) | search summaries (curriculum, not proposal) |
| Martian $1M prize; Decode Research funders; TSG Lab publications | fetched their own public pages — primary, but self-descriptive |
| Nisbett & Wilson (1977); Ericsson & Simon | ⬛/🟨 **from my own knowledge, not fetched this session.** Well-established, but the specific bridge to CoT is my argument, not a citation. |

## 🟩 S1 — read in full

| source | what it supports |
|---|---|
| Transluce, *Circuits Are Sparse in the Neuron Basis* | the neuron-basis result, ~100×, RelP, three reproduced case studies |
| **ABC**, *Rigorous Agentic Benchmarks* (2507.02825v2) | τ-bench 38%, SWE-Lancer 100%, SWE-bench-Verified 24% of top-50, CVE-Bench 33%, and the checklist's structure. **Reading this corrected an attribution error and softened a claim.** |

## ✅ S0 — verified by us this session

| claim | how |
|---|---|
| Rung 5's attention-sink bug: activation ~123 at position 0, content max 16.6, mean 0.029; 28× over-steer; steering ladder 0/5, 0/5, 5/5 | ran it |
| Gate result: +3.24 vs +0.71 and +1.37; corr +0.459 among active positions | ran it |
| All 6 rungs + gate pass from a clean venv on torch 2.13 / transformers 5.14 | ran it |
| Diplomacy paper has 13 citations, **zero** touching faithfulness/CoT/interpretability | Semantic Scholar API |
| SAELens → `decoderesearch/SAELens`; circuit-tracer → `decoderesearch/`; lucent last pushed 2025-03-21; tuned-lens 2025-08-07 | GitHub API |
| ARENA deep link 404; OREL 404; Microscope 503; MIB leaderboard live | HTTP status |
| Our own results — SoT reversal, J-lens/NLA 42×, OLMo ~10×, Adebayo pass, three retractions | our published artifacts |

---

## What to do before submission

1. **Cut or source the four ⬛ claims in the first table's top rows.** The Anthropic "quarter of prompts" and DeepMind quotes are the highest-risk items in the document: specific, quotable, about named labs, and unsourced. A reviewer from either lab would notice.
2. **Read the Diplomacy paper.** It is now a worked case in the proposal and my "ablation" reading came from a press release.
3. **Read the dead salmon paper.** It is the opening frame.
4. **Spot-check three ARC numbers** against the primary sources our research report links.

Nothing here is known to be false. The point is that **"I could not find a problem" and "I checked" are different claims**, and this proposal has been making the second while doing the first.
