# Two new ladders: data quantity does *not* drive the J-space shift

**2026-08-01.** Both fitted at 40 wikitext prompts, seq_len 128, every-4th layer — a **reduced
budget**, so read the caveat before the numbers.

## 1. Data quantity — flat across a 300× sweep

OpenThoughts-Agent released a log-spaced SFT ladder on a fixed base (`Qwen/Qwen3-32B`) and fixed
method, so only the number of SFT examples varies.

| arm | SFT examples | cos(J_base, J_arm) | move |
|---|---|---|---|
| SFT-316 | 316 | 0.7152 | **28.5%** |
| SFT-1K | 1,000 | 0.8241 | 17.6% |
| SFT-3.16K | 3,160 | 0.7629 | 23.7% |
| SFT-10K | 10,000 | 0.7773 | 22.3% |
| SFT-31.6K | 31,600 | 0.7895 | 21.1% |
| SFT-100K | 100,000 | 0.7469 | 25.3% |

**No monotonic trend.** 316 examples move the J-space as far as 100,000 do. Every arm lands in
17–29%, the same band as OLMo's SFT+DPO (31%). Read with our OLMo result — where *method* set
the magnitude and *domain* added ~1% — the emerging picture is that the viewpoint **installs
early and saturates**: it is a property of *being* post-trained, not of how much.

⚠ **The caveat that governs this.** The 17.6–28.5% scatter is non-monotonic and we have **no
same-arm refit floor at this 40-prompt budget**. Our 0.97 pooled floor comes from 616-prompt
fits; at 40 prompts the floor is certainly worse and may account for the entire spread. **The
flatness is the result; the ordering within it is not interpretable** until a matched-budget
refit floor exists. That measurement is one extra fit and should be run before this is cited.

## 2. The agentic method ladder — and RL barely moves it

`Qwen3-8B` → `OpenThinkerAgent-8B-ColdStartSFTForRL` → `OpenThinkerAgent-8B-RL`.

| step | cos | move |
|---|---|---|
| base → SFT | 0.9179 | 8.2% |
| base → RL | 0.9368 | 6.3% |
| **SFT → RL (the RL step alone)** | **0.9745** | **2.5%** |

Two things stand out. Agentic SFT moves the J-space **8.2%**, far less than OLMo's SFT+DPO
(31%) — consistent with method mattering, since this is SFT alone with no preference stage. And
the **RL step alone moves it 2.5%**, the smallest post-training effect we have measured,
extending our OLMo RLVR finding (~6%) to a different family and a different domain.

Note base→RL (6.3%) is *smaller* than base→SFT (8.2%): RL moved the representation partly back
toward base. Suggestive, but within the noise this budget can support.

## Bearing on Ring-Zero

[Ring-Zero](https://arxiv.org/abs/2607.12395) reports emergent self-verification from zero-RL at
1T parameters. We measure the RL step moving the workspace **2.5%** at 8B. Those are compatible
only if the emergent behaviour is not a workspace-level change — i.e. if it is behavioural
rather than representational, or scale-gated. It is a live tension, and rung 8 of the curriculum
sets it as an exercise rather than resolving it.

## Reproduce

`modal_data_ladder.py` (B200, adaptive dim_batch) · `spark_agent_ladder.py` (DGX Spark, /mnt/t9).
