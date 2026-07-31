# Does the workspace signal predict *agent task failure*? — design, and an honest cost

**2026-07-31.** Scoping note. Not yet run.

## Why this experiment and not best-of-N

Our best-of-N test failed (`results/posttrain/bestofn_validity.md`): the covert workspace
signal gave +0.008 over a mean-logprob baseline and was *worse* per-sample (AUROC 0.556 vs
0.584). The diagnosis was that **question-level and sample-level metacognition are different
tasks** and we only have the first — "is this model likely to be wrong here", not "which of
these 8 attempts is right".

Agentic tasks are matched to the signal we actually have:

- **Trajectory-level self-assessment** — *am I stuck, is this approach working* — is a
  question-level judgement, not a within-question ranking.
- **Ground truth is executable.** Terminal-Bench tasks pass or fail by running tests. No
  alias-matching, no LLM judge.
- **Contamination is far harder** than for TriviaQA, which is exactly the reuse problem
  RewardBench 2 avoided by commissioning new prompts — and which we walked into.

## What is actually available (checked)

| artifact | contents | usable as-is? |
|---|---|---|
| `open-thoughts/OpenThoughts-TBLite` | 1,848 files: per task a Dockerfile, `instruction.md`, `solution/solve.sh`, `task.toml` | **the benchmark**, not trajectories |
| `open-thoughts/OpenThoughts-Agent-v1-RL` | `tasks.parquet` + extraction script | task *definitions* only |
| `OpenThoughts-Agent-SFT-*` | teacher traces, ≥5-turn filtered | **successful** traces — no failure labels |

**There are no ready-made labelled trajectories.** Every published trace is a filtered success,
which is the wrong side of the label we need. So this experiment requires *generating* failures
by running an agent — which is the whole cost.

## Design

1. **Model:** `OpenThinkerAgent-8B-RL`, not the 32B. Eight billion parameters keeps residual
   capture cheap, and it is a released RL'd agent so its failures are representative rather than
   artefacts of an untrained policy.
2. **Tasks:** ~50 TBLite tasks, sampled across difficulty. Each runs in its own Docker sandbox
   via the terminus-2 harness.
3. **Label:** the task's own tests — pass/fail. Objective, no judge.
4. **Signal:** capture the layer-L residual at the end of the agent's **first** reasoning turn,
   before it has evidence of success. That is the prediction worth having: *does the model know
   early that this will not work?*
5. **Probe:** cross-validated difference-in-means, split **by task** so no leakage — identical to
   the metacognition probe, which needs no lens and so runs on any checkpoint.
6. **Baselines that must be beaten:**
   - mean logprob of the first turn (the model's own confidence),
   - task-length / instruction-length priors (a probe that only learns "long tasks fail" is
     measuring the task, not the model),
   - a random-direction probe of matched norm.

## Cost, stated plainly

This is **not** the cheap experiment. Fifty agent trajectories in Docker sandboxes, each
potentially many turns and minutes, plus a forward pass with hidden states per trajectory. Call
it a day of engineering for the harness and a few hours of GPU — materially more than the ~$4
best-of-N run. The build is the harness, not the science.

**Mitigation:** the Morgan/OREL side already has two Terminal-Bench tasks (NODDI, TUS) built
around simulator-as-verifier, so the harness pattern is familiar rather than new.

## What each outcome would mean

- **Probe beats logprob and the length prior** → the metacognitive cell has downstream validity
  in RewardBench 2's sense, on a contamination-resistant, objectively-labelled substrate. That
  is the strongest result available to the Scorecard.
- **Probe ties logprob** → the signal is real but redundant with output confidence; the cell
  should be described as descriptive, not reliability-relevant.
- **Probe fails** → question-level metacognition does not transfer to trajectory-level either,
  which materially narrows the claim and should be published as such.

All three are publishable. The second and third are the ones a grant application would otherwise
paper over.
