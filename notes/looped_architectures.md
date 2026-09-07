# Looped architectures and what they break for introspection

**Research note, 2026-09-07.** Prompted by Nanbeige4.2-3B-Base. Everything in §1 was checked
against the config and modeling code directly; §2 onward is analysis and marked where it is
speculation.

## 1. What Nanbeige 4.2 actually is

[`Nanbeige/Nanbeige4.2-3B-Base`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base), **Apache-2.0**,
en/zh, 28T training tokens (up from 23T). Their card claims it beats Qwen3.5-4B-Base and
Gemma4-E4B-Base at comparable non-embedding scale — **their benchmark table, unverified by us**.

From `config.json`:

```
num_hidden_layers : 22
num_loops         : 2          ← the whole point
loop_loss_weights : []
skip_loop_final_norm : False
hidden_size 3072 · 48 heads · 8 KV heads (GQA) · vocab 166144 · max_pos 131072
```

The card: *"hidden states are fed back into the same Transformer layers after completing a
bottom-to-top pass. The reuse schema significantly increases the model capacity without increasing
the number of parameters."*

So: **44 layer-applications over 22 distinct parameter blocks.**

Two details from `modeling_nanbeige.py` worth having:

- **LoopSplit is dormant in 4.2.** The code implements a sandwich — unlooped prefix, repeated
  middle block, unlooped suffix — but it is gated on `enable_double_loop_split`, which is
  **absent from 4.2's config** (`getattr(..., False)`). The card confirms LoopSplit, mHC with depth
  attention, and concatenated n-gram embeddings are for **4.5**. 4.2 loops the **full stack**.
- ★ **Their KV cache is already loop-indexed**: `layer_idx + loop_idx * num_hidden_layers`, giving
  44 slots. And `_supports_default_dynamic_cache()` returns True only when `num_loops == 1`.
  **The addressing problem is solved in their own code.** Interpretability tooling has not adopted it.

## 2. Layer index stops being an address

Nearly every method in [ECOSYSTEM](https://github.com/m9h/spinning-up-in-mech-interp/blob/master/ECOSYSTEM.md)
is layer-indexed, and the index silently assumes each layer runs exactly once:

| method | address | ambiguous under looping? |
|---|---|---|
| logit lens / tuned lens | read out at layer `L` | yes |
| **Jacobian lens** — our own `J̄_l` | fitted **per layer** | yes |
| SAEs | `blocks.7.hook_resid_pre` | yes |
| attribution graphs | node = (layer, position) | yes |
| `interp-engine`'s 34 "points" | `resid_post.16` | yes |

`resid_post.7` in a looped model is **two different computations sharing one name** — same weights,
different residual state, different function. The fix is mechanical (address by `(layer, loop)`,
exactly as their cache does), but nothing in the open stack does it today.

**This is a good test case for `interp-engine` specifically.** Its pitch is standardised addresses
across architectures; a looped model is precisely the architecture a flat address space cannot
name. Worth raising with Decode Research.

## 3. What it breaks, in order of how much it matters

**SAEs would pool two distributions.** Collect activations at layer 7 across both passes and you
train one dictionary over two computational regimes that happen to share weights. Features come out
as a mixture. Either train per `(layer, loop)`, or first *measure* whether the two passes'
activation distributions actually differ. **That measurement is the cheapest useful experiment here
and nobody has run it** — and it is a real question, not a formality: if the distributions are
near-identical the whole problem shrinks.

**The Jacobian lens survives formally, but its reading changes.** `J̄_l` is still the derivative of
a composed function. What gets muddy is the *interpretation* — "what layer 3 writes that layer 18
reads" — when layer 3's weights also act at unrolled position 25. More concretely: our OLMo result
that movement is early-layer concentrated (49.9% in L0–9 vs 10.6% in L21–30) is a claim about
**parameter depth**. In a looped model parameter depth (22) and computational depth (44) diverge,
and any such claim has to say which one it means.

**Developmental and scale-ladder work needs care.** "Layer" as a proxy for "depth into the
computation" stops holding.

## 4. ★ The reason to care: recurrence-dependent indicators

Our Consciousness-Indicator Scorecard work leans on the Butlin/Long framework, where **recurrent
processing** is an indicator that a strictly feedforward transformer cannot satisfy *by
construction*. That has made it a dead entry — untestable rather than unsatisfied.

A looped transformer has genuine state feedback through the same weights, so it is the first
**open-weights, laptop-scale** substrate where such indicators can be **tested rather than assumed
absent**.

⚠️ **But do not overclaim, and the reason is precise.** A *fixed* 2× unroll is formally equivalent
to a 44-layer feedforward network with tied weights. It satisfies "the same weights act at multiple
depths"; it does **not** satisfy "dynamic, input-dependent iteration," which is what recurrent
processing theory (Lamme) actually requires. The strong case would be an **adaptive-depth** looped
model where loop count varies with the input. Nanbeige 4.2 is the weak case, and the honest framing
is that it makes the indicator *measurable* rather than *satisfied*.

## 5. Latent iteration vs token-space iteration — a testable question

This connects to the Kambhampati thread ([2504.09762](https://arxiv.org/abs/2504.09762) and
*Beyond Semantics*, TMLR 2026), where swapped and semantically empty traces preserve or improve
accuracy — suggesting intermediate tokens may supply computation rather than content.

A looped model does its refinement **in latent space**; chain-of-thought does it **in token space**.
Comparing them isolates the question:

> **Does a looped model need less chain-of-thought for the same accuracy than an unlooped model of
> matched capability?**

If latent looping substitutes for token-space "reasoning," that is direct evidence the tokens were
doing computation rather than reporting it. This is a real experiment at 3B scale and, as far as we
know, unrun. *(We have not checked the literature on this — apply the usual rule before building.)*

## 6. Practical notes

3B, Apache-2.0, laptop-scale — a plausible curriculum substrate. But it needs `trust_remote_code`
(custom `NanbeigeForCausalLM`, `model_type: nanbeige`), so tools assuming standard architectures
will not load it. That likely includes TransformerLens, and is worth checking for `interp-engine`.

## What we have not done

Not read the [technical report](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/Nanbeige42_report.pdf).
Not run the model. Not measured whether the two passes' activations differ (§3). Benchmark claims
are theirs. The §5 experiment is proposed, not surveyed.
