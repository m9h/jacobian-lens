# Cross-validation against Neuronpedia's independently fitted OLMo-3-7B J-lens

**Date:** 2026-07-25

Neuronpedia (in collaboration with Anthropic) publishes pre-fitted Jacobian lenses at
`huggingface.co/neuronpedia/jacobian-lens`, including `olmo-3-1025-7b`. We publish our own fit
of the same base model at `mhough/olmo3-jacobian-lenses`. This is the first external check of
our fitting pipeline against a second party's independent fit of the same model.

## Are they independent fits?

Yes. **0 of 11 shared layers are bit-identical.** Both converged at `n_prompts = 616` on the
same corpus under the same stopping rule, which is a coincidence of the protocol, not a copy.

Their fit (from `config.yaml`): `allenai/Olmo-3-1025-7B`, Salesforce/wikitext
wikitext-103-raw-v1 train, `max_chars 2000`, `n_prompts 1000` (stopped at 616),
`dim_batch 128`, `max_seq_len 128`, bfloat16, `stop_at_delta 0.002`, on a B200. Their
`convergence.csv` ends at `identity_distance = 0.220851` — the published anchor value our
pipeline was validated against.

## Is the estimator the same? — **Yes, identically so**

This was the main methodological risk, and it is resolved. `jlens_lab/fitting.py` does:

```python
from jlens.fitting import jacobian_for_prompt
```

i.e. our fitting **wraps Anthropic's own `jlens` package** and uses the identical primitive
Neuronpedia does (their `neuronpedia_utils/jlens/` is a verbatim copy of the same library).
The convention, from that package's own docstring:

> gradient at source position `p` is then `sum_{p' >= p} dh_final[p'] / dh_l[p]`, the sum over
> later target positions; we take the mean over source positions

So the documented caveat — that `J̄` is **not** a plain per-position `∂h_final[p]/∂h_l[p]`
average — is a property of the *shared reference implementation*, applying equally to both
fits. It is a caveat about what `J̄` means, **not a discrepancy between us and Neuronpedia.**
Position masking (`valid_position_mask`, which drops leading attention-sink positions and the
final position) is likewise shared; their `convergence.csv` records `n_valid_positions = 111`
of `seq_len = 128`, consistent with ours.

## Result: agreement is layer-dependent

| layer | cosine (float64) | rel. Frobenius diff | identity distance (theirs / ours) |
|---:|---:|---:|---:|
| 0 | 0.8844 | 0.478 | 1.1643 / 1.1696 |
| 3 | 0.8993 | 0.447 | 1.0884 / 1.0995 |
| 6 | 0.9439 | 0.331 | 0.9878 / 0.9946 |
| 9 | 0.9563 | 0.294 | 0.9591 / 0.9754 |
| 12 | 0.9763 | 0.217 | 0.8906 / 0.9042 |
| 15 | 0.9860 | 0.167 | 0.7839 / 0.7976 |
| 18 | 0.9928 | 0.121 | 0.6830 / 0.6743 |
| 21 | 0.9955 | 0.095 | 0.6130 / 0.6010 |
| 24 | 0.9972 | 0.075 | 0.5429 / 0.5343 |
| 27 | 0.9985 | 0.054 | 0.4555 / 0.4564 |
| 30 | **0.9998** | **0.020** | 0.2211 / 0.2201 |

At the final layer the two independent fits agree to **cosine 0.9998**, and their identity
distances differ by 0.4% (0.2211 vs 0.2201, against the published 0.220851). **Our pipeline is
externally validated.**

*(Compute cosine in float64. Over 16.7M-element matrices, float32 accumulation gives values
above 1.0 — we saw 1.0057 — which is pure numerical error.)*

## The more important finding: a per-layer refit-noise floor

The disagreement is **strongly layer-dependent**: two independent fits of the *same model*,
same corpus, same protocol, same estimator differ by **48% in Frobenius norm at layer 0** and
**2% at layer 30**.

This is a free measurement of the refit-noise floor, and it has a direct consequence for our
own results:

- **Early-layer J-lens readouts are weakly determined.** A cosine of ~0.88 between two fits of
  the *identical* model at layer 0 sets the ceiling on what any early-layer J-space comparison
  can claim.
- **Our post-training result must be read per-layer.** We report post-training moving the
  J-space to cosine ~0.69–0.76 from base. Against a *single* pooled refit floor (~0.96) that is
  clearly a real effect. But the floor is not uniform: at layer 30 the floor is 0.9998, so a
  0.69 is overwhelming; at layer 0 the floor is 0.884, so the margin is far smaller.

**Action:** report the post-training J-space change as excess over a *per-layer* refit floor,
not a pooled one. This does not overturn the result — the effect is well above the floor at the
mid-to-late layers where the workspace band sits — but it changes how the early layers may be
described, and it is the honest way to present it.

## Reproduce

```python
from huggingface_hub import hf_hub_download
import torch
A = torch.load(hf_hub_download("neuronpedia/jacobian-lens",
      "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt"),
      map_location="cpu", weights_only=False)["J"]
B = torch.load(hf_hub_download("mhough/olmo3-jacobian-lenses", "lenses/olmo-3-1025-7b.pt"),
      map_location="cpu", weights_only=False)["J"]
for l in sorted(set(A) & set(B)):
    x, y = A[l].double(), B[l].double()
    print(l, (x.flatten() @ y.flatten() / (x.norm() * y.norm())).item(),
             ((x - y).norm() / x.norm()).item())
```
