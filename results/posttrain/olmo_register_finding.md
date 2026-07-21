# OLMo-3-7B holds a suppressed informal/charged register in its workspace

## The finding
The Phase-0 gate killed "the workspace holds discourse markers" as a GENERAL claim
(1/7 models). But OLMo-3-7B was a 11.6x outlier, and on inspection it is real and broader
than discourse markers. Enrichment in the top 5% of the workspace/output ratio, vs a
permutation null:

               OLMo-3-7B          Qwen3-8B (control, identical tokens+measure)
  discourse    11.5x  p=0.000     1.3x  p=0.37  n.s.
  negative     12.1x  p=0.000     0.0x  p=1.00  n.s.
  profanity     8.6x  p=0.000     4.1x  p=0.03  marginal (n=3)
  informal     11.0x  p=0.000     0.0x  p=1.00  n.s.

**OLMo's workspace band preferentially expresses an informal / negative / profane /
connective register that its OUTPUT (motor) layers suppress.** 9-12x beyond chance,
across all four predefined lexicons, and absent in six other models.

## Why it is real, not an artifact
- Tokenizer confound ruled out: OLMo has 61 markers in vocab, Qwen3-8B 61 (identical).
- Robust across band choices: 3.6x-14.4x, all p=0, for every workspace-band definition.
- OLMo-SPECIFIC: on the same tokens with the same measure, Qwen is flat. A frequency or
  embedding-norm artifact of these tokens would show in Qwen too; it does not. So the
  effect is a property of OLMo (its data/training), not of the tokens or the measure.

## Caveats (do not drop)
- Motor-band denominator is imperfect: OLMo motor identity_distance is 0.22-0.42, not ~0,
  so the "divide out embedding norm" is approximate. The cross-model flatness (Qwen) is
  the control that handles "is the MEASURE broken" -- a broken measure would flag Qwen too.
- One remaining free control not yet run: a FREQUENCY-MATCHED NEUTRAL word set. Cross-model
  flatness is strong indirect evidence, but a direct frequency control would seal it.
- Characterised in detail on OLMo vs Qwen; the other 5 models were flat on the gate.

## Why this reframes the OLMo program (better than the original plan)
The finding is a claim about DATA and TRAINING, and OLMo is the one model where both are
inspectable AND the only one showing the effect:
- DATA (infini-gram, $0): do these workspace-held tokens trace to a distinct slice of
  Dolma (raw web / toxic register)? Direct test of Hoel's "just an output transformation":
  if workspace content maps to a specific data register, it encodes the training
  distribution, not mere output accumulation.
- TRAINING (checkpoint sweep, ~$25): does the hold-but-suppress separation emerge
  gradually over pretraining?

Recommendation: run the FREE infini-gram attribution FIRST; it is now the central test.
Hold the paid checkpoint sweep until attribution supports the data-register story.

## Frequency control (the flagged gap) — PASSES
- corr(workspace/output ratio, Zipf frequency) = -0.148. The ratio is not a frequency
  proxy (would be strongly + if it were).
- Register words beat frequency-matched neutral words in 5/5 Zipf bins. Within every
  frequency band, the workspace holds register over neutral. **Effect is register, not
  frequency.** All three controls now pass: real, OLMo-specific, not frequency.
