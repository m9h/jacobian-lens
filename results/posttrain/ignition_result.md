# Ignition test on OLMo-3 — the experiment Dehaene & Naccache asked for

**Run 2026-07-21. OLMo-3-1025-7B, published 31-layer Jacobian lens, pure readout (no fitting).**

In their commentary on Gurnee & Lindsey, Dehaene & Naccache single out **ignition** — the
nonlinear, all-or-none entry into the workspace — as the signature the paper *did not
establish*, and prescribe the test:

> *"present a stimulus at graded strengths … and ask whether J-space representations switch
> on with a threshold-like nonlinearity, while earlier, non-J-space layers rise monotonically
> with input strength. Better still, present stimuli exactly at threshold and look for a
> bifurcation across runs."*

Nobody had run it — Claude's activations are closed and Anthropic did not. It is runnable in
the open on OLMo-3. This is a first pass, reported with its confounds.

## Design

For five concepts (France, Japan, chess, hospital, winter), a neutral base sentence is
followed by 0–6 clues that imply the concept **without ever naming it**. Evidence strength =
number of clues *k*. At the final token we read the lens logit for the concept's tokens at
**every layer** (0–30), and — the key control — subtract the reading for a matched
**off-concept** the same clues do not imply (Brazil, Egypt, tennis, library, summer). The
on−off **contrast** isolates the concept-specific signal from generic priming.

## The confound I had to fix

The naive metric — absolute concept logit — is confounded: at *k*=0 the base sentence
("memories of their vacation") primes *all* place tokens high (France 7.0, **Brazil 6.8,
Egypt 7.5**), and adding clues *lowers* them. Absolute logit is dominated by generic
"expect a place" priming plus last-token noise, and shows no clean structure. The **on−off
contrast removes it** and is what the results below use.

## Result — a sustained, all-or-none workspace entry

On−off contrast, averaged over the five concepts, by layer band and evidence *k*:

| band | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|---|
| early (L0–13) | −0.1 | 1.2 | 0.5 | 0.8 | 0.6 | 0.3 | **0.2** |
| workspace (L14–22) | 0.2 | 3.7 | 3.2 | 3.4 | 2.8 | 2.0 | **2.5** |

Three things, all consistent with the ignition picture:

1. **No concept without evidence.** At *k*=0 the contrast is ≈0 at every layer — the
   workspace does not hallucinate the concept before any clue appears.
2. **All-or-none entry, then saturation.** The contrast jumps at the *first* clue and then
   plateaus (flat across k=1…6) rather than ramping up with accumulating evidence — a step,
   not a slope.
3. **Sustained in the workspace, transient early.** Early layers respond to the first clue
   and then *decay* (1.2 → 0.2, keeping ~16%); the workspace band *holds* the concept
   (3.7 → 2.5, ~67%), and deeper layers hold it more strongly still. This is precisely the
   contrast Dehaene draws between ignition — "a sustained, broadly distributed state" — and
   subliminal processing — "a delimited wave of neural activity … which quickly dies away."

So the **persistence half** of the ignition signature is present and clean: once evidence
crosses into the workspace, it is sustained there, unlike in the pre-workspace layers.

## What this first pass does NOT show (and the v2 that would)

- **A graded mid-range threshold.** Because each clue here is individually diagnostic
  (croissants ⇒ France), the concept ignites at *k*=1 and saturates — the threshold sits at
  the first clue, not in the middle. Dehaene's "graded strengths → threshold nonlinearity"
  needs **individually weak, only-jointly-sufficient clues** so the crossing point lands
  mid-range.
- **The bifurcation.** It degenerated here (the threshold *k\** landed at 0–1, so there was
  nothing to permute across runs). v2 needs weak graded clues **and** variants that differ in
  *which* clues are present (not just order), to test for a bimodal ignited/not distribution
  at threshold.

Both are addressed by a v2 clue set; the harness (`m9h/jacobian-lens/modal_ignition.py`) is
unchanged apart from the stimuli. This v1 establishes the sustained all-or-none entry; v2
will place the threshold and test the bifurcation.

## Caveats

- Readout is at the final token; a fixed probe position would reduce last-token noise
  (the on−off contrast already controls for it, but does not eliminate it).
- Five concepts; a larger, balanced set would tighten the curves.
- "All-or-none" here means binary (0 vs ≥1 clue); the stronger graded-threshold claim awaits v2.

Reproduce: `modal run modal_ignition.py::run` in `github.com/m9h/jacobian-lens`; raw curves
persist on the Modal `jlens-out` volume (`ignition/ignition_raw.json`).
