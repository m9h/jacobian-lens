# Draft reply — huggingface.co/mhough/olmo3-jacobian-lenses/discussions/1 (open since 2026-09-05, unanswered)

*Drafted 2026-10-22 by the agent. Not posted — Morgan's call. The thread is @venvoo's heads-up
that our `uv.lock` resolves transformers 5.9.0 and that the 11 lenses therefore encode the
buggy RoPE. The lock was never the fit environment; see `results/ROPE_ERRATUM.md` §Retraction.*

---

Thanks for this, and apologies for the six-week silence — it took that long to work out what the
lenses were actually fitted with, and the answer surprised us.

**The lock was not the environment.** The 11 lenses were fitted on Modal from an image whose pin
was `transformers>=5.5,<6` with no lock applied; the cached image from 21 July resolved
**5.14.1**, which has the per-layer RoPE fix from #46911. We confirmed this by probing the cached
image itself rather than inferring from `uv.lock` (which was only ever used for local tooling),
and then by refitting the base lens under 5.19: cosine ≥ 0.998 at every layer, identity distance
at L30 0.22015 vs 0.22014. So the files here encode the corrected forward.

**Your numbers are right, and we reproduce them.** Forcing YaRN back onto the sliding layers in
our pipeline (`modal_rope_emulation_test.py`) moves the single-prompt L30 identity distance from
0.24290 to 0.23380 on prompt 0 of the wikitext stream — Neuronpedia's convergence.csv row 1 is
0.242644 (Sept) and 0.233647 (June), i.e. your +4–6% and your emulation match to 0.1%. Over 8
prompts the bug costs cosine 0.886 at L0 and 0.9998 at L30. Everything you said about the
bug's size is correct; what was wrong was our own README/lock, which implied an environment we
did not use.

**What we changed.** (1) Every lens fitted since 15 Oct carries a `provenance` dict (transformers,
torch, commit, prompt set, estimator) — your point that the file should record its own forward
was the right one, and we adopted Neuronpedia's format. (2) The "refit floor" you note is a
clean sampling floor: both sides of that comparison were on the corrected forward after
Neuronpedia's 21 Sept refit as well. (3) The README here now states the fit environment and
links the full working, including the two days we spent believing you were right about our files.

One more thing, since you measured downstream effects: the identity-distance gate we used to
anchor fits (±5% against the published lens) cannot distinguish a wrong-RoPE fit from a
different-prompt-set fit of the same model — the two effects are the same size and opposite in
sign on OLMo-3-7B. Our July gate passed for that reason, not because it was checking the thing we
thought. Documented in `jlens-lab`.
