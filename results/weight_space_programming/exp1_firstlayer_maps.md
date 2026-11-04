# Experiment 1 of "weight-space programming" ("which game?"): what the first layer shows without reading anything

*3 Nov 2026, CPU, $0. Inputs: the four released networks `exp1_which_game/networks/<id>.npz` (CC BY 4.0) and
`eval/answer_key.json`. Script `experiments/wsp_exp1_firstlayer_maps.py`; numbers `exp1_firstlayer_maps.json`;
full maps `exp1_firstlayer_maps.log`. Context: Simon Willison (3 Nov) reproduced the no-tools game identification
with Qwen 3.8 27B locally (3 of 4, about an hour each) and Kimi K3.1 Pro (1 of 1); Goedecke argued the first-layer
L2 norms give the game away; HN commenters disputed how obvious that is. Only the four chosen seeds are released,
so no cross-seed classifier is possible; this is the readout itself.*

## The readout

Per input (row, column, channel), the L2 norm of its 32 first-layer weights. Inner 8×8, scaled 0–9 to each
network's maximum. The prompt tells the model what the five channels mean and asks, per channel, "where in the grid
its weights are large".

| game | agent-channel energy | where the agent channel's energy sits | moving-object channel | output-layer norms by action (noop, up, down, left, right) | b3 |
|---|---|---|---|---|---|
| Breakout | 2% of W1 | **97% in row 8** (bottom row) | 49%, rows 2–7 | 6.2, 6.5, 6.4, **10.9, 8.9** | flat |
| Pong | 5% | **96% in column 1** (left wall) | 52%, whole field | 3.5, **5.2, 5.4**, 3.4, 3.4 | flat |
| Pac-Man | 75% | every row and column, **with the maze walls as zeros** | 21%, whole field | 4.7, 7.1, 8.0, 6.5, 6.8 | noop −0.87 |
| Freeway | 74% | a central band, columns 3–6, rows 2–8 | 10%, every row (lanes) | 5.2, 7.2, 5.2, 6.3, 7.0 | **up +3.14**, down −1.86 |

Agent channel, inner 8×8 (0–9):

```
Breakout      Pong          Pac-Man       Freeway
00000000      40000000      65767444      00000000
00000000      50000000      50079004      00456431
00000000      20000000      70679506      00584221
00000000      10000000      89600876      01488610
00000000      20000000      77600787      11478711
00000000      30000000      40676806      15999751
00000000      30000000      50056004      13886611
42221124      30000000      75768556      12778410
```

A horizontal paddle that only moves left/right; a vertical paddle on one wall that only moves up/down; an agent
that can be anywhere except a symmetric set of wall cells and uses all four moves; an agent confined to a crossing
lane whose policy prior is "up", with moving objects in every row. The reward channel is below 2% of the energy in
every network, so "what gives reward" is not read from the weights at all; it is inferred from the layout.

## What this says about the experiment

1. **The task is picture-reading, and the prompt frames it as one.** The channel legend ("agent", "moving objects",
   "reward targets", "obstacles") plus the row/column indexing turn 17,253 floats into five labelled 8×8 images, and
   the question asks for exactly the per-channel saliency description above. Two numbers (which channel dominates,
   which actions dominate) separate the four games. That a 27B model can do this from text in an hour, and a
   frontier model in 90 seconds, is a real result about reading structured numeric text, not about "understanding
   weights".
2. **The missing control is a permutation.** Randomly permute the 500 input indices and the 5 action indices in the
   prompt (and drop the channel legend). The network is functionally unchanged; the picture is gone. If accuracy
   falls to chance, the capability is layout-reading. If it survives, that is the claim the post makes. Nobody has
   run it; it costs four prompts per model.
3. **Four items, open answer, one seed each.** Chance for an open answer is low, which is why 4/4 looks strong, but
   the four games were chosen to have distinct layouts and the released evidence cannot say what happens on games
   whose first-layer pictures look alike (two paddle games on the same wall, say). The twelve trained seeds exist;
   the eight unreleased ones would let anyone test seed-to-seed consistency of the readout.
4. **For our program** this is the Foreword pattern again (a published readout consumed without a validity control),
   now with the control costing nothing. It belongs on the scorecard as the "metadata-in-prompt" family alongside
   the answer-readout confound (`notes/lit_review_roadmap_2026-10.md` §2.3).
