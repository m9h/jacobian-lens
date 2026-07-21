---
title: "Controls for Consciousness-Indicator Claims in Language Models"
author:
  - Morgan Hough
institute: "Independent"
date: "2026-07-19"
theme: metropolis
header-includes:
  - \usepackage{booktabs}
---


# Introduction


## The claim

\textbf{July 2026.} Anthropic: \emph{“Verbalizable Representations Form a Global
Workspace in Language Models.”} Claude contains a “J-space” functionally analogous
to the global workspace of Baars and Dehaene. 1.2M views in a day.

\medskip
Reported properties: reportability, controllability, causal role in reasoning,
flexible reuse --- and \textbf{limited scope}: ablating the J-space destroys multi-step
reasoning and summarisation while sparing fluency, sentiment, and factual recall.

\medskip
\textbf{The method.} A Jacobian lens:
$$\mathrm{lens}_l(h) = \mathrm{unembed}(J_l\,h), \qquad J_l = \mathbb{E}\left[\partial h_{\mathrm{final}} / \partial h_l\right]$$
Apache-2.0, plus 38 fitted lenses for open models.



## The public response

Two camps. \textbf{Neither ran the code.}

\medskip
\textbf{(1) “It's just backprop.”} \emph{(A. Trask)} --- but that is the wrong
derivative. Backprop takes $\partial L/\partial\theta$: needs a label, moves weights.
The J-lens takes $\partial h_{\mathrm{final}}/\partial h_l$ on a \emph{frozen} model.
\emph{The steelman is better:} $J_l$ is an averaged linear map into the final residual
basis --- an analytically-derived \textbf{tuned lens}.

\medskip
\textbf{(2) “Unfalsifiable by construction.”} \emph{(E. Hoel)} --- strip global
workspace theory to reportability and it becomes trivial; predictions and inferences
“can never truly be pulled apart.” His one empirical falsification rests on a
third-party demo he declines to stand behind: a real replication “would be important
to see.”

\medskip
\textbf{State of the art for adjudicating a consciousness claim: two rhetorical
positions and no controls.}



# Background


## One move, four papers

\textbf{Write a direction into the residual stream. Read an effect. Name it after a
construct from cognitive science.}

\medskip
\scriptsize
\begin{tabular}{@{}llll@{}}
\toprule
 & \textbf{Direction from} & \textbf{Readout} & \textbf{Construct} \\
\midrule
Manifolds \tiny(Oct 25) & PCA of condition means & activation space & place cells \\
Introspection \tiny(Oct 25) & difference-in-means & verbal report & metacognition \\
Societies \tiny(Jan 26) & SAE feature & task accuracy & society of thought \\
J-space \tiny(Jul 26) & Jacobian lens & \textbf{vocabulary} & global workspace \\
\bottomrule
\end{tabular}

\normalsize
\medskip
Lineage is explicit: J-space reuses the manifolds counting task \emph{and} Lindsey's
injection protocol. Shared authors throughout; Lindsey is final author.

\medskip
\textbf{The contribution} is the readout space. PCA components have no words attached;
a vocabulary-indexed readout is what makes “verbalizable” a meaningful predicate.



## What each paper controls for

\scriptsize
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Control} & \textbf{Manif.} & \textbf{Intro.} & \textbf{SoT} & \textbf{J-space} \\
\midrule
matched random subspace      & \textbf{Y} & --- & --- & --- \\
norm-matched random vector   & --- & \textbf{Y} & --- & --- \\
negative / inverted direction & --- & \textbf{fails} & --- & --- \\
no-intervention baseline     & --- & \textbf{Y} & \textbf{Y} & \textbf{Y} \\
\textbf{model randomization} & --- & --- & --- & --- \\
\textbf{drift null (geometry)} & --- & n/a & n/a & --- \\
\bottomrule
\end{tabular}

\normalsize
\medskip
\textbf{Lindsey's introspection paper is the best controlled of the four} --- and its
weakness is one it reports itself: injecting the \emph{negation} of a concept vector was
“comparably effective,” words showing “no discernible pattern.”

\medskip
\textbf{No paper runs a model-randomization control} (Adebayo et al.\ 2018). Notably,
a co-author of \emph{An Interpretability Illusion for BERT} is on two of these papers ---
the expertise is in the room.



# Results


## Result 1 --- the lens passes randomization

Randomize the transformer \emph{blocks}; keep the \textbf{trained} embedding, final norm
and unembedding --- so token identity stays meaningful and the control cannot pass for
the wrong reason. Score two ways per layer.

\medskip
\begin{tabular}{@{}lrr@{}}
\toprule
 & \textbf{peak next\_acc} & \textbf{peak echo} \\
\midrule
trained & 0.3414 & 0.2936 \\
\textbf{random blocks} & \textbf{0.0003} & \textbf{0.0016} \\
\bottomrule
\end{tabular}

\medskip
Trained shows a clean crossover: early layers echo the current token, then echo decays
as next-token prediction climbs. \textbf{Random blocks read out nothing.}

\medskip
\textbf{The J-lens passes.} Its structure requires learned weights. The strong form of
“it's just the residual stream” is refuted --- and Anthropic never ran this control.
\emph{Controls should be able to exonerate as well as convict.}

\medskip
\tiny Trap: \texttt{model.\_init\_weights()} is a silent no-op in transformers v5. The
naive control leaves blocks fully trained and reports a confident false PASS.



## Result 2 --- the metric rewards noise

The paper's lens-quality score is \textbf{minimum rank over ${\sim}35$ layers}. That
hands a diffuse readout \emph{one lottery ticket per layer}. A confident lens correctly
reading the content puts an unrelated word far down at \emph{every} layer.

\medskip
Qwen3.5-27B, at the \texttt{\^{}} of the ASCII face, on \textbf{Anthropic's own lens}:

\medskip
\begin{tabular}{@{}lcl@{}}
\toprule
\textbf{lens} & \textbf{rank(“nose”)} & \textbf{top-5} \\
\midrule
J-lens & \textbf{2} & \texttt{smile, nose, '\^{}, noses, grin} \\
logit lens & 5 & \texttt{\textbackslash n, \textasciitilde, .., -, N} \\
\bottomrule
\end{tabular}

\medskip
Comparable scores. One has understood it is looking at a face; the other emits
punctuation. \textbf{\texttt{pass@k} cannot tell them apart.}

\medskip
This is why the logit lens “beat” the J-lens on mean \texttt{pass@10} at 4B and 8B in
our first pass --- an artifact of the metric, which we nearly reported as a refutation.



## Result 3 --- the tripartite figure is mostly drift

The sensory/workspace/motor block structure is the one part Hoel concedes is \emph{not}
baked in by construction. Build a \textbf{distance-only null}: entries depending solely
on $|i-j|$, same decay profile, \emph{zero blocks by construction}.

\medskip
\scriptsize
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{model} & \textbf{real} & \textbf{null} & \textbf{excess} \\
\midrule
qwen3-1.7b & 0.117 & 0.092 & +0.025 \\
qwen3-4b & 0.078 & 0.062 & +0.016 \\
qwen3-8b & 0.283 & 0.249 & +0.033 \\
qwen3-14b & 0.293 & 0.267 & +0.026 \\
gpt-oss-20b & 0.210 & 0.157 & \textbf{+0.053} \\
qwen3.5-27b & 0.267 & 0.217 & \textbf{+0.050} \\
\bottomrule
\end{tabular}

\normalsize
\medskip
\textbf{The null recovers 79--91\%.} Excess doubles at ${\ge}20$B but stays small.
Raw blockiness rising $0.08\!\to\!0.29$ with scale is the \emph{decay profile
steepening}, not blocks appearing.

\medskip
\textbf{Both sides overclaimed}: Anthropic on the sharpness, Hoel on the absence ---
his substitution argument generalises from small models, the regime where there is
nothing to find.



## Result 4 --- two retractions

\textbf{I reported a sharp emergence threshold.} The ASCII-face readout surfaced at
Qwen3.5-27B (rank 2) and not at dense Qwen3-14B (rank 164). Then I reported a
\emph{correction}: it tracked architecture, not scale.

\medskip
\textbf{Both were wrong.} The robust version --- the 102-vignette \texttt{association}
eval, concepts evoked but never named --- gives a smooth monotone rise in which a
\textbf{dense} 32B beats the 27B hybrid at every rank:

\medskip
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{model} & \textbf{J@1} & \textbf{J@10} & \textbf{J@50} \\
\midrule
14B dense & 0.040 & 0.091 & 0.222 \\
27B hybrid & 0.049 & 0.167 & 0.343 \\
\textbf{32B dense} & \textbf{0.062} & \textbf{0.208} & \textbf{0.438} \\
\bottomrule
\end{tabular}

\medskip
The original finding was \textbf{one lucky prompt}. Architecture-dependence is
\emph{untested}, not established.

\medskip
\textbf{This is the point of the exercise.} The discipline the field lacks is not
cleverness --- it is running the control that kills your own result, and saying so.



## Result 5 --- post-training moves the J-space, but the tidy reading is confounded

\textbf{Claim 6:} post-training shaped the J-space \emph{``toward a point of view''}
rather than pure prediction. Untestable from outside --- until OLMo-3. It is the only
open family shipping a base model, its post-trained variants, the \textbf{public training
data} (Dolma), and ${\sim}1{,}486$ checkpoints.

\medskip
\textbf{Magnitude --- holds.} $\mathrm{mean}\,\cos(J_{\text{base}}, J_{\text{instruct}})
= 0.76$ over 8 pairs, against a ${\sim}0.96$ same-model refit-noise floor. Post-training
\emph{does} move the workspace, well beyond fitting noise. First outside quantification of
the claim.

\medskip
\textbf{Structure --- confounded.} The tempting next step: $dJ = J_{\text{instruct}} -
J_{\text{base}}$ looks strongly low-rank (rank fraction $0.003$--$0.02$ vs $0.50$ for iid
drift) --- a ``structured viewpoint shift.'' But $J$ is \emph{already} low-rank, and the
change inherits it:

\medskip
\scriptsize
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{model} & \textbf{rank}$(J_{\text{base}})$ & \textbf{rank}$(dJ)$ & \textbf{ratio} \\
\midrule
gemma-3-270m & 0.048 & 0.022 & 0.45x \\
gemma-3-1b & 0.019 & 0.019 & \textbf{0.98x} \\
gemma-2-2b & 0.119 & 0.003 & 0.02x \\
\bottomrule
\end{tabular}

\normalsize
\medskip
For 2 of 3 pairs $\mathrm{rank}(dJ) \approx \mathrm{rank}(J)$: the low rank is inherited,
not evidence of concentration. \textbf{The effective-rank result must not be reported
alone} --- the third self-correction, and the confound is free to see.



## Result 6 --- one model's workspace holds a suppressed register

A cheap, no-GPU content probe (divide the embedding norm out through the motor band)
suggested OLMo-3-7B's workspace holds discourse markers over literal next-tokens. $n{=}1$,
hand-picked layers. \textbf{So I pre-registered a gate} and ran it band-wide, with a
1000-permutation null, across 7 cross-family models.

\medskip
\textbf{As a general claim it fails: significant in 1 of 7.} I had over-generalised an
OLMo-specific result by calling it ``the workspace.'' The general line is dropped.

\medskip
But OLMo was an $11.6\times$ outlier among $\sim\!1\times$, and it is real and broader than
discourse. Its workspace band preferentially expresses an informal / negative / profane /
connective register its \emph{output} layers suppress:

\medskip
\scriptsize
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{lexicon} & \textbf{OLMo-3-7B} & \textbf{Qwen3-8B (same tokens+measure)} \\
\midrule
discourse & 11.5x \; $p{=}0$ & 1.3x \; n.s. \\
negative & 12.1x \; $p{=}0$ & 0.0x \; n.s. \\
profanity & 8.6x \; $p{=}0$ & 4.1x \; marg. \\
informal & 11.0x \; $p{=}0$ & 0.0x \; n.s. \\
\bottomrule
\end{tabular}

\normalsize
\medskip
\textbf{Survives all three free controls:} permutation null; \emph{cross-model flatness}
(a broken measure would flag Qwen too --- it is flat); and \emph{frequency} (register beats
frequency-matched neutral words in 5/5 Zipf bins; $\mathrm{corr}$ with Zipf $=-0.15$).

\medskip
\textbf{Still $n{=}1$ model.} The open test is the one OLMo uniquely permits: do these
tokens trace to a distinct slice of Dolma (raw-web / toxic register)? Runnable \emph{free}
on infini-gram --- a direct test of Hoel's ``just an output transformation.'' Not yet run.



# Discussion


## Who is allowed to check any of this?

\textbf{No external party holds activations for Claude, GPT or Gemini.} There is no
programme to apply to. The published ceiling (METR, May 2026) is black-box plus raw
chain-of-thought, under NDA, with the lab approving publications --- and Apollo and
UK AISI got \textbf{under a week} with Sonnet 4.5.

\medskip
How did the invited commentators check the J-space claims? \textbf{They didn't.}
Nanda received “an advance draft” and replicated on \textbf{Qwen3.5-27B}. Dehaene and
Naccache: \emph{“We suggested to the Anthropic team that they could run exactly the
same tests\ldots”} --- the originators of the theory could not test it themselves.

\medskip
\textbf{And the checkability is borrowed.} Neuronpedia --- an MIT-licensed project
created June 2023 and run by \emph{one person} --- fits and hosts these lenses, alongside
DeepMind's and OpenMOSS's SAEs. Anthropic's own HuggingFace org publishes zero models.
The convergence criterion that is missing from the public chain lives in
\emph{Neuronpedia's} unreleased wrapper.

\medskip
Auditing all 38 published lenses: \textbf{one is defective} --- \texttt{qwen3-32b} is a
mid-fit checkpoint at $n{=}80$ against a config claiming 615. We used it as a dense
control and it returned identical J-lens and logit-lens top-5s.



# Conclusion


## Where this leaves the claim

\textbf{What survives.} The J-lens is a real instrument: it passes randomization, and it
reads content at mid-layers that a logit lens cannot. The ablation dissociation --- fluency
and recall spared, multi-step reasoning destroyed --- is a genuine finding.

\medskip
\textbf{What does not.} The sharp tripartite geometry is mostly smooth drift. The
lens-quality metric rewards noise. The emergence story was one prompt. The ``structured
point-of-view shift'' from post-training is a low rank $dJ$ inherits from $J$. And the paper
itself concedes the architecture: \emph{“no obviously separable input processors”};
broadcast “within a single feedforward pass rather than through recurrent loops.”

\medskip
\textbf{One positive result to chase.} Post-training \emph{does} move the J-space
($\cos 0.76$), and OLMo-3-7B's workspace holds a suppressed informal/charged register that
survives three controls --- the one claim whose data (Dolma) is open enough to trace.

\medskip
On Butlin \& Long's scorecard that is \textbf{GWT-1 and GWT-3 given away}. What remains
is a capacity-limited, causally-central, top-down-gated bottleneck. That is a real object.
\emph{Whether it is a global workspace is a different question.}

\medskip
\textbf{The generalisable finding.} Two independent groups --- no shared authors, neither
citing the other --- made the same move and shipped the same missing control, and both
headline claims dissolved under it. The techniques are mature; the failure literature
exists; the controls are not being run.

\medskip
\small All results reproducible: \texttt{github.com/m9h/jacobian-lens} $\cdot$
tooling: \texttt{github.com/m9h/jlens-lab}

