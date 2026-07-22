---
category: research
section: results
weight: 82
title: "Result 5 --- post-training moves the J-space, but the tidy reading is confounded"
slide_summary: |
  \textbf{Anthropic's post-training finding} (qualitative): the J-space ``develops some
  signatures of adopting `Claude's point of view' ''. Untestable from outside --- until OLMo-3. It is the only
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
---
