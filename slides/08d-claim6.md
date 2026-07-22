---
category: research
section: results
weight: 86
title: "Result 7 --- post-training's point of view, on fully open weights"
slide_summary: |
  \textbf{What Anthropic report} (qualitatively, no numbers): during post-training the
  J-space ``develops some signatures of adopting `Claude's point of view' '' --- in the base
  model it tracks prediction; post-trained, it holds the model's own reactions. It rests on
  Sonnet 4.5 --- whose activations no outsider can touch; the paper's own commentators
  (Dehaene \& Naccache) could not check it. \textbf{OLMo-3 makes it checkable, and gives the
  first quantitative version.}

  \medskip
  \textbf{Post-training moves the J-space --- a lot, and by METHOD not domain.} Cosine of
  each arm's lens against the base lens, vs a 0.97 same-model refit floor:

  \medskip
  \scriptsize
  \begin{tabular}{@{}lrl@{}}
  \toprule
  \textbf{arm} & \textbf{cos(base, arm)} & \textbf{move} \\
  \midrule
  Instruct (SFT+DPO) & 0.69 & \textbf{${\sim}31$\%} \\
  Think (SFT+DPO)    & 0.73 & ${\sim}27$\% \\
  RL-Zero (RLVR only) & 0.94 & ${\sim}6$\% \\
  \midrule
  RL-Zero domain pairwise & \textbf{0.99+} & ${\sim}1$\% (Math/Code/IF/General) \\
  \bottomrule
  \end{tabular}

  \normalsize
  \medskip
  Instruction/CoT tuning reshapes the J-space \textbf{about 5x more than RLVR}, and varying
  the RLVR \emph{domain} at matched capability adds only about 1\%. The viewpoint is
  \textbf{method/format-driven and nearly domain-invariant}.

  \medskip
  \textbf{The clincher:} capability (MMLU) is \emph{flat-to-down} across post-training while
  the J-space moves about 31\%. A large representational shift with \emph{no competence gain}
  is exactly the viewpoint-not-prediction shift Anthropic describe. \textbf{Their qualitative
  claim, supported and sharpened.} Anchor-gated (identity distance 0.4\%), capability-controlled
  (RL-Zero spread 1.7pp), every number independently reproducible.
---
