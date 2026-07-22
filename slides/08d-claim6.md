---
category: research
section: results
weight: 86
title: "Result 7 --- Claim 6, tested on fully open weights"
slide_summary: |
  \textbf{Anthropic's Claim 6:} post-training shaped the J-space \emph{``toward a point of
  view rather than pure prediction.''} It rests on Sonnet 4.5 --- whose activations no
  outsider can touch; the paper's own commentators (Dehaene \& Naccache) could not check it.
  \textbf{OLMo-3 makes it checkable on FULLY OPEN artifacts.} First external test.

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
  is exactly ``a point of view rather than pure prediction.'' \textbf{Claim 6 supported ---
  and sharpened.} Anchor-gated (identity distance 0.4\%), capability-controlled (RL-Zero
  spread 1.7pp), every number independently reproducible.
---
