# Project context

- This repository contains research code illustrating experiments from *Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations* by Hugo Lavenant and Giacomo Zanella. The primary reference is the August 11, 2026 revision identified in `.codex/paper-context.md`.
- The Python code currently corresponds specifically to Section 6, **“Estimating the information profile.”** This includes estimator (19), Algorithm 2, the ModernBERT/WikiText-2 information-profile experiment, and the exact and asymptotic optimal-schedule computations shown in Figure 1.
- For work involving the experiments, algorithms, notation, figures, or theoretical claims, consult `.codex/paper-context.md` when it is available and read the linked current revision, especially Section 6, before modifying the Python code.
- The code is intended for a public repository, but it is a collection of runnable research scripts rather than an installable Python package. Do not introduce packaging metadata or reorganize it as a package unless explicitly requested.
- The intended numerical scope is Figure 1 only. Keep the scripts convenient to run interactively in Spyder and assume CPU execution; do not add performance tuning or unrelated experiments without a request.
- Use Python-standard `snake_case` for functions and variables and `UPPER_SNAKE_CASE` for constants.
- Keep the committed files in `data/`, and document how they can be regenerated from WikiText-2 and ModernBERT.
- Keep machine-specific paths and other local-only research context out of version control.
