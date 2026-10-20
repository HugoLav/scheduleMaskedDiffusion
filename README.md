# Information-profile experiments for masked diffusion schedules

This repository contains the Python code and numerical data used for Figure 1 of
*Error Bounds and Optimal Schedules for Masked Diffusions with Factorized
Approximations* by Hugo Lavenant and Giacomo Zanella.

The code corresponds to Section 6, **Estimating the information profile**. It
implements estimator (19), Algorithm 2, the normalized information profiles in
Figure 1(a), and the exact and asymptotic schedules in Figure 1(b).

## Repository layout

- `src/`: experiment and plotting scripts.
- `data/`: tokenized WikiText-2 chunks and raw conditional log-probabilities.
- `figures/`: generated figures.

This is a collection of research scripts, not an installable Python package.

## Environment

The scripts require Python and the dependencies listed in `requirements.txt`.
For example, in an existing Python environment:

```text
python -m pip install -r requirements.txt
```

The experiments use `answerdotai/ModernBERT-base` on CPU by default. The first
run that tokenizes data or evaluates the model downloads resources from
Hugging Face.

## Running from Spyder

Each executable script keeps its editable parameters near the top and can be
run directly in Spyder. Paths are resolved relative to the repository rather
than Spyder's current working directory.

To regenerate everything from the original corpus:

1. Run `src/compute_tokenize_dataset.py` to download and tokenize WikiText-2.
2. In `src/compute_logprob_for_information_profile.py`, set
   `SEQUENCE_LENGTH` and `NUM_REPETITIONS`, then run it once for each desired
   pair. Figure 1(a) uses `(N, M) = (25, 100), (50, 100), (100, 100),
   (200, 100)`.
3. Run `src/plot_profiles_alltogether.py` to generate Figure 1(a).
4. Run `src/plot_optimal_schedules.py` to generate Figure 1(b).
5. Optionally run `src/compute_jensen_gap.py` to print the estimated profile
   range and predicted relative error decrease reported in Section 6.

Model evaluation on CPU is the expensive part. The plotting and schedule
scripts can be rerun directly from the committed files in `data/`.

## Raw-data convention and error bars

Each raw conditional-log-probability file has two columns:

```text
num_masked_tokens log_probability
```

For a fixed number `q` of masked tokens, the file contains `M` consecutive
blocks of `q` values. Each block is one independent repetition of Algorithm 2.
The corresponding paper index is the number of unmasked tokens, `N - q`.

For each repetition, the code first averages its `q` conditional
log-probabilities. The plotted error bar is one standard error across the `M`
independent repetition-level averages:

```text
standard deviation of repetition averages / sqrt(M)
```

The committed raw files can therefore be used to recompute the corrected error
bars without rerunning ModernBERT.
