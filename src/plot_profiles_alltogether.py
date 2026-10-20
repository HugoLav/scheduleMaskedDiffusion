# -*- coding: utf-8 -*-
"""Generate Figure 1(a): normalized estimated information profiles."""

import numpy as np
import matplotlib.pyplot as plt

from global_parameters import DATA_DIR, FIGURES_DIR, MODEL_NAME
from routines_estimate_profile import estimate_information_profile


# ============================================================
# Parameters
# ============================================================

EXPERIMENTS = (
    (25, 100),
    (50, 100),
    (100, 100),
    (200, 100),
)


# ============================================================
# Plot
# ============================================================

def create_profile_figure():
    """Create and return the normalized-profile figure and its output path."""
    safe_model_name = MODEL_NAME.replace("/", "-")
    figure, axis = plt.subplots(figsize=(8, 6))

    for sequence_length, num_repetitions in EXPERIMENTS:

        input_path = DATA_DIR / (
            f"conditional_log_probs_{safe_model_name}_"
            f"N{sequence_length}_M{num_repetitions}.txt"
        )

        data = np.loadtxt(input_path)


    # Normalized profile
        fraction_unmasked, mean_profile, standard_error = (
            estimate_information_profile(
                data=data,
                sequence_length=sequence_length,
                num_repetitions=num_repetitions,
                normalize=True,
            )
        )

    
    # Print normalized profile with error bar 

        axis.plot(
            fraction_unmasked,
            mean_profile,
            "-o",
            linewidth=1.5,
            markersize=2.0,
            label=f"N={sequence_length}",
        )

        axis.fill_between(
            fraction_unmasked,
            mean_profile - standard_error,
            mean_profile + standard_error,
            alpha=0.2,
        )

    axis.set_xlabel("Fraction of unmasked tokens")
    axis.set_ylabel("Normalized information profile")
    axis.legend()
    axis.grid(alpha=0.3)


# ============================================================
# Save plot
# ============================================================


    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"normalized_profiles_{safe_model_name}.pdf"
    return figure, output_path


def main():
    """Generate, save, and display Figure 1(a)."""
    figure, output_path = create_profile_figure()
    figure.savefig(output_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
