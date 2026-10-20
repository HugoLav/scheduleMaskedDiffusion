# -*- coding: utf-8 -*-
"""Generate Figure 1(b): exact and asymptotic optimal schedules."""

import numpy as np
import matplotlib.pyplot as plt

from global_parameters import DATA_DIR, FIGURES_DIR, MODEL_NAME
from routines_estimate_profile import (
    estimate_information_profile,
    smooth_profile_and_derivative,
)
from routines_optimal_schedule import (
    compute_asymptotic_schedule,
    compute_optimal_schedule,
)

# ============================================================
# Parameters
# ============================================================

# Which data to load

SEQUENCE_LENGTH = 200
NUM_REPETITIONS = 100

# Number of computational steps 

NUM_STEPS = 10

# Bandwith fro the smoohting (in number of tokens)
SMOOTHING_BANDWIDTH = 4.0



# ============================================================
# Load the data 
# ============================================================

def create_schedule_figure():
    """Create and return the schedule figure and its output path."""
    safe_model_name = MODEL_NAME.replace("/", "-")
    input_path = DATA_DIR / (
        f"conditional_log_probs_{safe_model_name}_"
        f"N{SEQUENCE_LENGTH}_M{NUM_REPETITIONS}.txt"
    )
    data = np.loadtxt(input_path)


# ============================================================
# Estimate the profile and its derivative
# ============================================================

# Raw profile from the data 
    _, raw_profile, _ = estimate_information_profile(
        data=data,
        sequence_length=SEQUENCE_LENGTH,
        num_repetitions=NUM_REPETITIONS,
        normalize=False,
    )

# Smoothing and computing the derivatives
    smoothed_profile, profile_increment = smooth_profile_and_derivative(
        profile_values=raw_profile,
        sequence_length=SEQUENCE_LENGTH,
        bandwidth_tokens=SMOOTHING_BANDWIDTH,
    )


# ============================================================
# Compute the schedule
# ============================================================

# Compute the schedules via the two methods
    exact_schedule, _ = compute_optimal_schedule(
        profile=smoothed_profile,
        num_steps=NUM_STEPS,
    )
    asymptotic_schedule = compute_asymptotic_schedule(
        profile_increment=profile_increment,
        sequence_length=SEQUENCE_LENGTH,
        num_steps=NUM_STEPS,
    )

# Obtain s_k the number of tokens per iteration
    exact_step_sizes = np.diff(exact_schedule)
    asymptotic_step_sizes = np.diff(asymptotic_schedule)


# ============================================================
# Plot the schedules
# ============================================================


    step_numbers = np.arange(1, NUM_STEPS + 1)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        step_numbers,
        exact_step_sizes,
        "o-",
        markersize=4,
        linewidth=1.5,
        label="Exact",
    )
    axis.plot(
        step_numbers,
        asymptotic_step_sizes,
        "o-",
        markersize=4,
        linewidth=1.5,
        label="Asymptotic",
    )
    axis.set_xlabel(r"Computational step $k$")
    axis.set_ylabel(r"Number of tokens to unmask $s_k$")
    axis.legend()
    axis.grid(alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / (
        f"optimal_schedules_N{SEQUENCE_LENGTH}_K{NUM_STEPS}.pdf"
    )
    return figure, output_path


def main():
    """Generate, save, and display Figure 1(b)."""
    figure, output_path = create_schedule_figure()
    figure.savefig(output_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()

