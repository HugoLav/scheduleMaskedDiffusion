# -*- coding: utf-8 -*-
"""Estimate D(pi) and the predicted gain from schedule optimization."""

import numpy as np

from global_parameters import DATA_DIR, MODEL_NAME
from routines_estimate_profile import (
    estimate_information_profile,
    smooth_profile_and_derivative,
)


SEQUENCE_LENGTH = 200
NUM_REPETITIONS = 100
SMOOTHING_BANDWIDTH = 4.0  # Measured in token-grid points.


def main():
    """Load the estimated profile and print the Section 6 summaries."""
    safe_model_name = MODEL_NAME.replace("/", "-")
    input_path = DATA_DIR / (
        f"conditional_log_probs_{safe_model_name}_"
        f"N{SEQUENCE_LENGTH}_M{NUM_REPETITIONS}.txt"
    )
    data = np.loadtxt(input_path)

    _, raw_profile, _ = estimate_information_profile(
        data=data,
        sequence_length=SEQUENCE_LENGTH,
        num_repetitions=NUM_REPETITIONS,
        normalize=False,
    )

    profile_range = np.max(raw_profile) - np.min(raw_profile)
    print(f"D(pi) = {profile_range}")

    _, profile_increment = smooth_profile_and_derivative(
        profile_values=raw_profile,
        sequence_length=SEQUENCE_LENGTH,
        bandwidth_tokens=SMOOTHING_BANDWIDTH,
    )

    relative_bound = (
        np.sum(np.sqrt(profile_increment)) ** 2 / SEQUENCE_LENGTH
    )
    print(f"Predicted relative error decrease = {1 - relative_bound}")


if __name__ == "__main__":
    main()
