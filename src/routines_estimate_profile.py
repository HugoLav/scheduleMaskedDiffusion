# -*- coding: utf-8 -*-
"""Estimate, normalize, and smooth the information profile."""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def estimate_information_profile(
    data,
    sequence_length,
    num_repetitions,
    normalize=False,
):
    
    """
    Estimate the information profile from a collection of (i, log p) samples.

    Parameters
    ----------
    data : ndarray of shape (n_samples, 2)
        First column: number of masked tokens.
        Second column: log probabilities.
    sequence_length : int
        Sequence length.
    num_repetitions : int
        Number of independent Monte Carlo repetitions at each masking level.
    normalize : bool, default=False
        If True, rescale the profile to lie in [0,1].

    Returns
    -------
    unmasked_grid : ndarray
        Number of unmasked tokens, or their fraction if ``normalize=True``.
    mean_profile : ndarray
        Estimated information profile.
    standard_error : ndarray
        Standard error across independent repetition-level averages.
    """

    masked_counts = data[:, 0].astype(int)
    log_probs = data[:, 1]

    if num_repetitions < 2:
        raise ValueError("num_repetitions must be at least 2.")

    mean_profile = np.empty(sequence_length)
    standard_error = np.empty(sequence_length)

    # At masking level q, each of the M repetitions contributes q contiguous
    # token-level values. Algorithm 2 first averages within each repetition.
    for num_masked_tokens in range(1, sequence_length + 1):
        values = log_probs[masked_counts == num_masked_tokens]
        expected_num_values = num_repetitions * num_masked_tokens

        if len(values) != expected_num_values:
            raise ValueError(
                f"Masking level {num_masked_tokens} contains {len(values)} "
                f"values; expected M * num_masked_tokens = {expected_num_values}."
            )

        repetition_means = values.reshape(
            num_repetitions,
            num_masked_tokens,
        ).mean(axis=1)

        mean_profile[num_masked_tokens - 1] = repetition_means.mean()
        standard_error[num_masked_tokens - 1] = (
            repetition_means.std(ddof=1) / np.sqrt(num_repetitions)
        )

    
    # Reorder from all tokens masked to no tokens masked.
    num_unmasked_tokens = sequence_length - np.arange(1, sequence_length + 1)

    order = np.argsort(num_unmasked_tokens)

    num_unmasked_tokens = num_unmasked_tokens[order]
    mean_profile = mean_profile[order]
    standard_error = standard_error[order]
    
    # Normalize both axes when requested for Figure 1(a).

    if normalize:
        profile_min = mean_profile.min()
        profile_max = mean_profile.max()
        profile_range = profile_max - profile_min

        if profile_range <= 0:
            raise ValueError("Cannot normalize a constant information profile.")

        mean_profile = (mean_profile - profile_min) / profile_range
        standard_error = standard_error / profile_range
        num_unmasked_tokens = num_unmasked_tokens / sequence_length

    return num_unmasked_tokens, mean_profile, standard_error



def smooth_profile_and_derivative(
    profile_values,
    sequence_length,
    bandwidth_tokens,
):
    """
    Compute the discrete derivative of an information profile, smooth it,
    enforce monotonicity, and reconstruct the normalized profile.

    Parameters
    ----------
    profile_values : array-like of shape (N,)
        Profile values ordered by increasing number of unmasked tokens.
          
    sequence_length : int
        Number of profile values.

    bandwidth_tokens : float
        Standard deviation of the Gaussian smoothing kernel, measured in
        numbers of token-grid points.

    Returns
    -------
    smoothed_profile : np.ndarray of shape (N,)
        Monotone profile reconstructed from the smoothed increments.
        It starts at 0 and ends at 1.
    profile_increment : np.ndarray of shape (N - 1,)
        Nonnegative discrete derivative normalized to sum to one.
    """

    profile_values = np.asarray(profile_values, dtype=float)

    if profile_values.ndim != 1:
        raise ValueError("profile_values must be one-dimensional.")

    if len(profile_values) != sequence_length:
        raise ValueError(
            f"Expected profile_values to have length N={sequence_length}, "
            f"but received length {len(profile_values)}."
        )

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")

    if bandwidth_tokens <= 0:
        raise ValueError("bandwidth_tokens must be positive.")

    if not np.all(np.isfinite(profile_values)):
        raise ValueError("profile_values contains NaN or infinite values.")

    # ========================================================
    # Compute the discrete derivative
    # ========================================================

    profile_increment = np.diff(profile_values)
    
    # ========================================================
    # Smooth the derivative of the profile
    # ========================================================
    
    profile_increment = gaussian_filter1d(
        profile_increment,
        sigma=bandwidth_tokens,
        mode="nearest",
    )

    # Clamp only after smoothing to enforce nonnegative profile increments.
    profile_increment = np.maximum(profile_increment, 0.0)

    # ========================================================
    # Normalize so that sum_j Delta f_j = 1
    # ========================================================

    total_increment = np.sum(profile_increment)

    if total_increment <= 0:
        raise ValueError(
            "The nonnegative discrete derivative has zero total increment."
        )

    profile_increment = profile_increment / total_increment

    # ========================================================
    # Reconstruct the monotone profile by cumulative summation
    # ========================================================

    smoothed_profile = np.concatenate(
        ([0.0], np.cumsum(profile_increment))
    )

    return smoothed_profile, profile_increment


