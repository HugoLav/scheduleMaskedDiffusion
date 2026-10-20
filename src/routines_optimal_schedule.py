# -*- coding: utf-8 -*-
"""Compute exact and asymptotically optimal unmasking schedules."""

import numpy as np


def compute_optimal_schedule(profile, num_steps):
    """
    Solve

        min_a  sum_{i=0}^{N-1} f(i)
               - sum_{k=0}^{K-1} (a_{k+1} - a_k) f(a_k)

    subject to

        0 = a_0 < a_1 < ... < a_K = N,

    where all a_k are integers.

    Parameters
    ----------
    profile : array-like of shape (N,)
        Values f(0), ..., f(N-1).

    num_steps : int
        Number of intervals. The returned schedule contains K+1 points:

            a_0, a_1, ..., a_K.

    Returns
    -------
    schedule : np.ndarray of shape (K+1,)
        Optimal integer schedule:

            [0, a_1, ..., a_{K-1}, N].

    objective_value : float
        Minimum value of A(schedule).

    """

    profile = np.asarray(profile, dtype=float)

    if profile.ndim != 1:
        raise ValueError("profile must be a one-dimensional array.")

    if not np.all(np.isfinite(profile)):
        raise ValueError("profile contains NaN or infinite values.")

    sequence_length = len(profile)

    if sequence_length == 0:
        raise ValueError("profile must contain at least one value.")

    if not isinstance(num_steps, (int, np.integer)):
        raise TypeError("num_steps must be an integer.")

    if num_steps < 1 or num_steps > sequence_length:
        raise ValueError(
            f"num_steps must satisfy 1 <= num_steps <= N={sequence_length}."
        )

    # dp[k, i] is the largest reward achievable using exactly k
    # intervals, starting at 0 and ending at i.
    dynamic_program = np.full(
        (num_steps + 1, sequence_length + 1),
        -np.inf,
    )

    # predecessor[k, i] stores the starting point j of the last
    # interval in an optimal path ending at i.
    predecessor = np.full(
        (num_steps + 1, sequence_length + 1),
        -1,
        dtype=int,
    )

    dynamic_program[0, 0] = 0.0

    for step in range(1, num_steps + 1):

        # At least k positions are needed to form k nonempty intervals.
        for end in range(step, sequence_length + 1):

            best_reward = -np.inf
            best_start = -1

            # The preceding k-1 intervals must end at start.
            # We require start < end so the final interval is nonempty.
            for start in range(step - 1, end):

                if not np.isfinite(dynamic_program[step - 1, start]):
                    continue

                interval_reward = (end - start) * profile[start]

                candidate = (
                    dynamic_program[step - 1, start]
                    + interval_reward
                )

                if candidate > best_reward:
                    best_reward = candidate
                    best_start = start

            dynamic_program[step, end] = best_reward
            predecessor[step, end] = best_start

    # Reconstruct the optimal schedule backward from a_K = N.
    schedule = np.empty(num_steps + 1, dtype=int)
    schedule[num_steps] = sequence_length

    current_end = sequence_length

    for step in range(num_steps, 0, -1):
        current_start = predecessor[step, current_end]

        if current_start < 0:
            raise RuntimeError("Failed to reconstruct the optimal schedule.")

        schedule[step - 1] = current_start
        current_end = current_start

    objective_value = np.sum(profile) - dynamic_program[num_steps, sequence_length]

    return schedule, objective_value


def compute_asymptotic_schedule(
    profile_increment,
    sequence_length,
    num_steps,
):
    """
    Construct the schedule defined by quantiles of sqrt(profile_increment).

    Parameters
    ----------
    profile_increment : array-like of shape (N - 1,)
        Estimated discrete derivative of the information profile.
        Its entries should be nonnegative.

    sequence_length : int
        Final schedule position. The schedule begins at 0 and ends at N.

    num_steps : int
        Number of schedule intervals. The output contains K + 1 points.

    Returns
    -------
    schedule : np.ndarray of shape (K + 1,)
        Schedule (a_0, ..., a_K), with a_0 = 0 and a_K = N.
    """

    profile_increment = np.asarray(profile_increment, dtype=float)

    if profile_increment.ndim != 1:
        raise ValueError("profile_increment must be one-dimensional.")

    if len(profile_increment) != sequence_length - 1:
        raise ValueError(
            f"profile_increment must have length N - 1 = "
            f"{sequence_length - 1}, but has length {len(profile_increment)}."
        )

    if not isinstance(num_steps, (int, np.integer)):
        raise TypeError("num_steps must be an integer.")

    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")

    if num_steps > sequence_length:
        raise ValueError(
            "num_steps cannot exceed sequence_length for an increasing schedule."
        )

    if not np.all(np.isfinite(profile_increment)):
        raise ValueError("profile_increment contains NaN or infinite values.")

    if np.any(profile_increment < 0):
        raise ValueError("profile_increment must be nonnegative.")

    weights = np.sqrt(profile_increment)
    total_mass = np.sum(weights)

    if total_mass <= 0:
        raise ValueError("sqrt(profile_increment) has zero total mass.")

    cumulative_mass = np.cumsum(weights)

    schedule = np.empty(num_steps + 1, dtype=int)
    schedule[0] = 0
    schedule[num_steps] = sequence_length

    for step in range(1, num_steps):
        threshold = (step / num_steps) * total_mass

        # searchsorted returns an index in {0, ..., N-2}.
        # Add 1 because cumulative_mass[j] sums increments 0,...,j,
        # corresponding to schedule position n = j + 1.
        schedule[step] = np.searchsorted(
            cumulative_mass,
            threshold,
            side="left",
        ) + 1

    return schedule
