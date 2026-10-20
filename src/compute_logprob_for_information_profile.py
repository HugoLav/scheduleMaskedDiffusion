# -*- coding: utf-8 -*-
"""
For each number q = 1, ..., N of masked tokens:
    - repeat M times:
        * choose one length-N tokenized chunk at random,
        * mask q random positions,
        * compute the q log-probabilities of the true masked tokens.

The output file has two columns:
    num_masked_tokens, log_probability

Rows for each masking level are stored as M consecutive repetition blocks.
The paper's index is the number N - q of unmasked tokens.
"""

import time
import random
import numpy as np
import torch

from global_parameters import DATA_DIR, MODEL_NAME, RANDOM_SEED
from routines_compute_probabilities import (
    load_model_and_tokenizer,
    load_token_sequences,
    repeat_masked_token_experiment,
)


# ============================================================
# Parameters: choose these manually
# ============================================================

SEQUENCE_LENGTH = 200  # Choose 25, 50, 100, or 200.
NUM_REPETITIONS = 100


def main():
    """Run Algorithm 2 for every masking level and save the raw values."""
    # Set seeds.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load data and model.
    print(
        f"Running the information-profile experiment with "
        f"N={SEQUENCE_LENGTH}, M={NUM_REPETITIONS}"
    )
    print(f"Model: {MODEL_NAME}")

    start_time = time.perf_counter()

    token_sequences = load_token_sequences(
        sequence_length=SEQUENCE_LENGTH,
        data_dir=DATA_DIR,
    )
    print(
        f"Loaded {len(token_sequences)} tokenized chunks "
        f"of length {SEQUENCE_LENGTH}."
    )

    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

    # Run the experiment.
    result_rows = []

    for num_masked_tokens in range(1, SEQUENCE_LENGTH + 1):
        print(
            f"Processing {num_masked_tokens}/{SEQUENCE_LENGTH} "
            "masked tokens"
        )

        log_probabilities, _ = repeat_masked_token_experiment(
            num_masked_tokens=num_masked_tokens,
            num_repetitions=NUM_REPETITIONS,
            token_sequences=token_sequences,
            tokenizer=tokenizer,
            model=model,
        )

        for log_probability in log_probabilities:
            result_rows.append([num_masked_tokens, log_probability])

    results = np.asarray(result_rows)

    elapsed_time = time.perf_counter() - start_time

    print("Finished.")
    print(f"Computations took {elapsed_time:.3f} seconds.")
    print("Output array shape:", results.shape)

    # Save results.
    safe_model_name = MODEL_NAME.replace("/", "-")
    output_path = DATA_DIR / (
        f"conditional_log_probs_{safe_model_name}_"
        f"N{SEQUENCE_LENGTH}_M{NUM_REPETITIONS}.txt"
    )

    np.savetxt(
        output_path,
        results,
        fmt=["%d", "%.18e"],
        header="num_masked_tokens log_probability",
    )

    print(f"Saved log-probabilities to: {output_path}")


if __name__ == "__main__":
    main()
