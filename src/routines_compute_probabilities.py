# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:54:02 2026

@author: Lavenant

Utilities for estimating conditional log-probabilities from tokenized text.

For a fixed number of masked tokens:
    - choose one sentence/chunk at random,
    - mask i positions uniformly at random,
    - run the masked language model once,
    - return the i log-probabilities assigned to the true masked tokens.
"""

import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_token_sequences(sequence_length, data_dir):
    """
    Load tokenized chunks of length N.

    The expected filename is:
        wikitext_<sequence_length>.txt

    Each line should contain exactly N token IDs separated by spaces.
    """
    input_path = data_dir / f"wikitext_{sequence_length}.txt"

    token_sequences = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            token_ids = [int(value) for value in line.split()]
            if len(token_ids) != sequence_length:
                raise ValueError(
                    f"{input_path} line {line_number} contains {len(token_ids)} "
                    f"tokens; expected {sequence_length}."
                )
            token_sequences.append(token_ids)

    if not token_sequences:
        raise ValueError(f"No token sequences found in {input_path}.")

    return token_sequences



def load_model_and_tokenizer(model_name):
    """
    Load tokenizer and masked language model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model



def compute_masked_token_log_probabilities(
    num_masked_tokens,
    token_sequences,
    tokenizer,
    model,
):
    """
    For a fixed i:
        - choose one sentence at random,
        - choose i random positions to mask,
        - compute the log-probabilities of the true masked tokens.

    Parameters
    ----------
    num_masked_tokens : int
        Number of tokens to mask.
    token_sequences : list[list[int]]
        Tokenized sentences/chunks, each of length N.
    tokenizer : transformers tokenizer
        Tokenizer associated with the model.
    model : transformers masked language model
        Model used to compute conditional probabilities.

    Returns
    -------
    log_probs : np.ndarray
        Array of length i. Each entry is log p_theta(x_j | unmasked tokens)
        for one masked position j.
    masked_positions : list[int]
        Positions that were masked, in the same order as log_probs.
    """
    
    original_token_ids = random.choice(token_sequences)
    sequence_length = len(original_token_ids)

    if num_masked_tokens < 1 or num_masked_tokens > sequence_length:
        raise ValueError(
            "num_masked_tokens must satisfy "
            f"1 <= num_masked_tokens <= {sequence_length}. "
            f"Got {num_masked_tokens}."
        )

    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if mask_id is None or cls_id is None or sep_id is None:
        raise ValueError("The tokenizer must define mask, CLS, and SEP token IDs.")

    masked_positions = random.sample(
        range(sequence_length),
        num_masked_tokens,
    )

    # Build input: [CLS] + original tokens + [SEP]
    input_ids = torch.tensor([[cls_id] + original_token_ids + [sep_id]])

    # Mask the selected positions. Add 1 because of [CLS].
    for pos in masked_positions:
        input_ids[0, pos + 1] = mask_id

    log_probabilities = np.empty(num_masked_tokens)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]
        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)

        for k, pos in enumerate(masked_positions):
            bert_pos = pos + 1
            true_token_id = original_token_ids[pos]
            log_probabilities[k] = log_softmax[bert_pos, true_token_id].item()

    return log_probabilities, masked_positions



def repeat_masked_token_experiment(
    num_masked_tokens,
    num_repetitions,
    token_sequences,
    tokenizer,
    model,
):
    """
    Repeat one fixed masking-level experiment independently.

    Returns
    -------
    log_probabilities : np.ndarray
        Flat array containing one contiguous block per repetition.
    masked_positions_by_repetition : list[list[int]]
        The masked positions used for each repetition.
    """
    
    log_probabilities = []
    masked_positions_by_repetition = []

    for _ in range(num_repetitions):
        repetition_log_probabilities, masked_positions = (
            compute_masked_token_log_probabilities(
                num_masked_tokens=num_masked_tokens,
                token_sequences=token_sequences,
                tokenizer=tokenizer,
                model=model,
            )
        )

        log_probabilities.extend(repetition_log_probabilities)
        masked_positions_by_repetition.append(masked_positions)

    return np.asarray(log_probabilities), masked_positions_by_repetition


