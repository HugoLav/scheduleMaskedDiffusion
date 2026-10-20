# -*- coding: utf-8 -*-
"""Download WikiText-2, tokenize it, and save fixed-length chunks."""

from datasets import load_dataset
from transformers import AutoTokenizer

from global_parameters import DATA_DIR, MODEL_NAME, NUM_SENTENCES


# Length for the data set
SEQUENCE_LENGTHS = (25, 50, 100, 200)


def main():
    """Create the tokenized WikiText-2 files used by the experiments."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving files in: {DATA_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Downloading WikiText-2...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    print("Done.")

# -----------------------------
# Tokenize whole corpus
# -----------------------------
    all_token_ids = []

    for line in dataset["train"]["text"]:
        line = line.strip()

        if not line:
            continue

        token_ids = tokenizer(line, add_special_tokens=False)["input_ids"]
        all_token_ids.extend(token_ids)

    print(f"Total number of tokens: {len(all_token_ids)}")

# -----------------------------
# Cut into fixed-length chunks
# -----------------------------
    for sequence_length in SEQUENCE_LENGTHS:
        output_path = DATA_DIR / f"wikitext_{sequence_length}.txt"

        num_tokens_needed = sequence_length * NUM_SENTENCES

        if len(all_token_ids) < num_tokens_needed:
            print(
                f"Not enough tokens for length {sequence_length}. "
                f"Need {num_tokens_needed}, have {len(all_token_ids)}."
            )
            continue

        with output_path.open("w", encoding="utf-8") as output_file:
            for chunk_index in range(NUM_SENTENCES):
                start = chunk_index * sequence_length
                chunk = all_token_ids[start : start + sequence_length]
                output_file.write(" ".join(str(token_id) for token_id in chunk) + "\n")

        print(
            f"Saved {NUM_SENTENCES} chunks of length {sequence_length} "
            f"to {output_path}"
        )


if __name__ == "__main__":
    main()
