# -*- coding: utf-8 -*-
"""Project-wide configuration shared by the experiment scripts."""

from pathlib import Path


# --------------------------------------------------
# Choice of model
# --------------------------------------------------

MODEL_NAME = "answerdotai/ModernBERT-base"
# MODEL_NAME = "bert-base-uncased"

# --------------------------------------------------
# Total number of sentences
# --------------------------------------------------

NUM_SENTENCES = 5000

# --------------------------------------------------
# Seed
# --------------------------------------------------

RANDOM_SEED = 0

# --------------------------------------------------
# Folder names 
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
