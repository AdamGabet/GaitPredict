"""Entry point for running the MotionBERT training loop without sweeps.

This script instantiates a full Weights & Biases configuration, launches the
standard training loop, and lets that loop handle the final evaluation pass.
"""

import argparse

import wandb
from model.training.default_config import (
    LONG_CONFIG,
    DEFAULT_CONFIG,
    SHORT_CONFIG,
    LIKE_OLD_CONFIG,
    FINETUNE_CONFIG,
    LONG_CONFIG_LOWMEM,
)
from model.training.train_loop import train
from model.training.eval_loop import run_final_evaluation
from model.training.continue_training import keep_training
from dotenv import load_dotenv
import os

load_dotenv(interpolate=True)  # Load environment variables from .env file if present

# Default configuration mirrors the fields consumed inside train_loop.py. Update
# values here to change behaviour without introducing extra CLI wiring.



# Selectable named configs. Defaults to LONG_CONFIG so existing behaviour is unchanged;
# set TRAIN_CONFIG=finetune (env) to run the proprietary-data fine-tune, or
# TRAIN_CONFIG=long_lowmem for the same recipe with gradient checkpointing +
# accumulation (fits in ~5GB instead of ~200+GB, same effective batch/size_seq).
_CONFIGS = {
    "long": LONG_CONFIG,
    "short": SHORT_CONFIG,
    "default": DEFAULT_CONFIG,
    "like_old": LIKE_OLD_CONFIG,
    "finetune": FINETUNE_CONFIG,
    "long_lowmem": LONG_CONFIG_LOWMEM,
}


def build_wandb_config() -> dict:
    """Return a copy of the selected configuration for the training loop."""
    name = os.getenv("TRAIN_CONFIG", "long").lower()
    if name not in _CONFIGS:
        raise ValueError(
            f"Unknown TRAIN_CONFIG={name!r}. Choose one of: {sorted(_CONFIGS)}"
        )
    return dict(_CONFIGS[name])


def main() -> None:
    """Launch a non-sweep training run under wandb tracking."""
    config = build_wandb_config()
    wandb_kwargs = {
        "project": os.getenv("WANDB_PROJECT", "GaitPredict"),
        "config": config,
        "name": config['name'],
    }
    if config['debug_mode']:
        wandb_kwargs["mode"] = "offline"
    wandb_kwargs["entity"] = os.getenv("WANDB_ENTITY", "GaitPredict")


    # Initialise wandb so train_loop can consume wandb.config directly.
    run = wandb.init(**wandb_kwargs)

    try:
        context = train(run_final_eval=True, config=wandb.config)

    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
