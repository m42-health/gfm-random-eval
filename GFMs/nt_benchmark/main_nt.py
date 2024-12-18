"""Main script for the NT benchmark."""

import os
import sys

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from config import Config  # noqa: E402
from train_utils import finetune  # noqa: E402

if __name__ == "__main__":
    use_wandb = "WANDB_SWEEP_ID" in os.environ
    config = Config(
        learning_rate=5e-5,
        model_type="mistral_max_pool",
        weight_init="random",
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        fold_number=0,
        dataset_name="H3K4me3",
        # dataset_base_path="/data/evaluation/genomics/",
    )

    run = None
    if use_wandb:
        run = wandb.init()
        config = wandb.config
        print(config)
        print(wandb.config)

    finetune(config, run)
