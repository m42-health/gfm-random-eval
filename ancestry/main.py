from typing import Any, Dict

import torch

import wandb
from ancestry.dataset import load_ancestry_dataset
from ancestry.wrapper import AncestrySupervisedWrapper


def log_metrics(metrics: Dict[str, Any]):
    print(f"metrics: {metrics}")
    if wandb.run is None:
        pass
    else:
        print(f"logging metrics into {wandb.run.get_url()}")
        wandb.run.log(metrics)


def main():
    run = wandb.init()
    config = wandb.config
    print(config)

    torch.multiprocessing.set_sharing_strategy("file_system")
    model_name = config.model_name
    tokenizer_type = config.tokenizer
    pretrained = config.pretrained
    probing_type = config.probing_type
    pooling = config.pooling

    model = AncestrySupervisedWrapper(model_name, pretrained, tokenizer_type, pooling, probing_type)
    print("model loaded")

    dataset = load_ancestry_dataset()
    print(f"Dataset {dataset} loaded")

    metrics = model.fit_transform(dataset)
    print("metrics calculated")
    log_metrics(metrics)


if __name__ == "__main__":
    main()
