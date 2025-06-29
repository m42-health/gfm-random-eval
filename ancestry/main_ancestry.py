import os

import torch

import wandb
from ancestry.dataset import load_ancestry_dataset
from ancestry.wrapper import AncestrySupervisedWrapper


def main():
    use_wandb = "WANDB_SWEEP_ID" in os.environ
    run = None
    if use_wandb:
        run = wandb.init()
        config = wandb.config
        print(config)
        model_name = config.model_name
        tokenizer_type = config.tokenizer
        pretrained = config.pretrained
        probing_type = config.probing_type
        pooling = config.pooling
    else:
        model_name = "hyenadna"
        tokenizer_type = "default"
        pretrained = True
        probing_type = "xgboost"
        pooling = "default"
        print(
            f"Running without wandb with parameters: model_name={model_name}, "
            f"tokenizer_type={tokenizer_type}, pretrained={pretrained}, "
            f"probing_type={probing_type}, pooling={pooling}"
        )

    torch.multiprocessing.set_sharing_strategy("file_system")

    model = AncestrySupervisedWrapper(model_name, pretrained, tokenizer_type, pooling, probing_type)
    print("model loaded")

    dataset = load_ancestry_dataset()
    print(f"Dataset {dataset} loaded")

    metrics = model.fit_transform(dataset)
    print("metrics calculated")
    print(f"metrics: {metrics}")
    if run is not None:
        print(f"logging metrics into {run.get_url()}")
        run.log(metrics)


if __name__ == "__main__":
    main()
