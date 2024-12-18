"""Implements the training logic for NT Benchmark."""

import numpy as np
from config import Config
from datasets import Dataset
from ft_datasets import get_datasets, nt_benchmarks
from models import load_model_tokenizer
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import Trainer, TrainingArguments

import wandb


def compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray], dataset_name: str
) -> dict[str, float]:
    """Compute the metrics for the evaluation.

    Args:
        eval_pred (tuple[np.ndarray, np.ndarray]): The evaluation predictions.
        dataset_name (str): The name of the dataset.

    Returns:
        dict[str, float]: The evaluation results.

    """
    logits, labels = eval_pred

    # Handle DNABERTv2 output
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    metric = nt_benchmarks[dataset_name]["metric"]
    if metric == "mcc":
        score = matthews_corrcoef(labels, predictions)
    elif metric == "f1_score":
        score = f1_score(labels, predictions, average="macro")
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return {metric: score}


class CustomTrainer(Trainer):
    """Custom trainer to save the best model and evaluate on the test set."""

    def __init__(self: "CustomTrainer", dataset_name: str, *args: object, **kwargs: object) -> None:
        """Initialize the custom trainer.

        Args:
            dataset_name (str): The name of the dataset.
            *args (object): The arguments for the trainer.
            **kwargs (object): The keyword arguments for the trainer.

        """
        super().__init__(*args, **kwargs)
        self.best_metric = float("-inf")
        self.best_model = None
        self.dataset_name = dataset_name
        self.best_test_metric = None

    def evaluate(
        self: "CustomTrainer",
        eval_dataset: Dataset | None = None,  # noqa
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",  # noqa
    ) -> dict[str, float]:
        """Evaluate the model on the validation and test sets.

        Args:
            eval_dataset (Dataset | None): The evaluation dataset.
            ignore_keys (list[str] | None): The keys to ignore.
            metric_key_prefix (str): The prefix for the metric key.

        Returns:
            dict[str, float]: The evaluation results.

        """
        val_results = super().evaluate(
            self.eval_dataset["val"], ignore_keys, metric_key_prefix="val"
        )

        test_results = super().evaluate(
            self.eval_dataset["test"], ignore_keys, metric_key_prefix="test"
        )

        combined_results = {**val_results, **test_results}
        metric = nt_benchmarks[self.dataset_name]["metric"]
        val_metric = combined_results[f"val_{metric}"]
        test_metric = combined_results[f"test_{metric}"]

        if val_metric > self.best_metric:
            self.best_metric = val_metric
            self.best_test_metric = test_metric

        return combined_results


def finetune(config: Config, run: wandb.run) -> None:
    """Entry point for finetuning the model.

    Args:
        config (Config): The configuration for the experiment.
        run (wandb.run): The wandb run object to log the results.

    Returns:
        None

    """
    config.num_classes = nt_benchmarks[config.dataset_name]["num_labels"]
    model, tokenizer = load_model_tokenizer(config)

    train_dataset, validation_dataset, test_dataset = get_datasets(config, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./results/{config.dataset_name}",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=f"./logs/{config.dataset_name}",
        report_to="wandb" if run else "none",
        bf16="hyena" not in config.model_type.lower(),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        evaluation_strategy="epoch",
        logging_strategy="no",
        save_strategy="no",
        disable_tqdm=False,
    )

    trainer = CustomTrainer(
        dataset_name=config.dataset_name,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"val": validation_dataset, "test": test_dataset},
        data_collator=None,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, config.dataset_name),
    )

    trainer.train()

    metric = nt_benchmarks[config.dataset_name]["metric"]
    print("\nBest model performance:")
    print(f"Best validation {metric}: {round(trainer.best_metric, 3)}")
    print(f"Corresponding test {metric}: {round(trainer.best_test_metric, 3)}")
    if run:
        # names for the final metrics in wandb
        # final metric on test is called either "ms_test_mcc" or "ms_test_f1_score"
        # depending on the dataset
        run.log(
            {
                f"ms_val_{metric}": round(trainer.best_metric, 3),
                f"ms_test_{metric}": round(trainer.best_test_metric, 3),
            }
        )
