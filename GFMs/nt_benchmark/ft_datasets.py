"""Implementation of the dataset creation logic for the NT Benchmark."""

from config import Config
from datasets import Dataset, load_dataset
from sklearn.model_selection import KFold
from transformers import PreTrainedTokenizer

nt_benchmarks = {
    "enhancers": {"num_labels": 2, "metric": "mcc"},
    "enhancers_types": {"num_labels": 3, "metric": "mcc"},
    "H3": {"num_labels": 2, "metric": "mcc"},
    "H3K14ac": {"num_labels": 2, "metric": "mcc"},
    "H3K36me3": {"num_labels": 2, "metric": "mcc"},
    "H3K4me1": {"num_labels": 2, "metric": "mcc"},
    "H3K4me2": {"num_labels": 2, "metric": "mcc"},
    "H3K4me3": {"num_labels": 2, "metric": "mcc"},
    "H3K79me3": {"num_labels": 2, "metric": "mcc"},
    "H3K9ac": {"num_labels": 2, "metric": "mcc"},
    "H4": {"num_labels": 2, "metric": "mcc"},
    "H4ac": {"num_labels": 2, "metric": "mcc"},
    "promoter_all": {"num_labels": 2, "metric": "f1_score"},
    "promoter_no_tata": {"num_labels": 2, "metric": "f1_score"},
    "promoter_tata": {"num_labels": 2, "metric": "f1_score"},
    "splice_sites_acceptors": {"num_labels": 2, "metric": "f1_score"},
    "splice_sites_all": {"num_labels": 3, "metric": "f1_score"},
    "splice_sites_donors": {"num_labels": 2, "metric": "f1_score"},
}


def get_datasets(
    config: Config, tokenizer: PreTrainedTokenizer
) -> tuple[Dataset, Dataset, Dataset]:
    """Get and prepare datasets for training and evaluation.

    Args:
        config: Configuration object containing dataset settings.
        tokenizer: Tokenizer to do pretokenization.

    Returns:
        tuple[Dataset, Dataset, Dataset]: Tokenized train, validation, and test datasets.

    """
    # train_dataset = load_from_disk(
    #     os.path.join(config.dataset_base_path, "NT_Benchmarks_" + config.dataset_name, "train")
    # )
    # test_dataset = load_from_disk(
    #     os.path.join(config.dataset_base_path, "NT_Benchmarks_" + config.dataset_name, "test")
    # )

    dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks", config.dataset_name
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if "seq" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("seq", "sequence")
        test_dataset = test_dataset.rename_column("seq", "sequence")
    elif "name" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("name")
        test_dataset = test_dataset.remove_columns("name")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    ds_folds = []
    for train_index, val_index in kf.split(train_dataset):
        ds_train_fold = train_dataset.select(train_index)
        ds_val_fold = train_dataset.select(val_index)
        ds_folds.append((ds_train_fold, ds_val_fold, train_index, val_index))

    ds_train, ds_validation, train_index, val_index = ds_folds[config.fold_number]

    print(train_index, val_index, len(train_index), len(val_index))

    ds_test = test_dataset

    return tokenize_datasets(tokenizer, ds_train, ds_validation, ds_test)


def tokenize_fn(tokenizer: PreTrainedTokenizer, samples: dict) -> dict:
    """Tokenize the sequences.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer.
        samples (dict): The samples to tokenize.

    Returns:
        dict: The tokenized samples.

    """
    outputs = tokenizer(samples["sequence"])
    outputs.pop("token_type_ids", None)
    return outputs


def tokenize_datasets(
    tokenizer: PreTrainedTokenizer, ds_train: Dataset, ds_validation: Dataset, ds_test: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
    """Tokenize the datasets.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer.
        ds_train (Dataset): The training dataset.
        ds_validation (Dataset): The validation dataset.
        ds_test (Dataset): The test dataset.

    Returns:
        tuple[Dataset, Dataset, Dataset]: The tokenized datasets.

    """
    ds_train = ds_train.map(
        lambda examples: tokenize_fn(tokenizer, examples),
        batched=True,
        desc="Tokenizing training data",
    )
    ds_validation = ds_validation.map(
        lambda examples: tokenize_fn(tokenizer, examples),
        batched=True,
        desc="Tokenizing validation data",
    )
    ds_test = ds_test.map(
        lambda examples: tokenize_fn(tokenizer, examples),
        batched=True,
        desc="Tokenizing test data",
    )

    return ds_train, ds_validation, ds_test
