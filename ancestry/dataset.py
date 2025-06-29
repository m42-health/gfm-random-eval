from datasets import DatasetDict, load_dataset


def load_ancestry_dataset(dataset_path: str = "m42-health/ancestry_dataset") -> DatasetDict:
    """Load ancestry dataset from huggingface.
    
    Args:
        dataset_path (str): Path to the dataset on huggingface.
    
    Returns:
        DatasetDict: The loaded dataset.
    """
    return load_dataset(dataset_path)


if __name__ == "__main__":
    dataset_dict = load_ancestry_dataset()
    print(dataset_dict["train"])
    print(dataset_dict["test"])
    print(dataset_dict["train"][0])
