from datasets import DatasetDict, load_from_disk


def load_ancestry_dataset() -> DatasetDict:
    train_dataset = load_from_disk("/data/evaluation/genomics/Ancestry_all_genome_parent/train")
    test_dataset = load_from_disk("/data/evaluation/genomics/Ancestry_all_genome_parent/test")
    print(train_dataset)
    print(test_dataset)

    return {"train": train_dataset, "test": test_dataset}


if __name__ == "__main__":
    import numpy as np

    dataset_dict = load_ancestry_dataset()
    print(dataset_dict["train"])
    print(dataset_dict["test"])

    unique_indices = np.unique([sample["start_idx"] for sample in dataset_dict["train"]])
    print(unique_indices)
    print()
    for i in range(30):
        sample = dataset_dict["train"][i]
        variants = eval(sample["variants"])
        for variant in variants:
            ref, alt, start, end = variant
            print(variant)
            print(sample["sequence"][start - 1 : end + 1])
            print(len(sample["sequence"]))
            if end > len(sample["sequence"]):
                continue
            assert sample["sequence"][start:end] == alt, (
                sample["sequence"][start - 1 : end + 1] + "|" + alt
            )
