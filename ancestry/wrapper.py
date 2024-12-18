from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from biotype.utils import load_model_and_tokenizer


def extract_decoder_hidden_states(
    hidden_states: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    """Extract the last hidden states from the decoder model.

    Args:
        hidden_states (torch.Tensor): The hidden states from the decoder model.
        input_ids (torch.Tensor): The input ids from the decoder model.
        pad_token_id (int): The pad token id.

    Returns:
        torch.Tensor: The last hidden states.

    """
    mask = input_ids[:, :-1] != pad_token_id

    # Get the indices of the last True value in each sequence
    last_non_padding_idx = mask.sum(dim=1) - 1
    last_hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_non_padding_idx]

    return last_hidden_states


def is_decoder(model_type: str):  # noqa
    return model_type in ["hyenadna", "llama"]


NUCLEOTIDE_LEN_MAP = {
    "nt_500m": 1000 * 6,
    "nt_50m": 2048 * 6,
    "dnabertv2": 512,
    "hyenadna": 1024,
    "llama": 4096,
    "long_hyenadna": 1024 * 1000,
    "genalm": 512 * 4,  # approximate, because it uses BPE
    "caduceus": 32768,  # more than the 32K sequence length in our dataset
    "mistral": 4096,
}


class AncestrySupervisedWrapper:
    """Wrapper for the Ancestry Supervised Learning."""

    def __init__(
        self: "AncestrySupervisedWrapper",
        model_type: str,
        pretrained: bool,
        tokenizer_type: str,
        pooling: str,
        probing_type: str,
        embedding_dim: int = -1,
        batch_size: int = 32,
    ) -> None:
        self.pooling = pooling
        self.tokenizer_type = tokenizer_type
        self.probing_type = probing_type
        if self.pooling not in ["default", "mean", "max"]:
            raise ValueError(
                f"pooling {self.pooling} is not supported, should be one of default, mean, max"
            )
        self.embedding_dim = embedding_dim

        self.exact_model_type = model_type
        self.pretrained = pretrained
        self.model_type = "decoder" if is_decoder(model_type) else "encoder"
        self.is_hyena = model_type in ["hyenadna", "long_hyenadna"]
        self.model, self.tokenizer, self.max_length = load_model_and_tokenizer(
            model_type, pretrained, tokenizer_type, embedding_dim
        )
        self.model.eval()
        self.batch_size = batch_size

    def chunk_sequences(self: "AncestrySupervisedWrapper", seq: str) -> List[str]:
        """Chunk the sequence into chunks of the length of the model.

        Args:
            seq (str): The sequence to chunk.

        Returns:
            List[str]: The chunks of the sequence.

        """
        chunk_len = NUCLEOTIDE_LEN_MAP[self.exact_model_type]
        chunks = []
        for i in range(0, len(seq), chunk_len):
            chunks.append(seq[i : i + chunk_len])
        return chunks

    def get_embedding(self: "AncestrySupervisedWrapper", batch: List[str]):
        tok_seq = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        # tok_seq = {k: v[:, 1:-1].to(device) for k, v in tok_seq.items()}
        # print(f'tok_seq keys are {list(tok_seq.keys())}')
        # print(tok_seq['token_type_ids'].shape)
        tok_seq = {k: v.to("cuda") for k, v in tok_seq.items() if k != "token_type_ids"}
        if self.is_hyena:
            tok_seq.pop("attention_mask", None)
        output = self.model(**tok_seq, output_hidden_states=True)
        if self.exact_model_type == "dnabertv2":
            hs = output.hidden_states.detach().clone()
        else:
            hs = output.hidden_states[-1].detach().clone()
        del output
        # embedding_idx = -2 if self.model_type == 'decoder' else hs.shape[1] // 2
        if self.model_type == "decoder":
            if self.pooling == "default":
                emb = extract_decoder_hidden_states(
                    hs, tok_seq["input_ids"], self.tokenizer.pad_token_id
                )
            elif self.pooling == "mean":
                emb = hs.mean(dim=1)
            elif self.pooling == "max":
                emb = hs.max(dim=1).values
            emb = emb.float().cpu().numpy()
        else:
            if self.pooling == "default":
                emb = hs[:, 0]
            elif self.pooling == "mean":
                emb = hs.mean(dim=1)
            elif self.pooling == "max":
                emb = hs.max(dim=1).values
            emb = emb.float().cpu().numpy()
        # print(tok_seq['input_ids'][:, embedding_idx])
        return emb

    def get_batch_embedding(
        self: "AncestrySupervisedWrapper", batch: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the embedding for a batch of sequences.

        Args:
            batch (Dict[str, Any]): The batch of sequences.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The embeddings and the labels.

        """
        embeddings = []
        for seq in batch["sequence"]:
            chunked = self.chunk_sequences(seq)
            chunks = self.get_embedding(chunked)
            if self.pooling == "mean" or self.pooling == "default" or self.pooling == "max":
                seq_emb = np.mean(chunks, axis=0)
            embeddings.append(seq_emb.reshape(1, -1))

        return embeddings

    def _embed_dataset(
        self: "AncestrySupervisedWrapper", dataset: Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            dataset,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            prefetch_factor=2,
            batch_size=self.batch_size,
        )
        labels = []
        sequences = []
        with torch.inference_mode(True):
            for _, batch in tqdm(enumerate(loader)):
                embed_sequences = self.get_batch_embedding(batch)
                sequences.extend(embed_sequences)

                labels.extend(batch["label"])

            x = np.concatenate(sequences, axis=0)
            y = np.array(labels)

        return x, y

    def fit_xgboost(
        self: "AncestrySupervisedWrapper", dataset_dict: DatasetDict, return_preds: bool = False
    ) -> Dict[str, float]:
        """Fit the XGBoost classifier.

        Args:
            dataset_dict (DatasetDict): The dataset to fit the classifier on.
            return_preds (bool, optional): Whether to return the predictions. Defaults to False.

        Returns:
            Dict[str, float]: The metrics for the classifier.

        """
        x_train, y_train = self._embed_dataset(dataset_dict["train"])
        print(f"Calculated {x_train.shape} embeddings for train dataset")
        x_test, y_test = self._embed_dataset(dataset_dict["test"])
        print(f"Calculated {x_test.shape} embeddings for test dataset")

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=42
        )
        params = {
            "objective": "multi:softmax",  # Use 'multi:softmax' for softmax objective
            "num_class": len(set(y_train)),  # Number of classes
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "colsample_bytree": 0.5,
            "eval_metric": "mlogloss",  # Evaluation metric for multi-class classification
            "tree_method": "hist",
            "verbosity": 2,
            "device": "gpu",  # Use GPU accelerated algorithm
            "early_stopping_rounds": 100,
        }
        # train_p, val_p, test_p = map(np.mean, [y_train, y_val, y_test])
        # print(f'Prevalence: Train={train_p:.3f}\tVal={val_p:.3f}\tTest={test_p:.3f}')

        model = XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=10)
        print("Fitted xgboost classifier")
        # Predictions
        y_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)
        y_test_pred_proba = model.predict_proba(x_test)
        # Calculate and print the accuracy
        val_pred_p = np.mean(y_pred)
        print(f"Predicted y_val Prevalence is {val_pred_p:.3f}")

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        # auc = roc_auc_score(y_val, y_pred_proba[:, 1])
        print(f"Val Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        auc = roc_auc_score(y_test, y_test_pred_proba, average="weighted", multi_class="ovr")
        if return_preds:
            return {"test_f1_score": f1, "test_accuracy": accuracy, "test_auc": auc}, {
                "y_test_pred": y_test_pred,
                "y_test_pred_proba": y_test_pred_proba,
            }
        else:
            return {"test_f1_score": f1, "test_accuracy": accuracy, "test_auc": auc}

    def fit_linear(
        self: "AncestrySupervisedWrapper", dataset_dict: DatasetDict, return_preds: bool = False
    ) -> Dict[str, Any]:
        """Fit the Logistic Regression classifier.

        Args:
            dataset_dict (DatasetDict): The dataset to fit the classifier on.
            return_preds (bool, optional): Whether to return the predictions. Defaults to False.

        Returns:
            Dict[str, float]: The metrics for the classifier.

        """
        x_train, y_train = self._embed_dataset(dataset_dict["train"])
        x_test, y_test = self._embed_dataset(dataset_dict["test"])
        label_encoder = LabelEncoder()
        # one hot encode y_train
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)
        # y_train = np.eye(len(set(y_train)))[y_train]
        # y_test = np.eye(len(set(y_test)))[y_test]
        model = LogisticRegression(random_state=42, max_iter=3000)
        model.fit(x_train, y_train)
        y_test_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        y_test_pred_proba = model.predict_proba(x_test)
        auc = roc_auc_score(y_test, y_test_pred_proba, average="weighted", multi_class="ovr")
        # auprc = average_precision_score(y_test, y_test_pred_proba, average='weighted')
        print(f"Ancestry Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
        if return_preds:
            return {"test_f1_score": f1, "test_accuracy": accuracy, "test_auc": auc}, {
                "y_test_pred": y_test_pred,
                "y_test_pred_proba": y_test_pred_proba,
            }
        else:
            return {"test_f1_score": f1, "test_accuracy": accuracy, "test_auc": auc}

    def fit_transform(
        self: "AncestrySupervisedWrapper", dataset_dict: DatasetDict
    ) -> Dict[str, Any]:
        """Fit the model and transform the dataset.

        Args:
            dataset_dict (DatasetDict): The dataset to fit the model on.

        Returns:
            Dict[str, Any]: The metrics for the model.

        """
        # start_indices = [1000000, 62964105, 124928211, 186892316, 248956422 - 5e6]
        start_indices = [
            2015011,
            18354991,
            24308808,
            36628720,
            45995594,
            52182164,
            62543311,
            74672986,
            75197358,
            85769129,
            119478211,
        ]
        metrics = []
        for si in start_indices:
            si_train = dataset_dict["train"].filter(
                lambda sample: sample["start_idx"] == si, keep_in_memory=True
            )
            si_test = dataset_dict["test"].filter(
                lambda sample: sample["start_idx"] == si, keep_in_memory=True
            )
            if self.probing_type == "linear":
                m = self.fit_linear({"train": si_train, "test": si_test})
            elif self.probing_type == "xgboost":
                m = self.fit_xgboost({"train": si_train, "test": si_test})
            print(
                f'tokenizer_type={self.tokenizer_type}, model_type={self.exact_model_type}, '
                f'start_idx={si}, accuracy={m["test_accuracy"]:.3f}, f1={m["test_f1_score"]:.3f}, '
                f'auc={m["test_auc"]:.3f}'
            )
            metrics.append(m)

        aggregated = {
            "mean_test_f1_score": np.mean([m["test_f1_score"] for m in metrics]),
            "mean_test_accuracy": np.mean([m["test_accuracy"] for m in metrics]),
            "mean_test_auc": np.mean([m["test_auc"] for m in metrics]),
        }
        # add std to aggregated
        aggregated["std_test_f1_score"] = np.std([m["test_f1_score"] for m in metrics])
        aggregated["std_test_accuracy"] = np.std([m["test_accuracy"] for m in metrics])
        aggregated["std_test_auc"] = np.std([m["test_auc"] for m in metrics])
        return aggregated


if __name__ == "__main__":
    from dataset import load_ancestry_dataset

    dataset_dict = load_ancestry_dataset()
    tokenizer_type = "default"
    model_type = "caduceus"
    pretrained = False
    pooling = "default"
    probing_type = "linear"
    wrapper = AncestrySupervisedWrapper(
        model_type,
        pretrained,
        tokenizer_type,
        pooling,
        probing_type,
        embedding_dim=-1,
        batch_size=32,
    )
    metrics = wrapper.fit_transform(dataset_dict)
    print(metrics)
