"""Main script for biotype classification."""

import os
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pyfaidx import Fasta
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoModel, PreTrainedTokenizer
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))
from models import load_model_and_tokenizer

import wandb

biotypes_of_interest = [
    "protein_coding",
    "lncRNA",
    "processed_pseudogene",
    "unprocessed_pseudogene",
    "snRNA",
    "miRNA",
    "TEC",
    "snoRNA",
    "misc_RNA",
]

@torch.no_grad()
def generate_embedding(
    model: AutoModel, tokenizer: PreTrainedTokenizer, genes_df: pd.DataFrame
) -> list[np.ndarray]:
    """Generate embeddings for the given biotype sequences.

    Args:
        model (AutoModel): The model to use.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        genes_df (pd.DataFrame): Dataframe that contains biotype sequences and their labels.

    Returns:
        list[np.ndarray]: The embeddings for the given genes.

    """
    embeddings_list = []
    for seq in tqdm(genes_df["sequence"], desc="Generating embeddings"):
        tokens = tokenizer.tokenize(seq)
        chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]

        chunk_embeddings = []
        for chunk in chunks:
            input_ids = tokenizer.convert_tokens_to_ids(chunk)
            input_ids = torch.tensor([input_ids]).cuda()

            if model_name == "nt_500m" or model_name == "nt_50m":
                attention_mask = input_ids != tokenizer.pad_token_id
                with torch.amp.autocast("cuda"):
                    output = model(
                        input_ids,
                        attention_mask=attention_mask,
                        encoder_attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                output = output["hidden_states"][-1].cpu()
            elif model_name == "hyenadna" or model_name == "caduceus":
                output = (
                    model(input_ids, output_hidden_states=True).hidden_states[-1].detach().cpu()
                )
            elif model_name == "dnabertv2":
                with torch.amp.autocast("cuda"):
                    output = model(input_ids)[1].detach().cpu()
            elif model_name == "llama" or model_name == "mistral":
                with torch.amp.autocast("cuda"):
                    output = (
                        model(input_ids, output_hidden_states=True).hidden_states[-1].detach().cpu()
                    )
            elif model_name == "genalm":
                with torch.amp.autocast("cuda"):
                    output = model(input_ids).hidden_states[-1]
            chunk_embedding = torch.max(output, dim=1)[0].squeeze().cpu().numpy()
            chunk_embeddings.append(chunk_embedding)

        # average over chunks
        sequence_embedding = np.mean(chunk_embeddings, axis=0)
        embeddings_list.append(sequence_embedding)

    return embeddings_list


if __name__ == "__main__":
    use_wandb = "WANDB_SWEEP_ID" in os.environ

    if use_wandb:
        run = wandb.init()
        model_name = wandb.config.model_name
        pretrained = wandb.config.pretrained
        tokenizer_type = wandb.config.tokenizer
        embedding_dim = wandb.config.embedding_dim
        use_local_data = wandb.config.use_local_data
    else:
        run = None
        model_name = "caduceus"
        pretrained = False
        tokenizer_type = "char"
        embedding_dim = 4096
        use_local_data = False
        print(f"Model Name: {model_name}, Pretrained: {pretrained}, Tokenizer: {tokenizer_type}")

    model, tokenizer, max_length = load_model_and_tokenizer(
        model_name, pretrained, tokenizer_type, embedding_dim
    )

    if use_local_data:
        genes_df = pd.read_csv("../data/biotypes.csv")
    else:
        dataset = load_dataset("m42-health/biotypes", token=True)
        genes_df = dataset["train"].to_pandas()

    print(f"Number of samples: {len(genes_df)}")

    embedding_list = generate_embedding(model, tokenizer, genes_df)

    genes_df["embeddings"] = embedding_list

    X = np.stack(genes_df["embeddings"].values)

    le = LabelEncoder()
    y = le.fit_transform(genes_df["gene_type"])

    print("Unique classes:", le.classes_)
    print("Number of unique classes:", len(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "objective": "multi:softmax",
        "num_class": len(le.classes_),
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "verbosity": 2,
        "device": "gpu",
    }
    xgb_model = xgb.XGBClassifier(**params, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Weighted F1 Score: {f1}")
    if use_wandb:
        run.log({"f1_score": f1})
        run.finish()
