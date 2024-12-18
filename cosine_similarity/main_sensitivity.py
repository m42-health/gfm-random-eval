"""Conducts sensitivity experiments on the TP53 using various pretrained models."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics.pairwise import cosine_similarity

from cosine_similarity.models import load_model_and_tokenizer
from cosine_similarity.utils import (
    MODEL_NAMES,
    mutate_n_nucleotides,
    process_single_sequence,
    sample_sequences_from_genome,
    supported_pooling,
)


@torch.no_grad()
def compute_sequence_embeddings(
    ref_sequence: str,
    mutated_sequences: list[str],
    model_name: str,
    max_len: int,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
) -> list[dict[str, np.ndarray]]:
    """Compute embeddings for the reference and mutated sequences.

    Sequence first gets chunked based on max_len that is model specific.
    Then for each chunk, embeddings are computed using the model.
    The embeddings are then pooled using the specified pooling type.
    The average of the embeddings is computed and returned.
    The same is repeated for multiple mutated sequences.

    Args:
        ref_sequence: a string containing the reference sequence.
        mutated_sequences: a list of strings containing the mutated sequences.
        model_name: a string containing the name of the model.
        max_len: an integer specifying the maximum sequence length.
        model: a torch.nn.Module model.
        tokenizer: a transformers.PreTrainedTokenizer tokenizer.

    Returns:
        a list of dictionaries containing the embeddings for each sequence.
        Each dictionary contains embeddings for different pooling types.

    """
    embeddings = []

    ref_output = process_single_sequence(str(ref_sequence), model, tokenizer, model_name, max_len)
    embeddings.append(ref_output)

    for mutated_sequence in mutated_sequences:
        pooled_output = process_single_sequence(
            mutated_sequence, model, tokenizer, model_name, max_len
        )
        embeddings.append(pooled_output)

    return embeddings


def compute_cosine_similarities(
    embeddings_all: list[dict[str, np.ndarray]],
    model_name: str,
) -> dict[str, dict[str, float]]:
    """Compute cosine similarities between the reference and mutated embeddings.

    Embeddings are first converted to numpy arrays and then
    cosine similarity is computed.
    The first embedding in the array is the reference sequence
    and the rest are from mutated sequences.
    The cosine similarity is averaged over all the mutated sequences.

    Args:
        embeddings_all: a list of dictionaries containing the embeddings for each sequence.
        model_name: a string containing the name of the model.

    Returns:
        a dictionary containing the average cosine similarities for each pooling type.

    """
    results = {}
    for pooling_type in supported_pooling.get(model_name, []):
        embeddings_all_np = np.array([emb[pooling_type] for emb in embeddings_all])
        cos_sims = cosine_similarity([embeddings_all_np[0]], embeddings_all_np[1:])
        avg_cos_sim = round(np.mean(cos_sims), 3)
        results[pooling_type] = avg_cos_sim
    return results


def run_mutation_experiments() -> None:
    """Run mutation-based sensitivity experiments.

    This function handles the mutation augmentation process by processing reference sequences,
    generating mutated sequences with varying numbers of mutations, computing embeddings using
    different models, and aggregating cosine similarity scores.

    """
    ref_sequences = sample_sequences_from_genome()
    transform_vals = [1, 64, 128, 256, 512, 1024]

    results = {
        model_name: {
            pooling: {val: [] for val in transform_vals}
            for pooling in supported_pooling.get(model_name, [])
        }
        for model_name in MODEL_NAMES
    }

    for model_name in MODEL_NAMES:
        model, tokenizer, max_length = load_model_and_tokenizer(model_name, pretrained=True)

        for _, _, ref_seq in ref_sequences:
            for num_mutations in transform_vals:
                mutated_seqs = mutate_n_nucleotides(
                    sequence=ref_seq,
                    num_mutations=num_mutations,
                    num_sequences=5,
                )
                embeddings_all = compute_sequence_embeddings(
                    ref_seq,
                    mutated_seqs,
                    model_name,
                    max_length,
                    model,
                    tokenizer,
                )

                scores = compute_cosine_similarities(embeddings_all, model_name)
                for pooling in supported_pooling.get(model_name, []):
                    results[model_name][pooling][num_mutations].append(
                        scores[pooling],
                    )

    finalize_and_save_results(results)


def finalize_and_save_results(results: dict) -> None:
    """Finalize results by averaging and save to files.

    This method handles both mutation and transform results. If an augmentation type is provided,
    it treats the results as transform results; otherwise, it treats them as mutation results.

    Args:
        results: The results dictionary containing mutation scores.

    """
    print("Mutation results:", results)
    transform_vals = [1, 64, 128, 256, 512, 1024]

    for model_name in MODEL_NAMES:
        if model_name not in results:
            continue
        for pooling in supported_pooling.get(model_name, []):
            
            for val in transform_vals:
                if results[model_name][pooling][val]:
                    avg_score = round(np.mean(results[model_name][pooling][val]), 3)
                    results[model_name][pooling][val] = avg_score
                else:
                    results[model_name][pooling][val] = None

        filename = f"./embeddings-mistral-test/{model_name}_mutation_results.pkl"

        with Path(filename).open("wb") as f:
            pickle.dump(results[model_name], f)


if __name__ == "__main__":
    run_mutation_experiments()
