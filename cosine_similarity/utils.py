from __future__ import annotations

import random

import numpy as np
import pysam
import torch
import transformers

max_sequence_length = {
    "nt_50m": 2048,
    "nt_500m": 1000,
    "dnabertv2": 512,
    "hyenadna": 1024,
    "mistral": 4096,
    "genalm": 512,
    "caduceus": 131072,
}

embed_dims = {
    "nt_50m": 512,
    "nt_500m": 1280,
    "dnabertv2": 768,
    "hyenadna": 128,
    "mistral": 1408,
    "genalm": 768,
    "caduceus": 256,
}

supported_pooling = {
    "nt_50m": ["cls", "max", "avg"],
    "nt_500m": ["cls", "max", "avg"],
    "dnabertv2": ["cls", "max", "avg"],
    "hyenadna": ["last", "max", "avg"],
    "mistral": ["last", "max", "avg"],
    "genalm": ["cls", "max", "avg"],
    "caduceus": ["last", "max", "avg"],
}

MODEL_NAMES = [
    "nt_50m",
    "nt_500m",
    "dnabertv2",
    "hyenadna",
    "mistral",
    "genalm",
    "caduceus",
]


def chunk_sequence(sequence: str, chunk_length: int = 1024) -> list[dict]:
    """Chunk the sequence into parts with specified fixed length.

    Args:
        sequence (str): The sequence to chunk.
        chunk_length (int, optional): The fixed length of each chunk. Defaults to 1024.

    Returns:
        list[dict]: A list of dictionaries containing chunk sequences and their start and end positions.

    """
    chunks = []
    for i in range(0, len(sequence), chunk_length):
        chunk_seq = sequence[i : i + chunk_length]
        chunk_info = {"seq": chunk_seq, "start": i, "end": min(i + chunk_length, len(sequence))}
        chunks.append(chunk_info)
    return chunks


def load_ref_genome() -> pysam.FastaFile:
    """Load the reference genome.

    Returns
    -------
    pysam.FastaFile
        The reference genome.

    """
    ref_genome_path = "/home/data/pretrain/genomics/hg38_reference/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna"
    return pysam.FastaFile(ref_genome_path)


def mutate_n_nucleotides(sequence: str, num_mutations: int, num_sequences: int = 10) -> list[str]:
    """Mutate the sequence.

    Args:
        sequence (str): The sequence to mutate.
        num_mutations (int): The number of mutations to introduce.
        num_sequences (int, optional): The number of sequences to generate. Defaults to 10.

    Returns:
        list[str]: A list of mutated sequences.

    """
    nucleotides = ["A", "C", "G", "T"]
    sequence_length = len(sequence)
    mutated_sequences = []

    for _ in range(num_sequences):
        mutable_sequence = list(sequence)
        positions_to_mutate = random.sample(
            range(sequence_length), min(num_mutations, sequence_length)
        )

        for pos in positions_to_mutate:
            original_nucleotide = mutable_sequence[pos]
            possible_replacements = [nuc for nuc in nucleotides if nuc != original_nucleotide]
            new_nucleotide = random.choice(possible_replacements)
            mutable_sequence[pos] = new_nucleotide

        mutated_seq = "".join(mutable_sequence)
        mutated_sequences.append(mutated_seq)

    return mutated_sequences


def sample_valid_sequence(
    chromosome: pysam.FastaFile, start: int, length: int, max_attempts: int = 100
) -> str:
    """Sample a valid sequence from the genome.

    Args:
        chromosome (pysam.FastaFile): The chromosome to sample from.
        start (int): The start position to sample from.
        length (int): The length of the sequence to sample.
        max_attempts (int, optional): The maximum number of attempts to sample a valid sequence. Defaults to 100.

    Returns:
        str: A valid sequence sampled from the genome.

    Raises:
        ValueError: If unable to sample a valid sequence after max_attempts.

    """
    attempts = 0
    while attempts < max_attempts:
        sequence = chromosome[start : start + length]
        if set(sequence).issubset({"A", "C", "G", "T"}):
            return sequence
        start += 1000
        attempts += 1
    raise ValueError("Failed to sample a valid sequence after multiple attempts.")


def sample_sequences_from_genome(sequence_length: int = 1024) -> list[tuple[str, int, str]]:
    """Sample sequences from the genome.

    Args:
        sequence_length (int, optional): The length of the sequence to sample. Defaults to 1024.

    Returns:
        list[tuple[str, int, str]]: A list of tuples containing the chromosome, start position, and sequence.

    """
    chromosomes = [7, 11, 12, 17, 19]
    start_positions = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000]
    sampled_sequences = []

    ref_genome = load_ref_genome()

    for chrom in chromosomes:
        chromosome = ref_genome[f"chr{chrom}"]
        for start in start_positions:
            seq = sample_valid_sequence(chromosome, start, sequence_length)
            sampled_sequences.append((chrom, start, seq))

    return sampled_sequences


def get_model_hidden_states(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: torch.Tensor,
    pretrained_model_name: str,
) -> torch.Tensor:
    """Get the model hidden states for the given input IDs.

    Args:
        model: a torch.nn.Module model.
        tokenizer: a transformers.PreTrainedTokenizer tokenizer.
        input_ids: a torch.Tensor containing the input IDs.
        pretrained_model_name: a string containing the name of the pretrained model.

    Returns:
        a torch.Tensor containing the model hidden states.

    """
    if "hyena" in pretrained_model_name or "caduceus" in pretrained_model_name:
        return model(input_ids, output_hidden_states=True).hidden_states[-1].detach().cpu()
    if (
        "nt" in pretrained_model_name
        or "llama" in pretrained_model_name
        or "mistral" in pretrained_model_name
    ):
        attention_mask = input_ids != tokenizer.pad_token_id
        return (
            model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            .hidden_states[-1]
            .detach()
            .cpu()
        )
    if "dnabert" in pretrained_model_name:
        return model(input_ids)[1].detach().cpu()
    if "genalm" in pretrained_model_name:
        return model(input_ids).hidden_states[-1].detach().cpu()
    msg = f"Model {pretrained_model_name} not found"
    raise ValueError(msg)


def get_last_non_special_token_indices(
    input_ids: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
) -> torch.Tensor:
    """Get the last non-special token indices for the given input IDs.

    Args:
        input_ids: a torch.Tensor containing the input IDs.
        tokenizer: a transformers.PreTrainedTokenizer tokenizer.

    Returns:
        a torch.Tensor containing the last non-special token indices.

    """
    special_token_ids = set(tokenizer.all_special_ids)
    non_special_mask = ~torch.isin(
        input_ids, torch.tensor(list(special_token_ids), device="cuda")
    ).bool()
    valid_token_counts = (input_ids != tokenizer.pad_token_id) & non_special_mask
    last_valid_indices = valid_token_counts.sum(dim=1) - 1
    return last_valid_indices.clamp(min=0)


@torch.no_grad()
def process_single_sequence(
    sequence: str,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    model_name: str,
    chunk_length: int,
) -> dict[str, np.ndarray]:
    """Process a single sequence to compute its embeddings.

    Args:
        sequence: The sequence string to process.
        model: The model to use.
        tokenizer: The tokenizer to use.
        model_name: The name of the model.
        chunk_length: The length of the chunks to use.

    Returns:
        A dictionary of pooled embeddings.

    """
    chunks = chunk_sequence(sequence, chunk_length)
    chunks = [c["seq"] for c in chunks]
    tokenized_seq = tokenizer(
        chunks,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = tokenized_seq["input_ids"].to("cuda")
    with torch.amp.autocast("cuda"):
        model_output = get_model_hidden_states(
            model,
            tokenizer,
            input_ids,
            model_name,
        )

        last_valid_indices = get_last_non_special_token_indices(
            input_ids,
            tokenizer,
        )
        max_pooled = model_output.max(dim=1)[0]
        avg_pooled = model_output.mean(dim=1)
        cls_pooled = model_output[:, 0] if "cls" in supported_pooling.get(model_name, []) else None

        last_pooled = None
        if "last" in supported_pooling.get(model_name, []):
            last_non_pad_tokens = model_output[
                torch.arange(input_ids.size(0)),
                last_valid_indices.cpu(),
            ]
            last_pooled = last_non_pad_tokens

        pooled_output = {
            "max": max_pooled.mean(dim=0).cpu().numpy(),
            "avg": avg_pooled.mean(dim=0).cpu().numpy(),
        }
        if cls_pooled is not None:
            pooled_output["cls"] = cls_pooled.mean(dim=0).cpu().numpy()
        if last_pooled is not None:
            pooled_output["last"] = last_pooled.mean(dim=0).cpu().numpy()

    return pooled_output
