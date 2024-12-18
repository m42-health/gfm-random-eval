"""Conducts sensitivity experiments on the TP53 gene with various pretrained GFMs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
import transformers
from Bio import SeqIO
from sklearn.metrics.pairwise import cosine_similarity

from cosine_similarity.models import load_model_and_tokenizer
from cosine_similarity.utils import process_single_sequence, supported_pooling

CHUNK_LENGTH = 1024  # Fixed chunk length

PREDEFINED_CHUNKS = [
    {"start": 10500, "end": 10500 + CHUNK_LENGTH},
    {"start": 11524, "end": 11524 + CHUNK_LENGTH},
    {"start": 12548, "end": 12548 + CHUNK_LENGTH},
    {"start": 13572, "end": 13572 + CHUNK_LENGTH},
    {"start": 16750, "end": 16750 + CHUNK_LENGTH},
]


def load_exonic_clinvar_variants(file_path: str) -> pd.DataFrame:
    """Load the ClinVar data from the given file path and filters the data to only include exonic variants.

    Args:
        file_path: path to the ClinVar data file that contains variants for TP53.

    Returns:
        a pandas DataFrame containing the exonic variants from ClinVar for the TP53 gene.

    """
    clinvar_df = pd.read_csv(file_path)
    return clinvar_df[clinvar_df["Name"].str.contains("(p.", regex=False)]


def load_tp53_sequence(file_path: str) -> str:
    """Load the TP53 sequence from the given file path.

    Args:
        file_path: path to the TP53 fasta file.

    Returns:
        a string containing the TP53 sequence. This string is already coming as reverse complemented.

    """
    with Path(file_path).open() as handle:
        record = next(SeqIO.parse(handle, "fasta"))
    return str(record.seq)


def filter_tp53_clinvar_mutations(
    clinvar_data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the Clinvar mutations for the TP53 gene. Filter the mutations to only include pathogenic and benign variants.

    Args:
        clinvar_data: a pandas DataFrame containing the ClinVar data.
        ref_sequence: a string containing the reference original sequence.

    Returns:
        a pandas DataFrame containing the mutation information.

    """
    mutation_info = {}

    gene_df = clinvar_data[clinvar_data["GeneSymbol"] == "TP53"]

    mutation_info = gene_df[
        ["NewIndex", "Name", "AlternateAlleleVCF", "ClinicalSignificance"]
    ].copy()

    benign_categories = ["Likely benign", "Benign/Likely benign", "Benign"]
    pathogenic_categories = [
        "Pathogenic",
        "Pathogenic/Likely pathogenic",
        "Likely pathogenic",
    ]

    mutation_info["is_benign"] = mutation_info["ClinicalSignificance"].isin(benign_categories)
    mutation_info["is_pathogenic"] = mutation_info["ClinicalSignificance"].isin(
        pathogenic_categories
    )

    mutation_info = mutation_info[mutation_info["is_benign"] | mutation_info["is_pathogenic"]]

    mutation_info = mutation_info.drop(columns=["ClinicalSignificance"])

    mutation_info = mutation_info.sort_values("NewIndex").drop_duplicates("NewIndex", keep="first")

    mutation_info = mutation_info.reset_index(drop=True)

    pathogenic_count = mutation_info["is_pathogenic"].sum()
    benign_count = mutation_info["is_benign"].sum()

    print(f"TP53: {pathogenic_count} pathogenic and {benign_count} benign variants")
    return mutation_info


@torch.no_grad()
def compute_cosine_similarities(
    ref_sequence: str,
    mutated_info_tp53: pd.DataFrame,
    model: torch.nn.Module,
    model_name: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> list:
    """Process the given gene for the given model and tokenizer, computing cosine similarities per mutated chunk.

    Args:
        ref_sequence: a string containing the reference original sequence.
        mutated_info_tp53: a pandas DataFrame containing the mutation information.
        model: a torch.nn.Module model.
        model_name: a string containing the name of the model.
        tokenizer: a transformers.PreTrainedTokenizer tokenizer.

    Returns:
        a list of dictionaries containing the cosine similarities for the mutated chunks.
        Each dictionary contains the chunk start and end positions, and the cosine similarities for the max, avg, cls, and last pooling types.

    """
    benign_sequence = list(ref_sequence)
    pathogenic_sequence = list(ref_sequence)

    print(f"Number of unique mutations after filtering: {len(mutated_info_tp53)}")
    print(f"Pathogenic mutations: {mutated_info_tp53['is_pathogenic'].sum()}")
    print(f"Benign mutations: {mutated_info_tp53['is_benign'].sum()}")

    mutation_counts = {i: 0 for i in range(len(PREDEFINED_CHUNKS))}

    # Count mutations per chunk and analyze mutation types
    for i, chunk in enumerate(PREDEFINED_CHUNKS):
        chunk_mutations = mutated_info_tp53[
            (mutated_info_tp53["NewIndex"] >= chunk["start"])
            & (mutated_info_tp53["NewIndex"] < chunk["end"])
        ]
        mutation_count = len(chunk_mutations)
        mutation_counts[i] = mutation_count

        print(f"Chunk {i}: {mutation_count} mutations")
        if mutation_count > 0:
            mutation_types = (
                chunk_mutations["is_pathogenic"]
                .map({True: "Pathogenic", False: "Benign"})
                .value_counts()
            )
            print(f"  Mutation Types: {mutation_types.to_dict()}")

    # Apply mutations directly using NewIndex
    for _, mutation in mutated_info_tp53.iterrows():
        pos = mutation["NewIndex"]
        ref, alt = mutation["Name"].split(">")[0][-1], mutation["Name"].split(">")[1][0]
        assert (
            ref_sequence[pos] == ref
        ), f"Reference mismatch at position {pos}: expected {ref_sequence[pos]}, got {ref}"
        if mutation["is_benign"]:
            assert (
                benign_sequence[pos] == ref
            ), f"Benign mutation mismatch at position {pos}: expected {ref}, got {benign_sequence[pos]}"
            benign_sequence[pos] = alt
        elif mutation["is_pathogenic"]:
            assert (
                pathogenic_sequence[pos] == ref
            ), f"Pathogenic mutation mismatch at position {pos}: expected {ref}, got {pathogenic_sequence[pos]}"
            pathogenic_sequence[pos] = alt

    benign_sequence = "".join(benign_sequence)
    pathogenic_sequence = "".join(pathogenic_sequence)

    ref_embeddings = []
    benign_embeddings = []
    pathogenic_embeddings = []

    for chunk in PREDEFINED_CHUNKS:
        start, end = chunk["start"], chunk["end"]
        ref_chunk = ref_sequence[start:end]
        benign_chunk = benign_sequence[start:end]
        pathogenic_chunk = pathogenic_sequence[start:end]

        ref_embeddings.append(
            process_single_sequence(ref_chunk, model, tokenizer, model_name, CHUNK_LENGTH)
        )
        benign_embeddings.append(
            process_single_sequence(benign_chunk, model, tokenizer, model_name, CHUNK_LENGTH)
        )
        pathogenic_embeddings.append(
            process_single_sequence(pathogenic_chunk, model, tokenizer, model_name, CHUNK_LENGTH)
        )

    gene_results = []

    for idx, chunk in enumerate(PREDEFINED_CHUNKS):
        cos_sim = {}
        for pooling_type in supported_pooling.get(model_name, []):
            ref_emb = ref_embeddings[idx][pooling_type].reshape(1, -1)
            benign_emb = benign_embeddings[idx][pooling_type].reshape(1, -1)
            pathogenic_emb = pathogenic_embeddings[idx][pooling_type].reshape(1, -1)

            benign_similarity = cosine_similarity(ref_emb, benign_emb)[0][0]
            pathogenic_similarity = cosine_similarity(ref_emb, pathogenic_emb)[0][0]

            cos_sim[f"{pooling_type}_benign"] = round(benign_similarity, 3)
            cos_sim[f"{pooling_type}_pathogenic"] = round(pathogenic_similarity, 3)

        gene_results.append(
            {
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"],
                **cos_sim,
            }
        )

    return gene_results


def main(args: argparse.Namespace) -> None:
    """Run the sensitivity experiments.

    Args:
        args: an argparse.Namespace containing the command line arguments.

    """
    results = {}
    clinvar_data = load_exonic_clinvar_variants(args.clinvar_file)
    tp53_sequence = load_tp53_sequence(args.tp53_fasta)
    mutation_info = filter_tp53_clinvar_mutations(clinvar_data)

    model_names = [
        "mistral",
        "nt_500m",
        "nt_50m",
        "dnabertv2",
        "hyenadna",
        "genalm",
        "caduceus",
    ]

    for model_name in model_names:
        print(f"Loading model: {model_name}")
        model, tokenizer, _ = load_model_and_tokenizer(model_name, pretrained=True)

        print(f"Processing gene for model: {model_name}")
        gene_result = compute_cosine_similarities(
            tp53_sequence,
            mutation_info,
            model,
            model_name,
            tokenizer,
        )
        results[model_name] = gene_result
        print(f"Finished processing gene for model: {model_name}")

        for chunk_info in gene_result:
            print(f"Chunk {chunk_info['chunk_start']}-{chunk_info['chunk_end']}:")
            for key, sim in chunk_info.items():
                if key not in ["chunk_start", "chunk_end"]:
                    pooling, status = key.rsplit("_", 1)
                    print(f"  {pooling} - {status.capitalize()}: {sim:.4f}")
        print("\n")

    print("=======================", "\n\n")
    print(results)

    with Path("clinvar_results.pkl").open("wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinvar_file", type=str, default="./tp53_clinvar.csv")
    parser.add_argument(
        "--genome_file",
        type=str,
        default="/data/pretrain/genomics/hg38_reference/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna",
    )
    parser.add_argument("--tp53_fasta", type=str, default="./tp53.fasta")
    args = parser.parse_args()
    main(args)
