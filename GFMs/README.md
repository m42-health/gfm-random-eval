# Genomic Foundationless Models: Pretraining Does Not Promise Performance

**Authors:** Kirill Vishniakov, Karthik Viswanathan, Aleksandr Medvedev, Praveenkumar Kanithi, Marco AF Pimentel, Ronnie Rajan, Shadab Khan

> **Abstract:** The success of Large Language Models has inspired the development of Genomic Foundation Models (GFMs) through similar pretraining techniques. However, the relationship between pretraining performance and effectiveness in downstream genomic tasks remains unclear. Additionally, the high computational cost of pretraining raises questions about its cost-efficiency. To assess the usefulness of pretraining in genomics, we evaluated seven different GFMs across various benchmarks, comparing them to their counterparts with randomly initialized weights. Surprisingly, we found that randomly initialized models can match or even surpass the performance of pretrained GFMs in finetuning and feature extraction tasks. We also discovered that pretrained GFMs fail to capture clinically relevant genetic mutations, which are crucial for understanding genetic disorders and phenotypic traits. Our results indicate that most of the current pretrained GFMs lack a ``foundational'' understanding of genomics and provide minimal utility, even for basic tasks such as sequence classification. These findings collectively highlight the need for critically rethinking the pretraining approaches for genomics.

## How to Run

### NT Benchmark

1. Create a sweep using `wandb` and one of the sweep config files from nt_benchmark/sweeps directory.

```
wandb sweep nt_benchmark/sweeps/test.yaml
```

2. Copy SWEEP_ID into nt_benchmark/run_sweep.sh

```
SWEEP_ID="<PLACE YOUR SWEEP ID HERE>"
```

3. Run the sweep using the script `nt_benchmark/run_sweep.sh`

```
bash nt_benchmark/run_sweep.sh
```

### Biotype Benchmark

1. Create a sweep using `wandb` and one of the sweep config files from biotype/sweeps directory.

```
wandb sweep biotype/sweeps/test.yaml
```

2. Copy SWEEP_ID into biotype/run_sweep.sh

```
SWEEP_ID="<PLACE YOUR SWEEP ID HERE>"
```

3. Run the sweep using the script `biotype/run_sweep.sh`

```
bash biotype/run_sweep.sh
```

### Cosine Similarity

1. Run the script `cosine_similarity/main_sensitivity.py`

```
python cosine_similarity/main_sensitivity.py
```

2. Run the script `cosine_similarity/main_clinic.py`

```
python cosine_similarity/main_clinic.py
```

### Ancestry

1. Run the script `ancestry/main_ancestry.py`

```
python ancestry/main.py
```
