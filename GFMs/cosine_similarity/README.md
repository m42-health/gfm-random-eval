# Sensitivity Experiments

This folder contains scripts and utilities for conducting sensitivity experiments on genomic sequences using various pretrained genomic foundation models.

### Main Entry Points

There are two main entry points for running experiments:

1. `main_clinvar.py`: Conducts sensitivity experiments on the TP53 gene using ClinVar data.

To run the ClinVar experiment:

```
python main_clinvar.py [--clinvar_file PATH] [--genome_file PATH] [--tp53_fasta PATH] [--device DEVICE]
```

2. `main_sensitivity.py`: Conducts mutation-based sensitivity experiments.

To run the general sensitivity experiment:

```
python main_sensitivity.py
```

### Key Files

`main_clinvar.py`: Processes ClinVar data, generates mutated versions of the TP53 gene, and compares embeddings of reference and mutated sequences.

`main_sensitivity.py`: Handles mutation-based sensitivity experiments on sampled genomic sequences.

`models.py`: Contains functions for loading various language models and tokenizers used in the experiments.

`utils.py`: Includes utility functions for sequence manipulation, embedding computation, and other helper functions.

`debug.py`: A script for debugging and testing model behavior with simple test cases.

### Data

The experiments use genomic sequence data. Make sure you have the following files:

- ClinVar data file containing TP53 mutations (for `main_clinvar.py`)
- TP53 gene sequence in FASTA format (for `main_clinvar.py`)
- Reference genome file (for sampling sequences in `main_sensitivity.py`)

Place these files in the appropriate directories as referenced in the code.
