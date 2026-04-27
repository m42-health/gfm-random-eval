MODELS_TO_COMPARE = {
    "random_baseline": RandomModel(),
    "biofm": BioFMModel.from_pretrained("m42-health/biofm"),
    "dnabert2": DNABERT2.from_pretrained("zhihan1996/DNABERT-2-117M"),
    "nucleotide_transformer": NT.from_pretrained("InstaDeepAI/nucleotide-transformer-v2"),
    "hyena_dna": HyenaDNA.from_pretrained("LongSafari/hyenadna-large"),
    "evo": Evo.from_pretrained("togethercomputer/evo-1")
}

def run_full_benchmark(benchmark_suite):
    results = {}
    for name, model in MODELS_TO_COMPARE.items():
        results[name] = {
            "ancestry": evaluate_ancestry(model),
            "biotype": evaluate_biotype(model),
            "pgx": evaluate_pgx(model),
            "gulf_ancestry": evaluate_gulf_ancestry(model)
        }
    return pd.DataFrame(results).T
