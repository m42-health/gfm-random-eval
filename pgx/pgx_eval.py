PGX_GENES = ["CYP2D6", "CYP2C9", "CYP2C19", "CYP3A4", "VKORC1", "HLA-B"]

def evaluate_pgx_variant_prediction(model, pgx_benchmark):
    """
    Evaluate whether GFM embeddings can distinguish:
    - Poor metabolizers vs normal vs ultra-rapid
    - HLA-B*57:01 (abacavir hypersensitivity — prevalent in Gulf)
    - VKORC1 c.1173C>T (warfarin dosing)
    """
    results = {}
    for gene in PGX_GENES:
        variants = pgx_benchmark.get_variants(gene)
        embeddings = model.encode([v.sequence for v in variants])
        
        # Classification: clinical annotation from PharmGKB
        labels = [v.clinical_annotation for v in variants]
        score = evaluate_classification(embeddings, labels)
        results[gene] = score
    
    return results
