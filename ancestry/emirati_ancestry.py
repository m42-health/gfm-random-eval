GULF_POPULATIONS = {
    "Emirati": {"reference": "UAE_Emirati_1000G_equiv", "n_samples": 500},
    "Saudi": {"reference": "Saudi_1000G_equiv", "n_samples": 300},
    "Egyptian": {"reference": "Egyptian_1000G_equiv", "n_samples": 300},
    "Lebanese": {"reference": "Lebanese_1000G_equiv", "n_samples": 200}
}

def evaluate_gulf_ancestry_prediction(model, vcf_path, population="Emirati"):
    """
    Evaluate GFM ancestry discrimination for Gulf populations.
    Standard 1000G superpopulations collapse all Arab ancestry
    into 'Middle East' — this module resolves Gulf subpopulations.
    """
    variants = load_vcf_gulf_filtered(vcf_path, population)
    embeddings = model.encode(variants)
    
    # PCA + clustering accuracy
    pca = PCA(n_components=10)
    pca_emb = pca.fit_transform(embeddings)
    
    # Statistical test: are Gulf populations separable?
    from scipy.stats import permutation_test
    stat, pvalue = permutation_test(pca_emb, population_labels)
    
    return {
        "population": population,
        "separability_pvalue": pvalue,
        "pca_variance_explained": pca.explained_variance_ratio_[:5].tolist()
    }
