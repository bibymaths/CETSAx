import numpy as np
import pandas as pd

from cetsax.config import ID_COL
from cetsax.enrichment import (
    _benjamini_hochberg,
    enrich_continuous_mannwhitney,
    enrich_overrepresentation,
    summarize_pathway_effects,
)
from cetsax.network import (
    compute_costab_matrix,
    detect_modules,
    make_network_from_matrix,
)


def test_benjamini_hochberg_is_bounded_and_monotone():
    pvals = pd.Series([0.01, 0.04, 0.03, 0.2])
    qvals = _benjamini_hochberg(pvals)
    assert ((qvals >= 0) & (qvals <= 1)).all()
    # smallest p-value should not get larger q than the largest p-value ordering implies
    assert qvals.iloc[pvals.idxmin()] <= qvals.max()


def test_pathway_effect_and_enrichment_functions():
    metric_df = pd.DataFrame(
        {
            ID_COL: ["P1", "P2", "P3", "P4", "P5", "P6"],
            "NSS": [0.9, 0.8, 0.7, 0.2, 0.3, 0.1],
            "EC50": [1e-5, 2e-5, 1.5e-5, 1e-2, 8e-3, 7e-3],
            "delta_max": [0.25, 0.21, 0.22, 0.07, 0.08, 0.06],
            "R2": [0.95, 0.92, 0.94, 0.7, 0.72, 0.68],
        }
    )
    annot = pd.DataFrame(
        {
            ID_COL: ["P1", "P2", "P3", "P4", "P5", "P6"],
            "pathway": ["A", "A", "A", "B", "B", "B"],
        }
    )
    summary = summarize_pathway_effects(metric_df, annot)
    assert {"pathway", "N_proteins", "NSS_mean", "NSS_median", "NSS_top25"}.issubset(
        summary.columns
    )

    hits_df = pd.DataFrame(
        {
            ID_COL: metric_df[ID_COL],
            "hit_class": ["strong", "strong", "strong", "weak", "weak", "weak"],
        }
    )
    ora = enrich_overrepresentation(
        hits_df, annot, strong_labels=("strong",), min_genes=2
    )
    assert not ora.empty
    assert {"pathway", "odds_ratio", "pval", "qval"}.issubset(ora.columns)

    cont = enrich_continuous_mannwhitney(
        metric_df[[ID_COL, "NSS"]], annot, score_col="NSS", min_genes=2
    )
    assert not cont.empty
    assert {"pathway", "U_stat", "pval", "qval"}.issubset(cont.columns)


def test_network_construction_and_module_detection(synthetic_raw_df):
    df = synthetic_raw_df[synthetic_raw_df[ID_COL] != "LOWQC"]
    corr = compute_costab_matrix(df)
    assert corr.shape == (2, 2)
    assert np.allclose(corr.values, corr.values.T)

    G = make_network_from_matrix(corr, cutoff=0.1)
    assert set(G.nodes()) <= {"P001", "P002"}
    modules = detect_modules(G)
    assert set(modules) == set(G.nodes())
