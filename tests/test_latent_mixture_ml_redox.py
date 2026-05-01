import pandas as pd

from cetsax.config import ID_COL
from cetsax.dataio import apply_basic_qc
from cetsax.fit import fit_all_proteins
from cetsax.hits import call_hits, summarize_hits
from cetsax.latent import (
    attach_latent_to_metadata,
    build_feature_matrix,
    fit_factor_analysis,
    fit_pca,
)
from cetsax.mixture import (
    assign_mixture_clusters,
    build_mixture_features,
    fit_gmm_bic_grid,
    label_clusters_by_sensitivity,
)
from cetsax.ml import classify_curves_kmeans, detect_outliers, extract_curve_features
from cetsax.redox import build_redox_axes, summarize_redox_by_pathway
from cetsax.sensitivity import compute_sensitivity_scores


def test_latent_feature_building_and_projection(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    sens = compute_sensitivity_scores(fit_df)
    redox_input = pd.DataFrame(
        {
            ID_COL: sens[ID_COL],
            "axis_direct": [0.1, 0.9],
            "axis_indirect": [0.7, 0.2],
            "axis_network": [0.3, 0.4],
        }
    )
    feat = build_feature_matrix(sens, redox_input)
    assert feat.index.name == ID_COL
    assert feat.shape[0] == 2

    pca_res = fit_pca(feat, n_components=2)
    fa_res = fit_factor_analysis(feat, n_components=2)
    assert list(pca_res["scores"].columns) == ["PC1", "PC2"]
    assert list(fa_res["scores"].columns) == ["F1", "F2"]

    merged = attach_latent_to_metadata(sens[[ID_COL]], pca_res["scores"])
    assert {"PC1", "PC2"}.issubset(merged.columns)


def test_mixture_and_redox_workflow(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    sens = compute_sensitivity_scores(fit_df)
    hits = summarize_hits(call_hits(fit_df, r2_min=0.5, delta_min=0.05), min_reps=1)
    hits = hits[[ID_COL]].assign(dominant_class=["strong", "weak"])
    net_df = pd.DataFrame(
        {ID_COL: sens[ID_COL], "degree": [5, 1], "betweenness": [0.5, 0.1]}
    )
    redox = build_redox_axes(fit_df, sens, hits, net_df=net_df)
    assert {"axis_direct", "axis_indirect", "axis_network", "redox_role"}.issubset(
        redox.columns
    )

    annot = pd.DataFrame({ID_COL: sens[ID_COL], "pathway": ["A", "B"]})
    redox_summary = summarize_redox_by_pathway(redox, annot)
    assert {
        "pathway",
        "N",
        "frac_direct_core",
        "frac_indirect_responder",
        "frac_network_mediator",
        "frac_peripheral",
    }.issubset(redox_summary.columns)

    mix_feat = build_mixture_features(sens, redox)
    gmm_info = fit_gmm_bic_grid(mix_feat, n_components_grid=[1, 2], random_state=0)
    cluster_df = assign_mixture_clusters(mix_feat, gmm_info["best_model"])
    assert "cluster" in cluster_df.columns
    labeled = label_clusters_by_sensitivity(sens, cluster_df)
    assert {"cluster", "mean_score", "label"}.issubset(labeled.columns)


def test_curve_ml_helpers(synthetic_raw_df):
    df = synthetic_raw_df[synthetic_raw_df[ID_COL] != "LOWQC"]
    feat = extract_curve_features(df, n_components=2)
    assert list(feat.columns) == ["PC1", "PC2"]
    classified = classify_curves_kmeans(feat, k=2)
    assert "cluster" in classified.columns
    outliers = detect_outliers(feat)
    assert outliers["outlier"].dtype == bool
