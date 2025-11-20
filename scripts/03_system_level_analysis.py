#!/usr/bin/env python
"""
03_system_level_analysis.py

Run system-level analysis on CETSA NADPH data using cetsax modules:

    - sensitivity scores (NSS)  [H]
    - pathway summaries + enrichment [I]
    - co-stabilization network (optional) + redox axes [F, J]
    - latent factor models (PCA / FA) [K]
    - mixture modeling [L]
    - plots for each layer via cetsax.viz

Inputs:
    - ec50_fits.csv
    - cetsa_hits_ranked.csv
    - protein_annotations.csv

Outputs (in out_dir):
    - sensitivity_scores.csv
    - pathway_effects.csv
    - pathway_enrichment_overrepresentation.csv
    - redox_axes_per_protein.csv
    - redox_by_pathway.csv
    - pca_scores.csv, pca_loadings.csv
    - mixture_features.csv, mixture_clusters_per_protein.csv, mixture_cluster_labels.csv
    - multiple PNG plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cetsax import compute_sensitivity_scores
from cetsax import summarize_pathway_effects, enrich_overrepresentation
from cetsax import build_redox_axes, summarize_redox_by_pathway
from cetsax import build_feature_matrix, fit_pca, fit_factor_analysis
from cetsax import (
    build_mixture_features,
    fit_gmm_bic_grid,
    assign_mixture_clusters,
    label_clusters_by_sensitivity,
)
from cetsax import (
    plot_pathway_effects_bar,
    plot_pathway_enrichment_volcano,
    plot_redox_axes_scatter,
    plot_redox_role_composition,
    plot_pca_scores,
    plot_mixture_clusters_in_pca,
    plot_cluster_size_bar,
)
from cetsax import ID_COL


def main() -> None:
    p = argparse.ArgumentParser(description="Run system-level CETSA NADPH analysis.")
    p.add_argument(
        "--fits-csv",
        # required=True,
        default="../results/ec50_fits.csv",
        help="Path to ec50_fits.csv"
    )
    p.add_argument(
        "--hits-csv",
        # required=True,
        default="../results/hit_results/cetsa_hits_ranked.csv",
        help="Path to cetsa_hits_ranked.csv"
    )
    p.add_argument(
        "--annot-csv",
        # required=True,
        default="../results/protein_annotations.csv",
        help="Path to protein_annotations.csv (id,pathway,...)",
    )
    p.add_argument(
        "--out-dir",
        default="../results/system_results",
        help="Output directory for all results and plots.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fits = pd.read_csv(args.fits_csv)
    hits = pd.read_csv(args.hits_csv)
    annot = pd.read_csv(args.annot_csv)

    # ------------------------------------------------------------
    # 1. Sensitivity scores (NSS, etc.)  [H]
    # ------------------------------------------------------------
    sens = compute_sensitivity_scores(fits)
    sens_path = out_dir / "sensitivity_scores.csv"
    sens.to_csv(sens_path, index=False)
    print(f"[H] Saved sensitivity scores to {sens_path}")

    # ------------------------------------------------------------
    # 2. Pathway-level summaries + enrichment  [I]
    # ------------------------------------------------------------
    path_eff = summarize_pathway_effects(
        metric_df=sens,
        annot_df=annot,
        id_col=ID_COL,
        path_col="pathway",
    )
    path_eff_path = out_dir / "pathway_effects.csv"
    path_eff.to_csv(path_eff_path, index=False)

    enr = enrich_overrepresentation(
        hits_df=hits,
        annot_df=annot,
        id_col=ID_COL,
        path_col="pathway",
        hit_col="dominant_class",
        strong_labels=("strong", "medium"),
    )
    enr_path = out_dir / "pathway_enrichment_overrepresentation.csv"
    enr.to_csv(enr_path, index=False)
    print(f"[I] Saved pathway effects and enrichment to {path_eff_path}, {enr_path}")

    # Plot pathway bar + volcano
    fig, ax = plot_pathway_effects_bar(path_eff, metric="NSS_mean", top_n=20)
    fig.savefig(out_dir / "pathway_nss_mean_top20.png", dpi=300, bbox_inches="tight")

    fig, ax = plot_pathway_enrichment_volcano(enr, label_top_n=10)
    fig.savefig(out_dir / "pathway_enrichment_volcano.png", dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------
    # 3. Co-stabilization network + metrics  [C, F]
    # ------------------------------------------------------------
    # net = build_costabilization_network(
    #     fits_df=fits,
    #     id_col=ID_COL,
    #     cond_col=COND_COL,
    #     dose_cols=DOSE_COLS,
    #     method="pearson",
    #     corr_threshold=0.7,
    # )
    # net_metrics = summarize_network_metrics(net)
    # net_metrics_path = out_dir / "network_metrics.csv"
    # net_metrics.to_csv(net_metrics_path, index=False)
    # print(f"[F] Saved network metrics to {net_metrics_path}")

    # ------------------------------------------------------------
    # 4. Redox axes per protein  [J]
    # ------------------------------------------------------------
    redox = build_redox_axes(
        fits_df=fits,
        sens_df=sens,
        hits_df=hits,
        # net_df=net_metrics,
        id_col=ID_COL,
    )
    redox_path = out_dir / "redox_axes_per_protein.csv"
    redox.to_csv(redox_path, index=False)

    path_redox = summarize_redox_by_pathway(
        redox_df=redox,
        annot_df=annot,
        id_col=ID_COL,
        path_col="pathway",
    )
    path_redox_path = out_dir / "redox_by_pathway.csv"
    path_redox.to_csv(path_redox_path, index=False)
    print(f"[J] Saved redox axes + pathway summaries to {redox_path}, {path_redox_path}")

    # Plots: redox scatter + role composition
    fig, ax = plot_redox_axes_scatter(
        redox_df=redox,
        x_axis="axis_direct",
        y_axis="axis_indirect",
        color_by="redox_role",
    )
    fig.savefig(out_dir / "redox_axes_direct_vs_indirect.png", dpi=300, bbox_inches="tight")

    fig, ax = plot_redox_role_composition(path_redox, top_n=20)
    fig.savefig(out_dir / "redox_role_composition_top20_pathways.png", dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------
    # 5. Latent factor models (PCA / FA)  [K]
    # ------------------------------------------------------------
    feat_latent = build_feature_matrix(
        sens_df=sens,
        redox_df=redox,
        id_col=ID_COL,
    )
    pca_res = fit_pca(feat_latent, n_components=3)
    fa_res = fit_factor_analysis(feat_latent, n_components=3)

    # Save PCA scores/loadings
    pca_res["scores"].to_csv(out_dir / "pca_scores.csv", index=False)
    pca_res["loadings"].to_csv(out_dir / "pca_loadings.csv", index=False)

    # Save FA scores/loadings
    fa_res["scores"].to_csv(out_dir / "fa_scores.csv", index=False)
    fa_res["loadings"].to_csv(out_dir / "fa_loadings.csv", index=False)

    print("[K] Saved PCA/FA scores and loadings.")

    # Plot PCA scores colored by redox_role
    pca_scores = pca_res["scores"]
    meta = redox[[ID_COL, "redox_role"]]
    fig, ax = plot_pca_scores(
        scores_df=pca_scores,
        meta_df=meta,
        id_col=ID_COL,
        color_by="redox_role",
        pc_x="PC1",
        pc_y="PC2",
    )
    fig.savefig(out_dir / "pca_pc1_pc2_by_redox_role.png", dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------
    # 6. Mixture modelling  [L]
    # ------------------------------------------------------------
    feat_mix = build_mixture_features(
        sens_df=sens,
        redox_df=redox,
        id_col=ID_COL,
        feature_cols=["EC50", "delta_max", "NSS", "R2"],
        include_redox_axes=True,
        log_transform_ec50=True,
    )
    feat_mix.to_csv(out_dir / "mixture_features.csv")

    gmm_res = fit_gmm_bic_grid(
        feat_df=feat_mix,
        n_components_grid=[2, 3, 4, 5],
        covariance_type="full",
    )
    bic_table = gmm_res["bic_table"]
    bic_table.to_csv(out_dir / "gmm_bic_table.csv", index=False)

    clusters = assign_mixture_clusters(
        feat_df=feat_mix,
        gmm=gmm_res["best_model"],
        id_col=ID_COL,
    )
    clusters_path = out_dir / "mixture_clusters_per_protein.csv"
    clusters.to_csv(clusters_path, index=False)

    cluster_labels = label_clusters_by_sensitivity(
        sens_df=sens,
        cluster_df=clusters,
        id_col=ID_COL,
        score_col="NSS",
    )
    cluster_labels_path = out_dir / "mixture_cluster_labels.csv"
    cluster_labels.to_csv(cluster_labels_path, index=False)
    print(f"[L] Saved mixture clusters + labels to {clusters_path}, {cluster_labels_path}")

    # Plots: mixture clusters in PCA space + cluster sizes
    fig, ax = plot_mixture_clusters_in_pca(
        pca_scores=pca_scores,
        cluster_df=clusters,
        id_col=ID_COL,
        pc_x="PC1",
        pc_y="PC2",
    )
    fig.savefig(out_dir / "mixture_clusters_in_pca.png", dpi=300, bbox_inches="tight")

    fig, ax = plot_cluster_size_bar(clusters)
    fig.savefig(out_dir / "mixture_cluster_sizes.png", dpi=300, bbox_inches="tight")

    print("System-level analysis finished.")


if __name__ == "__main__":
    main()
