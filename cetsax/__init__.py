"""
CETSAx – CETSA-MS modelling toolkit.

This package currently implements ITDR-based binding curve fitting
(EC50, Hill, Emax) for proteome-wide NADPH CETSA data.
It also includes modules for hit calling, pathway enrichment,
latent factor analysis, mixture modelling, redox role analysis,
and sequence-based deep learning models of NADPH responsiveness.

"""
from .deeplearn import train_seq_model, NADPHSeqConfig, build_sequence_supervised_table
from .config import DOSE_COLS, QC_MIN_UNIQUE_PEPTIDES, QC_MIN_PSMS, QC_MIN_COUNTNUM, ID_COL, COND_COL
from .dataio import load_cetsa_csv, apply_basic_qc
from .models import itdr_model
from .fit import fit_all_proteins
from .hits import call_hits, summarize_hits
from .plotting import plot_protein_curve, plot_goodness_of_fit
from .bayes import bayesian_fit_ec50
from .enrichment import summarize_pathway_effects, enrich_overrepresentation, enrich_continuous_mannwhitney
from .latent import build_feature_matrix, fit_pca, fit_factor_analysis, attach_latent_to_metadata
from .mixture import build_mixture_features, fit_gmm_bic_grid, assign_mixture_clusters, label_clusters_by_sensitivity
from .ml import extract_curve_features, classify_curves_kmeans, detect_outliers
from .network import compute_costab_matrix, make_network_from_matrix, detect_modules
from .redox import build_redox_axes, summarize_redox_by_pathway
from .sensitivity import compute_sensitivity_scores, summarize_sensitivity_by_pathway, compute_sensitivity_heterogeneity
from .viz_hits import run_hit_calling_and_plots
from .viz import (
    plot_pathway_effects_bar,
    plot_pathway_enrichment_volcano,
    plot_redox_axes_scatter,
    plot_redox_axes_scatter,
    plot_redox_role_composition,
    plot_pca_scores,
    plot_factor_scores,
    plot_mixture_clusters_in_pca,
    plot_cluster_size_bar
)

__all__ = [
    "ID_COL",
    "COND_COL",
    "DOSE_COLS",
    "QC_MIN_UNIQUE_PEPTIDES",
    "QC_MIN_PSMS",
    "QC_MIN_COUNTNUM",
    "load_cetsa_csv",
    "apply_basic_qc",
    "itdr_model",
    "fit_all_proteins",
    "call_hits",
    "summarize_hits",
    "plot_protein_curve",
    "plot_goodness_of_fit",
    "bayesian_fit_ec50",
    "summarize_pathway_effects",
    "enrich_overrepresentation",
    "enrich_continuous_mannwhitney",
    "build_feature_matrix",
    "fit_pca",
    "fit_factor_analysis",
    "attach_latent_to_metadata",
    "build_mixture_features",
    "fit_gmm_bic_grid",
    "assign_mixture_clusters",
    "label_clusters_by_sensitivity",
    "extract_curve_features",
    "classify_curves_kmeans",
    "detect_outliers",
    "compute_costab_matrix",
    "make_network_from_matrix",
    "detect_modules",
    "build_redox_axes",
    "summarize_redox_by_pathway",
    "compute_sensitivity_scores",
    "summarize_sensitivity_by_pathway",
    "compute_sensitivity_heterogeneity",
    "run_hit_calling_and_plots",
    "plot_pathway_effects_bar",
    "plot_pathway_enrichment_volcano",
    "plot_redox_axes_scatter",
    "plot_redox_role_composition",
    "plot_pca_scores",
    "plot_factor_scores",
    "plot_mixture_clusters_in_pca",
    "plot_cluster_size_bar",
    "train_seq_model",
    "NADPHSeqConfig",
    "build_sequence_supervised_table",
]
