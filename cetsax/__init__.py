"""
CETSAx – CETSA-MS modelling toolkit.

This package currently implements ITDR-based binding curve fitting
(EC50, Hill, Emax) for proteome-wide NADPH CETSA data.
It also includes modules for hit calling, pathway enrichment,
latent factor analysis, mixture modelling, redox role analysis,
and sequence-based deep learning models of NADPH responsiveness.

"""
# BSD 3-Clause License
#
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
# Email: mishraabhinav36@gmail.com
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of Abhinav Mishra nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# --- Core (Always Available) ---
from .config import (
    DOSE_COLS,
    QC_MIN_UNIQUE_PEPTIDES,
    QC_MIN_PSMS,
    QC_MIN_COUNTNUM,
    ID_COL,
    COND_COL,
)
from .dataio import load_cetsa_csv, apply_basic_qc

# --- Optional Imports ---

try:
    from .deeplearn import (
        train_seq_model,
        NADPHSeqConfig,
        build_sequence_supervised_table,
    )
except ImportError:
    train_seq_model = NADPHSeqConfig = build_sequence_supervised_table = None

try:
    from .models import itdr_model
except ImportError:
    itdr_model = None

try:
    from .fit import fit_all_proteins
except ImportError:
    fit_all_proteins = None

try:
    from .hits import call_hits, summarize_hits
except ImportError:
    call_hits = summarize_hits = None

try:
    from .plotting import plot_protein_curve, plot_goodness_of_fit
except ImportError:
    plot_protein_curve = plot_goodness_of_fit = None

try:
    from .bayes import bayesian_fit_ec50
except ImportError:
    bayesian_fit_ec50 = None

try:
    from .enrichment import (
        summarize_pathway_effects,
        enrich_overrepresentation,
        enrich_continuous_mannwhitney,
    )
except ImportError:
    summarize_pathway_effects = enrich_overrepresentation = (
        enrich_continuous_mannwhitney
    ) = None

try:
    from .latent import (
        build_feature_matrix,
        fit_pca,
        fit_factor_analysis,
        attach_latent_to_metadata,
    )
except ImportError:
    build_feature_matrix = fit_pca = fit_factor_analysis = attach_latent_to_metadata = (
        None
    )

try:
    from .mixture import (
        build_mixture_features,
        fit_gmm_bic_grid,
        assign_mixture_clusters,
        label_clusters_by_sensitivity,
    )
except ImportError:
    build_mixture_features = fit_gmm_bic_grid = assign_mixture_clusters = (
        label_clusters_by_sensitivity
    ) = None

try:
    from .ml import extract_curve_features, classify_curves_kmeans, detect_outliers
except ImportError:
    extract_curve_features = classify_curves_kmeans = detect_outliers = None

try:
    from .network import compute_costab_matrix, make_network_from_matrix, detect_modules
except ImportError:
    compute_costab_matrix = make_network_from_matrix = detect_modules = None

try:
    from .redox import build_redox_axes, summarize_redox_by_pathway
except ImportError:
    build_redox_axes = summarize_redox_by_pathway = None

try:
    from .sensitivity import (
        compute_sensitivity_scores,
        summarize_sensitivity_by_pathway,
        compute_sensitivity_heterogeneity,
    )
except ImportError:
    compute_sensitivity_scores = summarize_sensitivity_by_pathway = (
        compute_sensitivity_heterogeneity
    ) = None

try:
    from .viz_hits import run_hit_calling_and_plots
except ImportError:
    run_hit_calling_and_plots = None

try:
    from .viz import (
        plot_pathway_effects_bar,
        plot_pathway_enrichment_volcano,
        plot_redox_axes_scatter,
        plot_redox_role_composition,
        plot_pca_scores,
        plot_factor_scores,
        plot_mixture_clusters_in_pca,
        plot_cluster_size_bar,
    )
except ImportError:
    plot_pathway_effects_bar = plot_pathway_enrichment_volcano = (
        plot_redox_axes_scatter
    ) = plot_redox_role_composition = plot_pca_scores = plot_factor_scores = (
        plot_mixture_clusters_in_pca
    ) = plot_cluster_size_bar = None

try:
    from .viz_predict import (
        visualize_predictions,
        analyze_fitting_data,
        generate_bio_insight,
        plot_training_loop,
    )
except ImportError:
    visualize_predictions = analyze_fitting_data = generate_bio_insight = (
        plot_training_loop
    ) = None

# --- Define Public API ---
__all__ = [
    # Core
    "ID_COL",
    "COND_COL",
    "DOSE_COLS",
    "QC_MIN_UNIQUE_PEPTIDES",
    "QC_MIN_PSMS",
    "QC_MIN_COUNTNUM",
    "load_cetsa_csv",
    "apply_basic_qc",
    # Optional
    "train_seq_model",
    "NADPHSeqConfig",
    "build_sequence_supervised_table",
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
    "visualize_predictions",
    "analyze_fitting_data",
    "generate_bio_insight",
    "plot_training_loop",
]
