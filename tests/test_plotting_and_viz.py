import matplotlib
matplotlib.use('Agg')

import pandas as pd
import pytest

from cetsax.config import COND_COL, ID_COL
from cetsax.dataio import apply_basic_qc
from cetsax.fit import fit_all_proteins
from cetsax.plotting import plot_goodness_of_fit, plot_protein_curve
from cetsax.sensitivity import compute_sensitivity_scores
from cetsax.viz import (
    plot_cluster_size_bar,
    plot_factor_scores,
    plot_mixture_clusters_in_pca,
    plot_pathway_effects_bar,
    plot_pathway_enrichment_volcano,
    plot_pca_scores,
    plot_redox_axes_scatter,
    plot_redox_role_composition,
)


def test_plotting_functions_return_axes(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    raw_df = apply_basic_qc(synthetic_raw_df)

    ax = plot_protein_curve(raw_df, fit_df, protein_id='P001', condition='NADPH.r1')
    assert ax.get_xscale() == 'log'

    fig, ax2 = plot_goodness_of_fit(raw_df, fit_df)
    assert ax2.get_xlabel() == 'Predicted (model)'


def test_plot_protein_curve_raises_for_missing_id(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    with pytest.raises(ValueError):
        plot_protein_curve(apply_basic_qc(synthetic_raw_df), fit_df, protein_id='MISSING')


def test_high_level_viz_helpers_render():
    path_df = pd.DataFrame({'pathway': ['A', 'B'], 'NSS_mean': [0.9, 0.4], 'frac_direct_core': [0.7, 0.2], 'frac_indirect_responder': [0.2, 0.3], 'frac_network_mediator': [0.1, 0.4], 'frac_peripheral': [0.0, 0.1]})
    enr_df = pd.DataFrame({'pathway': ['A', 'B'], 'odds_ratio': [4.0, 1.5], 'qval': [0.01, 0.2]})
    redox_df = pd.DataFrame({'axis_direct': [0.1, 0.9], 'axis_indirect': [0.8, 0.2], 'redox_role': ['mixed', 'direct_core']})
    scores_df = pd.DataFrame({'PC1': [1.0, -1.0], 'PC2': [0.5, -0.5]}, index=['P1', 'P2'])
    factor_df = pd.DataFrame({'F1': [0.4, -0.2], 'F2': [0.1, 0.3]}, index=['P1', 'P2'])
    cluster_df = pd.DataFrame({'id': ['P1', 'P2'], 'cluster': [0, 1]})
    cluster_sizes = pd.DataFrame({'cluster': [0, 1], 'n_proteins': [5, 3]})

    assert plot_pathway_effects_bar(path_df)[1].get_title()
    assert plot_pathway_enrichment_volcano(enr_df)[1].get_ylabel() == '-log10(q-value)'
    assert plot_redox_axes_scatter(redox_df)[1].get_xlabel() == 'axis_direct'
    assert plot_redox_role_composition(path_df)[1].get_ylabel() == 'Fraction of proteins'
    assert plot_pca_scores(scores_df)[1].get_xlabel() == 'PC1'
    assert plot_factor_scores(factor_df)[1].get_xlabel() == 'PC1'
    assert plot_mixture_clusters_in_pca(scores_df, cluster_df)[1].get_xlabel() == 'PC1'
    assert plot_cluster_size_bar(cluster_sizes)[1].get_ylabel() == 'Number of proteins'
