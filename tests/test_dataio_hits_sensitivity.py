import pandas as pd
import pytest

from cetsax.config import DOSE_COLS, ID_COL
from cetsax.dataio import apply_basic_qc, load_cetsa_csv
from cetsax.fit import fit_all_proteins
from cetsax.hits import call_hits, summarize_hits
from cetsax.sensitivity import (
    compute_sensitivity_heterogeneity,
    compute_sensitivity_scores,
    summarize_sensitivity_by_pathway,
)


def test_load_cetsa_csv_drops_unnamed_and_coerces_numeric(tmp_path, synthetic_raw_df):
    df = synthetic_raw_df.copy()
    df.insert(0, 'Unnamed: 0', range(len(df)))
    df[DOSE_COLS[0]] = df[DOSE_COLS[0]].astype(str)
    path = tmp_path / 'cetsa.csv'
    df.to_csv(path, index=False)
    loaded = load_cetsa_csv(str(path))
    assert 'Unnamed: 0' not in loaded.columns
    assert pd.api.types.is_numeric_dtype(loaded[DOSE_COLS[0]])


def test_apply_basic_qc_filters_low_quality_rows(synthetic_raw_df):
    qc = apply_basic_qc(synthetic_raw_df)
    assert 'LOWQC' not in set(qc[ID_COL])
    assert set(qc[ID_COL]) == {'P001', 'P002'}


def test_call_hits_and_summarize_hits(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    hits_df = call_hits(fit_df, r2_min=0.5, delta_min=0.05)
    assert not hits_df.empty
    summary = summarize_hits(hits_df, min_reps=2)
    assert set(summary.columns) == {ID_COL, 'n_reps', 'EC50_median', 'EC50_sd', 'Emax_median', 'Hill_median'}
    assert (summary['n_reps'] >= 2).all()


def test_compute_sensitivity_scores_ranks_proteins(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    sens = compute_sensitivity_scores(fit_df)
    assert {'NSS', 'NSS_rank', 'EC50_scaled', 'delta_max_scaled', 'Hill_scaled', 'R2_scaled'}.issubset(sens.columns)
    assert sens['NSS'].is_monotonic_decreasing
    assert sens['NSS_rank'].min() == 1.0


def test_compute_sensitivity_scores_invalid_agg_raises(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    with pytest.raises(ValueError):
        compute_sensitivity_scores(fit_df, agg='sum')


def test_sensitivity_pathway_summary_and_heterogeneity(synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    sens = compute_sensitivity_scores(fit_df)
    annot = pd.DataFrame({ID_COL: ['P001', 'P002'], 'pathway': ['redox', 'other']})
    summary = summarize_sensitivity_by_pathway(sens, annot)
    assert {'pathway', 'N', 'NSS_mean', 'NSS_top25', 'EC50_median', 'delta_max_median'}.issubset(summary.columns)
    hetero = compute_sensitivity_heterogeneity(sens, bins=5)
    assert {'gini', 'hist', 'edges', 'top10_threshold'} <= set(hetero)
    assert len(hetero['hist']) == 5
