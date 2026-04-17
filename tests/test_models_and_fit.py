import numpy as np

from cetsax.config import COND_COL, DOSE_COLS, ID_COL
from cetsax.fit import _fit_single_curve, _itdr_model_log, fit_all_proteins
from cetsax.models import itdr_model


def test_itdr_model_handles_zero_concentration():
    c = np.array([0.0, 1e-6, 1e-4, 1e-2])
    y = itdr_model(c, E0=1.0, Emax=1.5, logEC50=-4.0, h=1.0)
    assert y.shape == c.shape
    assert np.isfinite(y).all()
    assert y[0] < y[-1]


def test_itdr_model_log_matches_linear_model():
    doses = np.array(DOSE_COLS, dtype=float)
    params = {"E0": 0.95, "Emax": 1.25, "logEC50": -3.5, "h": 1.4}
    y_linear = itdr_model(doses, **params)
    y_log = _itdr_model_log(np.log10(doses), **params)
    np.testing.assert_allclose(y_linear, y_log, rtol=1e-10, atol=1e-10)


def test_fit_single_curve_returns_reasonable_parameters(doses):
    y = itdr_model(doses, 0.95, 1.30, -3.8, 1.2)
    fit = _fit_single_curve(doses, y)
    assert fit is not None
    assert 0.0 < fit['EC50'] < 1.0
    assert fit['delta_max'] > 0.05
    assert fit['R2'] > 0.95


def test_fit_single_curve_rejects_flat_signal(doses):
    y = np.ones_like(doses)
    assert _fit_single_curve(doses, y) is None


def test_fit_all_proteins_returns_expected_schema(synthetic_raw_df):
    fit_df = fit_all_proteins(synthetic_raw_df[synthetic_raw_df[ID_COL] != 'LOWQC'])
    expected = [ID_COL, COND_COL, 'E0', 'Emax', 'EC50', 'log10_EC50', 'Hill', 'R2', 'delta_max']
    assert list(fit_df.columns) == expected
    assert set(fit_df[ID_COL]) == {'P001', 'P002'}
    assert fit_df.shape[0] == 4
    assert (fit_df['R2'] > 0.9).all()
