import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cetsax.config import DOSE_COLS, ID_COL, COND_COL
from cetsax.models import itdr_model


@pytest.fixture(scope='session')
def doses():
    return np.array(DOSE_COLS, dtype=float)


@pytest.fixture()
def synthetic_raw_df(doses):
    rows = []
    proteins = {
        'P001': {
            'NADPH.r1': (0.95, 1.30, -3.8, 1.2),
            'NADPH.r2': (0.97, 1.28, -3.7, 1.1),
        },
        'P002': {
            'NADPH.r1': (1.20, 0.82, -3.4, 1.3),
            'NADPH.r2': (1.18, 0.85, -3.5, 1.2),
        },
    }
    for pid, conds in proteins.items():
        for cond, params in conds.items():
            y = itdr_model(doses, *params)
            row = {
                ID_COL: pid,
                COND_COL: cond,
                'sumUniPeps': 5,
                'sumPSMs': 30,
                'countNum': 12,
            }
            row.update({col: float(val) for col, val in zip(DOSE_COLS, y)})
            rows.append(row)

    # Add one low-QC row for filtering tests.
    low_qc = rows[0].copy()
    low_qc[ID_COL] = 'LOWQC'
    low_qc[COND_COL] = 'NADPH.r1'
    low_qc['sumUniPeps'] = 1
    low_qc['sumPSMs'] = 2
    low_qc['countNum'] = 1
    rows.append(low_qc)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _patch_joblib_parallel(monkeypatch):
    import cetsax.fit as fit_mod

    class _SequentialParallel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, tasks):
            out = []
            for task in tasks:
                out.append(task() if callable(task) else task)
            return out

    def _sequential_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(fit_mod, 'Parallel', _SequentialParallel)
    monkeypatch.setattr(fit_mod, 'delayed', _sequential_delayed)
    monkeypatch.setattr(fit_mod, 'tqdm', lambda iterable, **kwargs: iterable)
