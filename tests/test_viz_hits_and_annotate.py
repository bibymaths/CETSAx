import importlib
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pytest

from cetsax.config import COND_COL, ID_COL
from cetsax.dataio import apply_basic_qc
from cetsax.fit import fit_all_proteins
from cetsax.viz_hits import (
    build_hits_table,
    classify_hit,
    plot_ec50_replicates,
    run_hit_calling_and_plots,
)


def test_viz_hits_pipeline_writes_outputs(tmp_path, synthetic_raw_df):
    fit_df = fit_all_proteins(apply_basic_qc(synthetic_raw_df))
    row = pd.Series({'EC50': 1e-4, 'delta_max': 0.2, 'R2': 0.95})
    assert classify_hit(row) == 'strong'

    hits_table = build_hits_table(fit_df)
    assert {'dominant_class', 'class_counts', 'EC50_median'}.issubset(hits_table.columns)

    rep = plot_ec50_replicates(fit_df, cond_r1='NADPH.r1', cond_r2='NADPH.r2')
    assert rep is not None

    outputs = run_hit_calling_and_plots(fit_df, tmp_path)
    assert 'hits_table' in outputs
    assert Path(tmp_path / 'cetsa_hits_ranked.csv').exists()
    assert Path(outputs['ec50_vs_delta']).exists()
    assert Path(outputs['r2_vs_delta']).exists()
    assert Path(outputs['ec50_vs_r2']).exists()


def test_annotate_utilities(tmp_path, monkeypatch):
    # Inject a tiny mygene stub so the module can import even if mygene is absent.
    if 'mygene' not in sys.modules:
        fake = types.ModuleType('mygene')
        class FakeMG:
            def querymany(self, *args, **kwargs):
                return []
        fake.MyGeneInfo = FakeMG
        sys.modules['mygene'] = fake

    annotate = importlib.import_module('cetsax.annotate')

    fits_csv = tmp_path / 'fits.csv'
    pd.DataFrame({'id': ['P1', 'P2', None]}).to_csv(fits_csv, index=False)
    assert annotate.get_unique_ids(fits_csv) == ['P1', 'P2']
    assert annotate.strip_isoform_suffix('O00231-2') == 'O00231'
    assert annotate.strip_isoform_suffix('P12345') == 'P12345'

    class DummyResponse:
        def __init__(self, status_code=200, text='>sp|P1|desc\nMPEP'):
            self.status_code = status_code
            self.text = text

    monkeypatch.setattr(annotate.requests, 'get', lambda url, timeout=10.0: DummyResponse())
    fasta = annotate.fetch_uniprot_fasta('P1')
    assert fasta.startswith('>')

    result = annotate.fetch_fastas_parallel(['P1', 'P2'], max_workers=2)
    assert set(result) == {'P1', 'P2'}

    out_fasta = tmp_path / 'seqs.fasta'
    annot_df = pd.DataFrame({'id': ['P1-2'], 'uniprot': ['P1']})
    annotate.write_fastas_with_ids(annot_df, {'P1': '>sp|P1|x\nAAAA'}, out_fasta)
    text = out_fasta.read_text()
    assert text.startswith('>P1-2\nAAAA')
