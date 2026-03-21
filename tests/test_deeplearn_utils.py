from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')

from cetsax.deeplearn import my_seq_nadph as seqmod


def test_read_fasta_and_supervised_table(tmp_path):
    fasta = tmp_path / 'proteins.fasta'
    fasta.write_text('>P1\nMKT\n>P2\nAAA\n')
    seqs = seqmod.read_fasta_to_dict(fasta)
    assert seqs == {'P1': 'MKT', 'P2': 'AAA'}

    fits_df = pd.DataFrame({
        'id': ['P1', 'P1', 'P2', 'P2'],
        'EC50': [1e-4, 2e-4, 1e-2, 1e-2],
        'delta_max': [0.2, 0.18, 0.05, 0.04],
        'R2': [0.95, 0.93, 0.6, 0.5],
    })
    out_csv = tmp_path / 'supervised.csv'
    out = seqmod.build_sequence_supervised_table(fits_df, fasta, out_csv)
    assert {'hit_class', 'label_cls', 'label_reg', 'seq'}.issubset(out.columns)
    assert out_csv.exists()
    assert set(out['label_cls']) == {0, 1}


def test_attention_pooling_and_head_helpers():
    import torch

    pool = seqmod.AttentionPooling(embed_dim=4)
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    pooled = pool(x, mask)
    assert pooled.shape == (2, 4)

    class HeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 2)

    head = seqmod._get_head_module(HeadModel())
    assert isinstance(head, torch.nn.Module)
