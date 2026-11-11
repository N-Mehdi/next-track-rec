import numpy as np
import torch

from nexttrack.data.sequences import align, last_inputs, make_windows
from nexttrack.models.sasrec import SASRec


def test_windows_cover_history_from_the_end():
    windows = make_windows([np.arange(1, 8), np.array([5]), np.array([3, 4])], max_len=3, stride=2)
    # length 7, window size 4, stride 2: [4..7], [2..5], [1..3] left padded
    assert windows.tolist() == [[4, 5, 6, 7], [2, 3, 4, 5], [0, 1, 2, 3], [0, 0, 3, 4]]


def test_last_inputs_left_pad():
    inputs = last_inputs([np.array([1, 2, 3, 4]), np.array([9]), np.zeros(0, dtype=np.int64)], 3)
    assert inputs.tolist() == [[2, 3, 4], [0, 0, 9], [0, 0, 0]]


def test_align_follows_wanted_order():
    aligned = align(np.array([7, 3]), [np.array([1]), np.array([2, 2])], np.array([3, 8, 7]))
    assert aligned[0].tolist() == [2, 2]
    assert aligned[1].tolist() == []
    assert aligned[2].tolist() == [1]


def test_sasrec_is_causal_and_ignores_padding():
    torch.manual_seed(0)
    model = SASRec(num_items=20, max_len=5, dim=16, layers=2, heads=2, dropout=0.0).eval()
    a = torch.tensor([[0, 3, 4, 5, 6]])
    b = torch.tensor([[0, 3, 4, 5, 9]])
    with torch.no_grad():
        ha = model.encode(a)
        hb = model.encode(b)
    # Positions before the change are identical: no look-ahead.
    assert torch.allclose(ha[:, :4], hb[:, :4], atol=1e-6)
    assert not torch.allclose(ha[:, 4], hb[:, 4])
    loss = model.loss(a[:, :-1], a[:, 1:], num_negatives=8)
    assert torch.isfinite(loss)
