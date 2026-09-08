import torch

from kvheadroom.mechanism import _stable_seed, _summarize


def test_stable_seed_is_repeatable_and_trace_specific():
    assert _stable_seed(7, "trace-a") == _stable_seed(7, "trace-a")
    assert _stable_seed(7, "trace-a") != _stable_seed(7, "trace-b")


def test_landscape_summary_is_exact():
    rows = [{"mask": 0, "k": 0, "utility": -3.0}]
    rows += [
        {"mask": 3, "k": 2, "utility": -1.0},
        {"mask": 5, "k": 2, "utility": -2.0},
        {"mask": 6, "k": 2, "utility": -1.5},
    ]
    got = _summarize(rows)
    assert got["u_rand"] == -1.5
    assert got["u_oracle"] == -1.0
    assert got["h2"] == .5
    assert got["rr2"] == .75


def test_continuation_logit_slice_alignment():
    # For depth d, logit d predicts continuation[d]; the final selected logit
    # predicts continuation[d + target - 1].
    d, target = 16, 64
    tokens = [999] + list(range(d + target - 1))
    assert len(tokens) == d + target
    labels = list(range(d, d + target))
    assert labels[0] == d and labels[-1] == d + target - 1
