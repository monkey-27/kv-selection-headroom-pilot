from math import comb

import pytest

from kvheadroom.combinatorics import masks_by_cardinality, nested_dp, summarize_utilities


def test_enumerates_all_4096_by_cardinality():
    grouped = masks_by_cardinality(12)
    assert sum(map(len, grouped.values())) == 4096
    assert [len(grouped[k]) for k in range(13)] == [comb(12, k) for k in range(13)]
    assert len({m for masks in grouped.values() for m in masks}) == 4096


def test_nested_dp_finds_best_early_first_order():
    weights = [8.0, 4.0, 2.0, 1.0]
    utility = {mask: sum(weights[i] for i in range(4) if mask & (1 << i)) for mask in range(16)}
    result = nested_dp(utility, 4)
    assert result["block_order"] == [0, 1, 2, 3]
    assert [m.bit_count() for m in result["chain_masks"]] == list(range(5))


def test_exact_random_and_recovery():
    # Utility is additive; the exact random mean at k=1 is (0+1+2)/3 = 1.
    utility = {mask: float(sum(i for i in range(3) if mask & (1 << i))) for mask in range(8)}
    rows = [{"mask": mask, "utility": value} for mask, value in utility.items()]
    summary, _ = summarize_utilities(rows, 3)
    assert summary[1]["u_rand"] == pytest.approx(1.0)
    assert summary[1]["u_oracle"] == pytest.approx(2.0)
    assert summary[1]["headroom"] == pytest.approx(1.0)
    assert summary[1]["rr"] == pytest.approx(0.5)
