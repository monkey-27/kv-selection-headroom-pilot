from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable, Iterator

import numpy as np


def masks_by_cardinality(n_blocks: int) -> dict[int, list[int]]:
    """Enumerate every subset exactly once, grouped by cardinality."""
    out: dict[int, list[int]] = defaultdict(list)
    for k in range(n_blocks + 1):
        for members in combinations(range(n_blocks), k):
            mask = sum(1 << i for i in members)
            out[k].append(mask)
    assert sum(map(len, out.values())) == 1 << n_blocks
    return dict(out)


def members(mask: int, n_blocks: int) -> list[int]:
    return [i for i in range(n_blocks) if mask & (1 << i)]


def batched(values: Iterable[int], batch_size: int) -> Iterator[list[int]]:
    batch: list[int] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def nested_dp(utility_by_mask: dict[int, float], n_blocks: int) -> dict:
    """Best nested chain, maximizing the sum of utility over all budgets.

    A chain contains one subset of each cardinality and can add one block at a
    time. Dynamic programming makes the cross-budget objective explicit rather
    than sorting blocks by an implicitly additive score.
    """
    total = 1 << n_blocks
    score = np.full(total, -np.inf, dtype=np.float64)
    parent = np.full(total, -1, dtype=np.int64)
    score[0] = float(utility_by_mask[0])
    for mask in range(1, total):
        if mask not in utility_by_mask:
            raise KeyError(f"missing utility for mask {mask}")
        best_parent, best_score = -1, -np.inf
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            prev = mask ^ bit
            candidate = score[prev]
            if candidate > best_score:
                best_parent, best_score = prev, candidate
            remaining ^= bit
        parent[mask] = best_parent
        score[mask] = best_score + float(utility_by_mask[mask])

    chain = [total - 1]
    while chain[-1]:
        chain.append(int(parent[chain[-1]]))
    chain.reverse()
    order = []
    for before, after in zip(chain, chain[1:]):
        order.append((after ^ before).bit_length() - 1)
    return {
        "objective": float(score[-1]),
        "chain_masks": chain,
        "block_order": order,
    }


def summarize_utilities(rows: list[dict], n_blocks: int) -> tuple[list[dict], dict]:
    by_mask = {int(r["mask"]): float(r["utility"] ) for r in rows}
    if len(by_mask) != 1 << n_blocks:
        raise ValueError(f"expected {1 << n_blocks} masks, found {len(by_mask)}")
    nested = nested_dp(by_mask, n_blocks)
    nested_at_k = {int(mask).bit_count(): int(mask) for mask in nested["chain_masks"]}
    out = []
    u_empty = by_mask[0]
    for k in range(n_blocks + 1):
        kmasks = [m for m in by_mask if m.bit_count() == k]
        vals = np.asarray([by_mask[m] for m in kmasks], dtype=np.float64)
        oracle_mask = max(kmasks, key=by_mask.__getitem__)
        u_oracle = by_mask[oracle_mask]
        u_rand = float(vals.mean())
        denom = u_oracle - u_empty
        rr = (u_rand - u_empty) / denom if abs(denom) > 1e-12 else None
        chain_mask = nested_at_k[k]
        out.append({
            "k": k,
            "u_empty": u_empty,
            "u_rand": u_rand,
            "u_oracle": u_oracle,
            "oracle_mask": oracle_mask,
            "headroom": u_oracle - u_rand,
            "rr": rr,
            "nested_mask": chain_mask,
            "u_nested": by_mask[chain_mask],
            "nested_regret": u_oracle - by_mask[chain_mask],
        })
    return out, nested
