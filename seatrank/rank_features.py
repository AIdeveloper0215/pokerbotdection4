"""Hero seat-rank percentile features."""
from __future__ import annotations

from typing import Any

AGG = {"bet", "raise", "all_in"}


def _seat_metric(actions: list[dict[str, Any]]) -> tuple[float, float]:
    n = max(len(actions), 1)
    kinds = [str(a.get("action_type", "")).lower() for a in actions]
    return (sum(k in AGG for k in kinds) / n, kinds.count("fold") / n)


def hero_rank_row(hands: list[dict[str, Any]]) -> list[float]:
    """Chunk -> mean hero percentile rank on (aggression, fold) + coverage."""
    agg_ranks: list[float] = []
    fold_ranks: list[float] = []
    for hand in hands:
        hero = str(hand.get("metadata", {}).get("hero") or "")
        by_seat: dict[str, list] = {}
        for act in hand.get("actions", []):
            by_seat.setdefault(str(act.get("player") or ""), []).append(act)
        if hero not in by_seat or len(by_seat) < 3:
            continue
        metrics = {seat: _seat_metric(acts) for seat, acts in by_seat.items()}
        seats = list(metrics)
        denominator = len(seats) - 1
        for idx, ranks in ((0, agg_ranks), (1, fold_ranks)):
            hero_value = metrics[hero][idx]
            below = sum(metrics[s][idx] < hero_value for s in seats if s != hero)
            ranks.append(below / denominator)
    cover = len(agg_ranks) / max(len(hands), 1)
    mean = lambda v: sum(v) / len(v) if v else 0.5
    return [mean(agg_ranks), mean(fold_ranks), cover]
