"""Serve chunk bot-risk scores from the trained seat-rank forest vote."""
from __future__ import annotations

from typing import List, Sequence

import joblib

from .rank_features import hero_rank_row


class Scorer:
    """Load the ExtraTrees/RandomForest soft-vote model and score chunks."""

    def __init__(self, model_path: str) -> None:
        self.model = joblib.load(model_path)

    def score_chunks(self, chunks: Sequence[Sequence[dict]]) -> List[float]:
        if not chunks:
            return []
        rows = [hero_rank_row(chunk) for chunk in chunks]
        return [float(p) for p in self.model.predict_proba(rows)[:, 1]]
