"""Train and serve the ExtraTrees/RandomForest soft vote."""
from __future__ import annotations

import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier

from .rank_features import hero_rank_row


def build() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("et", ExtraTreesClassifier(n_estimators=400, random_state=3)),
            ("rf", RandomForestClassifier(n_estimators=400, random_state=3)),
        ],
        voting="soft",
    )


def fit(records: list[dict], out_path: str) -> None:
    x = [hero_rank_row(r["hands"]) for r in records]
    y = [int(r["is_bot"]) for r in records]
    model = build()
    model.fit(x, y)
    joblib.dump(model, out_path)


def score(model_path: str, chunks: list[list[dict]]) -> list[float]:
    model = joblib.load(model_path)
    rows = [hero_rank_row(c) for c in chunks]
    return [float(p) for p in model.predict_proba(rows)[:, 1]]
