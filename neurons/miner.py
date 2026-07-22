"""Poker44 miner: seatrank-forest with a heuristic fallback."""

import os
from collections import Counter
from pathlib import Path

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """Score each chunk with the trained contrast GBM when an artifact is
    present, otherwise with a deterministic behavioral heuristic."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "seatrank-forest",
                "model_version": "3.1.0",
                "framework": "scikit-learn",
                "license": "MIT",
                "repo_url": "https://github.com/AIdeveloper0215/pokerbotdection4",
                "notes": "Seat-rank percentile features + ExtraTrees/RandomForest vote.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained only on the public Poker44 training-benchmark "
                    "releases; contrast features are computed at request time."
                ),
                "training_data_sources": ["poker44-training-benchmark"],
                "private_data_attestation": (
                    "Does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._detector = self._load_detector(repo_root)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    def _load_detector(self, repo_root: Path):
        path = os.getenv("MODEL_PATH", str(repo_root / "models" / "contrast_gbm.joblib"))
        if not os.path.exists(path):
            bt.logging.info("No trained artifact found; using heuristic fallback.")
            return None
        try:
            from seatrank.scorer import Scorer as ContrastScorer

            det = ContrastScorer(path)
            bt.logging.info(f"Loaded seatrank-forest from {path}")
            return det
        except Exception as exc:  # pragma: no cover
            bt.logging.warning(f"model load failed ({exc}); heuristic fallback.")
            return None

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(f"Manifest digest={self.manifest_digest}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        scores = None
        if self._detector is not None and chunks:
            try:
                scores = self._detector.score_chunks(chunks)
            except Exception as exc:  # pragma: no cover
                bt.logging.warning(f"model scoring failed ({exc}); heuristic.")
                scores = None
        if scores is None:
            scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = [float(s) for s in scores]
        synapse.predictions = [s >= 0.5 for s in synapse.risk_scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Scored {len(chunks)} chunks (model={self._detector is not None}).")
        return synapse

    # ------------------------------------------------------------------ heuristic
    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        counts = Counter(str(a.get("action_type", "")).lower() for a in actions)
        n = max(sum(counts.values()), 1)
        agg = (counts.get("bet", 0) + counts.get("raise", 0) + counts.get("all_in", 0)) / n
        passive = (counts.get("call", 0) + counts.get("check", 0)) / n
        return cls._clamp01(0.5 + 0.4 * (agg - passive))

    def score_chunk(self, chunk: list) -> float:
        hands = chunk or []
        if not hands:
            return 0.5
        return self._clamp01(sum(self._score_hand(h) for h in hands) / len(hands))


if __name__ == "__main__":
    with Miner() as miner:
        import time

        while True:
            time.sleep(5 * 60)
