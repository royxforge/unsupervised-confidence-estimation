"""Confidence calibration metrics.

Definitions match the values recorded in the committed ``final_report.txt`` /
``evaluation_results.pkl`` exactly (verified by ``tests/test_metrics.py``):
15 equal-width bins over [0, 1], max-softmax confidence, and the correctness of
the predicted class as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_N_BINS = 15

__all__ = [
    "DEFAULT_N_BINS",
    "ReliabilityBins",
    "confidence_from_probs",
    "expected_calibration_error",
    "reliability_bins",
]


def confidence_from_probs(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(confidences, predictions)`` from class probabilities.

    Confidence is the maximum predicted probability, the standard choice when a
    single scalar is needed to rank predictions.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n_samples, n_classes), got shape {probs.shape}")
    predictions = probs.argmax(axis=1)
    confidences = probs[np.arange(probs.shape[0]), predictions]
    return confidences, predictions


def _as_correctness(correct: np.ndarray, n: int) -> np.ndarray:
    correct = np.asarray(correct)
    if correct.shape != (n,):
        raise ValueError(f"correct must have shape ({n},), got {correct.shape}")
    return correct.astype(bool)


@dataclass(frozen=True)
class ReliabilityBins:
    """Per-bin confidence/accuracy data for a reliability diagram and ECE."""

    bin_lower: np.ndarray
    confidences: np.ndarray
    accuracies: np.ndarray
    counts: np.ndarray

    @property
    def gaps(self) -> np.ndarray:
        """Absolute confidence-accuracy gap per bin; ``nan`` where empty."""
        return np.abs(self.accuracies - self.confidences)

    @property
    def expected_calibration_error(self) -> float:
        """Population-weighted mean of the per-bin gaps.

        Empty bins are skipped explicitly: their ``nan`` statistics must not
        propagate into the sum (``0 * nan`` is ``nan``).
        """
        total = int(self.counts.sum())
        if total == 0:
            return float("nan")
        populated = self.counts > 0
        return float((self.counts[populated] * self.gaps[populated]).sum() / total)


def reliability_bins(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> ReliabilityBins:
    """Bin predictions by confidence on an equal-width grid over [0, 1].

    Bin ``i`` covers ``[edge_i, edge_{i+1})``, except the last bin which is
    closed so a confidence of exactly 1.0 is included. Empty bins are reported
    with ``count=0`` and ``nan`` statistics so callers can plot the full grid
    without gaps; they contribute nothing to the ECE.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    correct = _as_correctness(correct, confidences.shape[0])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # A confidence of exactly 1.0 belongs to the final bin.
    indices = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, n_bins - 1)

    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = indices == b
        counts[b] = int(mask.sum())
        if counts[b]:
            bin_confidence[b] = confidences[mask].mean()
            bin_accuracy[b] = correct[mask].mean()

    return ReliabilityBins(
        bin_lower=edges[:-1],
        confidences=bin_confidence,
        accuracies=bin_accuracy,
        counts=counts,
    )


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Expected Calibration Error: mean |accuracy - confidence| over bins.

    Args:
        confidences: Max predicted probability per sample.
        correct: Boolean/0-1 correctness of the predicted class per sample.
        n_bins: Number of equal-width bins (the report uses 15).

    Returns:
        ECE in [0, 1]; lower is better calibrated.
    """
    return reliability_bins(confidences, correct, n_bins=n_bins).expected_calibration_error
