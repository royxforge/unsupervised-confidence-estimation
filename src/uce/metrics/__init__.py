"""Uncertainty and confidence evaluation metrics.

The repository ships a results artefact and a report, but no evaluation code to
regenerate them from. The functions here close that gap: they recompute every
number in ``final_report.txt`` from stored confidences/labels, and
``tests/test_metrics.py`` checks them against ``evaluation_results.pkl`` so the
definitions cannot silently drift.

Sub-modules:
    calibration: expected calibration error and reliability bins.
    ood: out-of-distribution AUROC/FPR@95 and predictive uncertainty.
    selective: rank correlation, top-quantile accuracy, risk-coverage.
    evidential: Dirichlet confidence and vacuity for the evidential models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .calibration import (
    DEFAULT_N_BINS,
    ReliabilityBins,
    confidence_from_probs,
    expected_calibration_error,
    reliability_bins,
)
from .evidential import dirichlet_confidence, dirichlet_probabilities, dirichlet_vacuity
from .ood import (
    average_ranks,
    detection_error,
    fpr_at_95_tpr,
    mutual_information,
    ood_detection_auroc,
    predictive_entropy,
)
from .selective import (
    DEFAULT_QUANTILE,
    CorrelationResult,
    RiskCoverageCurve,
    accuracy_in_top_quantile,
    area_under_risk_coverage,
    error_correlation,
    pearson_correlation,
    risk_coverage_curve,
    spearman_correlation,
)

__all__ = [
    "DEFAULT_N_BINS",
    "DEFAULT_QUANTILE",
    "ConfidenceReport",
    "CorrelationResult",
    "ReliabilityBins",
    "RiskCoverageCurve",
    "accuracy_in_top_quantile",
    "area_under_risk_coverage",
    "average_ranks",
    "confidence_from_probs",
    "detection_error",
    "dirichlet_confidence",
    "dirichlet_probabilities",
    "dirichlet_vacuity",
    "error_correlation",
    "evaluate_confidence",
    "expected_calibration_error",
    "fpr_at_95_tpr",
    "mutual_information",
    "ood_detection_auroc",
    "pearson_correlation",
    "predictive_entropy",
    "reliability_bins",
    "risk_coverage_curve",
    "spearman_correlation",
]


@dataclass(frozen=True)
class ConfidenceReport:
    """Everything the report table records for one confidence score."""

    n: int
    accuracy: float
    ece: float
    spearman_corr: float
    correctness_spearman: float
    aurc: float
    q4_accuracy: float
    quantile: float
    n_bins: int

    def as_dict(self) -> dict:
        """Row-shaped mapping, using the report's own column names."""
        data = asdict(self)
        data.pop("quantile")
        data.pop("n_bins")
        return data


def evaluate_confidence(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    quantile: float = DEFAULT_QUANTILE,
) -> ConfidenceReport:
    """Score a confidence signal against ground-truth correctness.

    Args:
        confidence: One score per sample; a higher value must mean "more
            trustworthy" (max softmax probability, a negative entropy, a
            composite unsupervised score, ...).
        correct: Boolean or 0/1 correctness of each prediction.
        n_bins: Bin count for the ECE (the report uses 15).
        quantile: Most-confident fraction used for the Q4 accuracy (0.25).

    Returns:
        A :class:`ConfidenceReport` with ``accuracy`` and ``q4_accuracy`` as
        percentages, matching the committed report's units.
    """
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct).astype(bool).ravel()
    if confidence.shape != correct.shape:
        raise ValueError(
            f"confidence and correct must have the same shape, "
            f"got {confidence.shape} and {correct.shape}"
        )
    if confidence.size == 0:
        raise ValueError("need at least one sample")

    return ConfidenceReport(
        n=int(confidence.size),
        accuracy=float(correct.mean() * 100.0),
        ece=expected_calibration_error(confidence, correct, n_bins=n_bins),
        spearman_corr=error_correlation(confidence, correct).statistic,
        correctness_spearman=spearman_correlation(confidence, correct).statistic,
        aurc=area_under_risk_coverage(confidence, correct),
        q4_accuracy=accuracy_in_top_quantile(confidence, correct, quantile=quantile),
        quantile=quantile,
        n_bins=n_bins,
    )
