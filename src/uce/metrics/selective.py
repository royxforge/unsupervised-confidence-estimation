"""Selective prediction and confidence/error correlation metrics.

These are the quantities that judge whether an *unsupervised* confidence score
is useful: rank correlation with error, accuracy on the most-confident subset
(the report's "Q4 Accuracy"), and the risk-coverage trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "CorrelationResult",
    "RiskCoverageCurve",
    "accuracy_in_top_quantile",
    "area_under_risk_coverage",
    "error_correlation",
    "pearson_correlation",
    "risk_coverage_curve",
    "spearman_correlation",
]

# The report's Q4 metric inspects the most-confident quarter of the test set.
DEFAULT_QUANTILE = 0.25


@dataclass(frozen=True)
class CorrelationResult:
    """A correlation coefficient plus the evidence behind it."""

    statistic: float
    p_value: float
    n: int

    @property
    def defined(self) -> bool:
        return not np.isnan(self.statistic)


def _correlate(a: np.ndarray, b: np.ndarray, method: str) -> CorrelationResult:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"arrays must have the same shape, got {a.shape} and {b.shape}")
    if a.size < 2:
        raise ValueError("need at least 2 samples for a correlation")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        # scipy raises/warns on constant input; a constant score carries no
        # ranking information, so report it as undefined instead of crashing.
        return CorrelationResult(statistic=float("nan"), p_value=float("nan"), n=a.size)

    if method == "spearman":
        result = stats.spearmanr(a, b)
        statistic, p_value = result.statistic, result.pvalue
    else:
        result = stats.pearsonr(a, b)
        statistic, p_value = result.statistic, result.pvalue
    return CorrelationResult(statistic=float(statistic), p_value=float(p_value), n=a.size)


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> CorrelationResult:
    """Spearman rank correlation (monotone association)."""
    return _correlate(a, b, "spearman")


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> CorrelationResult:
    """Pearson linear correlation."""
    return _correlate(a, b, "pearson")


def error_correlation(confidence: np.ndarray, correct: np.ndarray) -> CorrelationResult:
    """Spearman correlation between confidence and *error*, as the report records it.

    The report's ``spearman_corr`` for a UQ method is negative when confidence
    correctly anticorrelates with error, so this helper keeps that sign
    convention instead of the (more intuitive) confidence-vs-correctness one.
    """
    correct = np.asarray(correct, dtype=bool)
    return spearman_correlation(confidence, (~correct).astype(np.float64))


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Risk (error rate) as a function of the fraction of samples retained."""

    coverage: np.ndarray
    risk: np.ndarray
    retained: np.ndarray

    def risk_at_coverage(self, coverage: float) -> float:
        """Risk of the most-confident ``coverage`` fraction of the data."""
        coverage = float(coverage)
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        index = max(int(np.ceil(coverage * len(self.risk))) - 1, 0)
        return float(self.risk[index])

    @property
    def area(self) -> float:
        """Mean risk across every coverage level (discrete AURC)."""
        return float(self.risk.mean())


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> RiskCoverageCurve:
    """Build the risk-coverage curve by retaining the most confident samples first.

    Args:
        confidence: Score where a higher value means more trustworthy.
        correct: Correctness of each prediction.

    Returns:
        A :class:`RiskCoverageCurve` whose entry ``i`` covers the ``i + 1``
        most confident samples, so ``risk[0]`` is the risk of the single
        best-scoring prediction.
    """
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=bool).ravel()
    if confidence.shape != correct.shape:
        raise ValueError(
            f"confidence and correct must have the same shape, "
            f"got {confidence.shape} and {correct.shape}"
        )
    if confidence.size == 0:
        raise ValueError("need at least one sample")

    order = np.argsort(confidence, kind="mergesort")[::-1]
    sorted_errors = (~correct)[order].astype(np.float64)
    retained = np.arange(1, confidence.size + 1)
    # Risk of the top-k set: cumulative error rate as k grows.
    risk = np.cumsum(sorted_errors) / retained
    coverage = retained / confidence.size
    return RiskCoverageCurve(coverage=coverage, risk=risk, retained=retained)


def area_under_risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> float:
    """AURC: mean risk over all coverage levels; lower is better.

    A perfectly ordered score gives the minimum possible risk at every coverage,
    while an uninformative one gives the base error rate.
    """
    return risk_coverage_curve(confidence, correct).area


def accuracy_in_top_quantile(
    confidence: np.ndarray,
    correct: np.ndarray,
    quantile: float = DEFAULT_QUANTILE,
) -> float:
    """Accuracy (%) on the most-confident ``quantile`` fraction of samples.

    This is the report's "Q4 Accuracy" (``quantile=0.25``): an unsupervised
    score is useful if the subset it ranks highest is substantially more
    accurate than the whole test set.
    """
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=bool).ravel()
    if confidence.shape != correct.shape:
        raise ValueError(
            f"confidence and correct must have the same shape, "
            f"got {confidence.shape} and {correct.shape}"
        )
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")

    n_retained = max(int(np.floor(confidence.size * quantile)), 1)
    top = np.argsort(confidence, kind="mergesort")[::-1][:n_retained]
    return float(correct[top].mean() * 100.0)
