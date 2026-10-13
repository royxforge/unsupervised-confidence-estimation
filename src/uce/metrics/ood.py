"""Out-of-distribution detection and predictive-uncertainty metrics.

The AUROC and FPR@95TPR definitions here reproduce the values stored in the
committed ``evaluation_results.pkl`` exactly (see ``tests/test_metrics.py``):
in-distribution samples are the positive class and the confidence score is used
as the ranking statistic, so a *higher* AUROC means better separation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_ranks",
    "detection_error",
    "fpr_at_95_tpr",
    "mutual_information",
    "ood_detection_auroc",
    "predictive_entropy",
]

_FPR95_TARGET = 0.95


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values`` (1-based) with ties receiving their average rank."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    sorted_values = values[order]
    _, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        if count > 1:
            ranks[order[start : start + count]] = start + (count + 1) / 2.0
    return ranks


def ood_detection_auroc(in_dist_scores: np.ndarray, out_dist_scores: np.ndarray) -> float:
    """AUROC for separating in-distribution from OOD samples.

    Computed as the Mann-Whitney U statistic :math:`P(s_{id} > s_{ood})` with
    ties counted as a half, which is exactly the ROC-AUC of the in-distribution
    class. No scikit-learn dependency, so the value cannot drift with a library
    upgrade.

    Args:
        in_dist_scores: Confidence (higher = more in-distribution) for ID data.
        out_dist_scores: Same score for OOD data.

    Returns:
        AUROC in [0, 1]; 0.5 is chance, higher is better.
    """
    in_dist_scores = np.asarray(in_dist_scores, dtype=np.float64).ravel()
    out_dist_scores = np.asarray(out_dist_scores, dtype=np.float64).ravel()
    if in_dist_scores.size == 0 or out_dist_scores.size == 0:
        raise ValueError("both score arrays must be non-empty")

    combined = np.concatenate([in_dist_scores, out_dist_scores])
    ranks = average_ranks(combined)
    n_in = in_dist_scores.size
    rank_sum_in = ranks[:n_in].sum()
    u = rank_sum_in - n_in * (n_in + 1) / 2.0
    return float(u / (n_in * out_dist_scores.size))


def fpr_at_95_tpr(
    in_dist_scores: np.ndarray,
    out_dist_scores: np.ndarray,
    target_tpr: float = _FPR95_TARGET,
) -> float:
    """False-positive rate on OOD data at the ID true-positive rate ``target_tpr``.

    The threshold is the ``1 - target_tpr`` quantile of the in-distribution
    scores; OOD samples at or above it are counted as false positives. Lower is
    better. Matches the report's ``ood_fpr95`` column.
    """
    in_dist_scores = np.asarray(in_dist_scores, dtype=np.float64).ravel()
    out_dist_scores = np.asarray(out_dist_scores, dtype=np.float64).ravel()
    if in_dist_scores.size == 0 or out_dist_scores.size == 0:
        raise ValueError("both score arrays must be non-empty")
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in (0, 1]")

    threshold = np.quantile(in_dist_scores, 1.0 - target_tpr)
    return float((out_dist_scores >= threshold).mean())


def detection_error(in_dist_scores: np.ndarray, out_dist_scores: np.ndarray) -> float:
    """Minimum error over thresholds: ``1 - AUROC`` for a two-class problem."""
    return 1.0 - ood_detection_auroc(in_dist_scores, out_dist_scores)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each predictive distribution.

    Args:
        probs: ``(n_samples, n_classes)`` probabilities.

    Returns:
        ``(n_samples,)`` entropies; 0 for a certain prediction and ``log(K)``
        for a uniform one.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n_samples, n_classes), got shape {probs.shape}")
    total = probs / probs.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(total > 0, total * np.log(total), 0.0)
    return -terms.sum(axis=1)


def mutual_information(mc_probs: np.ndarray) -> np.ndarray:
    """Mutual information between predictions and model parameters (nats).

    ``I = H(E[p]) - E[H(p)]``: the part of the predictive entropy that comes
    from disagreement between stochastic forward passes, i.e. epistemic
    uncertainty. Zero when every pass produces the same distribution.

    Args:
        mc_probs: ``(n_samples, n_passes, n_classes)`` probabilities from MC
            dropout or an ensemble.
    """
    mc_probs = np.asarray(mc_probs, dtype=np.float64)
    if mc_probs.ndim != 3:
        raise ValueError(
            "mc_probs must be 3-D (n_samples, n_passes, n_classes), "
            f"got shape {mc_probs.shape}"
        )
    mean_probs = mc_probs.mean(axis=1)
    total_entropy = predictive_entropy(mean_probs)
    expected_entropy = predictive_entropy(mc_probs.reshape(-1, mc_probs.shape[-1]))
    expected_entropy = expected_entropy.reshape(mc_probs.shape[0], mc_probs.shape[1]).mean(axis=1)
    # Numerical noise can push a perfectly-agreeing model slightly below zero.
    return np.clip(total_entropy - expected_entropy, a_min=0.0, a_max=None)
