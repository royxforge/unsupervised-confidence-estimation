"""Dirichlet (evidential deep learning) uncertainty helpers.

:class:`uce.models.EvidentialResNet` and ``EvidentialResNet18`` output Dirichlet
concentration parameters ``alpha``; these helpers turn them into the confidence
and uncertainty scores the evaluation pipeline consumes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dirichlet_confidence",
    "dirichlet_probabilities",
    "dirichlet_vacuity",
]


def _as_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2-D (n_samples, n_classes), got shape {alpha.shape}")
    if np.any(alpha <= 0):
        raise ValueError("alpha must be strictly positive (alpha = evidence + 1)")
    return alpha


def dirichlet_probabilities(alpha: np.ndarray) -> np.ndarray:
    """Dirichlet mean ``alpha / sum(alpha)``: the predicted class probabilities."""
    alpha = _as_alpha(alpha)
    return alpha / alpha.sum(axis=1, keepdims=True)


def dirichlet_confidence(alpha: np.ndarray) -> np.ndarray:
    """Confidence score: the largest predicted probability per sample."""
    return dirichlet_probabilities(alpha).max(axis=1)


def dirichlet_vacuity(alpha: np.ndarray) -> np.ndarray:
    """Vacuity ``K / sum(alpha)``: uncertainty from lack of evidence in [0, 1].

    Tends to 1 when the model has almost no evidence (``alpha`` near 1) and to 0
    as evidence grows, which is the evidential analogue of predictive variance.
    """
    alpha = _as_alpha(alpha)
    n_classes = alpha.shape[1]
    return n_classes / alpha.sum(axis=1)
