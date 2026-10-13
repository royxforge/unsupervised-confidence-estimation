"""UCE - Unsupervised Confidence Estimation Package."""

from __future__ import annotations

from uce import metrics, models

__version__ = "0.3.0"
# Only modules that actually exist are advertised: importing a name from
# ``__all__`` must not raise.
__all__ = ["metrics", "models"]