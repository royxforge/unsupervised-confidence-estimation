"""UCE - Unsupervised Confidence Estimation Package."""

from __future__ import annotations

from uce import metrics

__version__ = "0.3.0"
# Only modules that actually exist are advertised: importing a name from
# ``__all__`` must not raise.
__all__ = ["metrics", "models"]


def __getattr__(name: str):
    """Import ``models`` lazily.

    ``uce.models`` imports torch to build the ResNet/evidential networks. The
    metrics path (``python -m uce.cli report``) never touches a model, so an
    eager import here forced every user of the artefact to install ~2.5 GB of
    torch first. Resolving the submodule on first access keeps
    ``from uce import models`` working without making torch a hard requirement
    for recomputing the report.
    """
    if name == "models":
        from uce import models

        return models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")