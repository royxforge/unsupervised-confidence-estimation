"""Tests for UCE final_report.txt content and integrity."""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path

import pytest

_MANIFEST_NAME = "artifacts.sha256"

# Only these modules may be reconstructed when reading committed artefacts.
# Anything else (os, subprocess, builtins.eval...) is refused, so a tampered
# artefact cannot execute code during `pytest`.
_ALLOWED_PICKLE_MODULES = ("numpy", "numpy._core", "numpy.core")
# Container-only modules: these reconstruct data, never execute arbitrary code.
_ALLOWED_CONTAINER_MODULES = ("collections",)
_ALLOWED_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "str", "bytes"}


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses globals outside a small allowlist."""

    def find_class(self, module: str, name: str):
        if module == "builtins" and name in _ALLOWED_BUILTINS:
            import builtins

            return getattr(builtins, name)
        if any(module == prefix or module.startswith(prefix + ".") for prefix in _ALLOWED_PICKLE_MODULES):
            return super().find_class(module, name)
        if module in _ALLOWED_CONTAINER_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to unpickle global {module}.{name} from a committed artefact"
        )


def _safe_pickle_load(path: Path):
    """Load a trusted-by-checksum artefact without trusting arbitrary globals."""
    return _RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()


def _manifest_hash(name: str) -> str | None:
    """Return the expected sha256 for *name* from tests/artifacts.sha256."""
    manifest = Path(__file__).with_name(_MANIFEST_NAME)
    if not manifest.exists():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            return parts[0]
    return None


class TestFinalReport:
    """Validate the final_report.txt content."""

    def test_report_exists(self, final_report_path: Path):
        """Report file must exist."""
        assert final_report_path.exists(), "final_report.txt not found"

    def test_report_is_not_empty(self, final_report_path: Path):
        """Report file must not be empty."""
        text = final_report_path.read_text(encoding="utf-8")
        assert len(text) > 500, "Report is too short"

    def test_contains_executive_summary(self, final_report_path: Path):
        text = final_report_path.read_text(encoding="utf-8")
        assert "EXECUTIVE SUMMARY" in text

    def test_contains_model_performance_table(self, final_report_path: Path):
        text = final_report_path.read_text(encoding="utf-8")
        assert "MODEL PERFORMANCE SUMMARY" in text

    def test_contains_baseline_accuracy(self, final_report_path: Path):
        """Baseline row exists with a plausible accuracy (no pinned literal).

        Asserting the literal "91.14" made the test a copy of one stale run:
        any legitimate recompute would fail, while a different (even wrong)
        number would still have to be hand-edited in by whoever runs it.
        """
        import re

        text = final_report_path.read_text(encoding="utf-8")
        assert "BASELINE" in text
        match = re.search(r"^BASELINE\s+(\d{1,3}\.\d+)", text, re.MULTILINE)
        assert match, "BASELINE row with numeric accuracy not found"
        accuracy = float(match.group(1))
        assert 0.0 <= accuracy <= 100.0, f"Baseline accuracy {accuracy} out of range"

    def test_contains_unsupervised_metric_results(self, final_report_path: Path):
        text = final_report_path.read_text(encoding="utf-8")
        assert "UNSUPERVISED METRIC PERFORMANCE" in text
        assert "Spearman Corr" in text
        assert "Q4 Accuracy" in text

    def test_contains_key_findings(self, final_report_path: Path):
        text = final_report_path.read_text(encoding="utf-8")
        assert "KEY FINDINGS" in text
        assert "Best Accuracy" in text
        assert "Best Calibration" in text

    def test_reports_all_four_methods(self, final_report_path: Path):
        """All 4 UQ methods should be present in the report."""
        text = final_report_path.read_text(encoding="utf-8")
        for method in ["BASELINE", "MC_DROPOUT", "EVIDENTIAL", "ENSEMBLE"]:
            assert method in text, f"Missing method: {method}"

    def test_accuracy_range(self, final_report_path: Path):
        """Accuracy values should be between 0 and 100."""
        text = final_report_path.read_text(encoding="utf-8")
        import re
        accuracies = re.findall(r"(\d+\.\d+)%", text)
        for acc in accuracies:
            val = float(acc)
            assert 0 <= val <= 100, f"Accuracy {val} out of range"

    def test_conference_ready_flag(self, final_report_path: Path):
        """Readiness verdict present and well-formed, without pinning YES.

        "CONFERENCE READY" is a self-attested conclusion; asserting YES
        cements a claim this repo cannot recompute (there is no training or
        evaluation code here to regenerate it from).
        """
        import re

        text = final_report_path.read_text(encoding="utf-8")
        assert re.search(r"CONFERENCE READY:\s*(YES|NO)", text), (
            "Missing CONFERENCE READY: YES|NO flag"
        )

    def test_report_has_conclusions(self, final_report_path: Path):
        text = final_report_path.read_text(encoding="utf-8")
        assert "CONCLUSIONS" in text

    def test_novel_contributions_section(self, final_report_path: Path):
        text = final_report_path.read_text(encoding="utf-8")
        assert "NOVEL CONTRIBUTIONS" in text


class TestCheckpointsIntegrity:
    """Validate checkpoint files exist and are valid."""

    def test_checkpoints_dir_exists(self, checkpoints_dir: Path):
        assert checkpoints_dir.exists(), "checkpoints/ directory not found"

    def test_baseline_checkpoint_exists(self, checkpoints_dir: Path):
        assert (checkpoints_dir / "baseline_best.pth").exists()

    def test_mcdropout_checkpoint_exists(self, checkpoints_dir: Path):
        assert (checkpoints_dir / "mcdropout_best.pth").exists()

    def test_evidential_checkpoint_exists(self, checkpoints_dir: Path):
        assert (checkpoints_dir / "evidential_best.pth").exists()

    def test_checkpoints_have_content(self, checkpoints_dir: Path):
        """Checkpoints should be larger than 1MB."""
        for fname in ["baseline_best.pth", "mcdropout_best.pth", "evidential_best.pth"]:
            fpath = checkpoints_dir / fname
            assert fpath.stat().st_size > 1_000_000, f"{fname} is too small"

    def test_ensemble_dir_exists(self, ensemble_dir: Path):
        assert ensemble_dir.exists(), "ensemble_model/ directory not found"

    def test_ensemble_members_exist(self, ensemble_dir: Path):
        """At least 3 ensemble member files should exist."""
        members = sorted(ensemble_dir.glob("ensemble_model_*.pth"))
        assert len(members) >= 3, f"Only {len(members)} ensemble members found"

    def test_ensemble_members_have_content(self, ensemble_dir: Path):
        members = sorted(ensemble_dir.glob("ensemble_model_*.pth"))
        for member in members:
            assert member.stat().st_size > 100_000, f"{member.name} is too small"


class TestRunMetadata:
    """Validate experiment run metadata."""

    def test_runs_dir_exists(self, project_root: Path):
        runs_dir = project_root / "runs"
        assert runs_dir.exists(), "runs/ directory not found"

    def test_meta_json_exists(self, project_root: Path):
        meta_files = list(project_root.glob("runs/meta_*.json"))
        assert len(meta_files) >= 1, "No meta JSON files found in runs/"

    def test_meta_json_valid(self, project_root: Path):
        import json
        meta_files = sorted(project_root.glob("runs/meta_*.json"))
        for meta_file in meta_files:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            assert "seed" in data
            assert "device" in data
            assert "args" in data


class TestEvaluationResults:
    """Validate evaluation_results.pkl."""

    def test_evaluation_results_exist(self, project_root: Path):
        pkl_path = project_root / "evaluation_results.pkl"
        assert pkl_path.exists(), "evaluation_results.pkl not found"

    def test_evaluation_results_valid(self, project_root: Path):
        """Verify the artefact checksum, then load it with a restricted unpickler."""
        pkl_path = project_root / "evaluation_results.pkl"
        expected = _manifest_hash(pkl_path.name)
        assert expected is not None, f"{_MANIFEST_NAME} has no entry for {pkl_path.name}"
        digest = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
        assert digest == expected, "evaluation_results.pkl does not match the committed checksum"
        data = _safe_pickle_load(pkl_path)
        assert data is not None


class TestRestrictedUnpickler:
    """The artefact loader must refuse code-executing globals."""

    def test_rejects_dangerous_globals(self):
        payload = pickle.dumps(__import__("os").system)
        with pytest.raises(pickle.UnpicklingError):
            _RestrictedUnpickler(io.BytesIO(payload)).load()


class TestImages:
    """Validate that generated images exist."""

    def test_images_dir_exists(self, images_dir: Path):
        assert images_dir.exists(), "images/ directory not found"

    def test_comparison_metrics_image_exists(self, images_dir: Path):
        assert (images_dir / "comparison_metrics.png").exists()

    def test_confidence_distributions_exists(self, images_dir: Path):
        assert (images_dir / "confidence_distributions.png").exists()

    def test_ood_roc_curves_exists(self, images_dir: Path):
        assert (images_dir / "ood_roc_curves.png").exists()

    def test_reliability_diagrams_exist(self, images_dir: Path):
        """At least 2 reliability diagrams should exist."""
        diagrams = list(images_dir.glob("*reliability_diagram*"))
        assert len(diagrams) >= 2, f"Only {len(diagrams)} reliability diagrams found"

    def test_training_curves_exist(self, images_dir: Path):
        """At least 2 training curve plots should exist."""
        curves = list(images_dir.glob("*training_curves*"))
        assert len(curves) >= 2, f"Only {len(curves)} training curves found"

    def test_unsupervised_analysis_exist(self, images_dir: Path):
        analysis = list(images_dir.glob("*unsupervised_analysis*"))
        assert len(analysis) >= 2, f"Only {len(analysis)} unsupervised analysis plots found"

    def test_images_not_empty(self, images_dir: Path):
        """All PNG images should be at least 1KB."""
        for img in images_dir.glob("*.png"):
            assert img.stat().st_size > 1_000, f"{img.name} is too small (empty image?)"
