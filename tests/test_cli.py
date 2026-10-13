"""Tests for the ``uce`` CLI (``python -m uce.cli``)."""

from __future__ import annotations

import io
import json
import pickle

import numpy as np
import pytest

from uce import cli


@pytest.fixture
def results_path(project_root):
    path = project_root / "evaluation_results.pkl"
    if not path.exists():
        pytest.skip("evaluation_results.pkl not present")
    return path


class TestReportCommand:
    def test_report_prints_recomputed_tables(self, results_path, capsys):
        exit_code = cli.main(["report", "--results", str(results_path)])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "MODEL PERFORMANCE SUMMARY" in out
        assert "UNSUPERVISED METRIC PERFORMANCE" in out
        assert "baseline" in out
        assert "Q4 Accuracy" in out

    def test_report_accuracy_column_matches_the_report_file(self, results_path, capsys):
        cli.main(["report", "--results", str(results_path)])

        out = capsys.readouterr().out
        assert "91.1400" in out  # baseline accuracy, recomputed from raw arrays

    def test_report_json_is_machine_readable(self, results_path, capsys):
        exit_code = cli.main(["report", "--results", str(results_path), "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert len(payload["sha256"]) == 64
        baseline = payload["models"]["baseline"]
        assert baseline["accuracy"] == pytest.approx(91.14, abs=1e-4)
        assert 0.0 <= baseline["ece"] <= 1.0
        assert 0.5 < baseline["ood_auroc"] < 1.0

    def test_missing_artefact_fails(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cli.main(["report", "--results", str(tmp_path / "nope.pkl")])


class TestVerifyCommand:
    def test_verify_passes_on_the_committed_artefact(self, results_path, capsys):
        exit_code = cli.main(["verify", "--results", str(results_path)])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "consistent" in out
        assert "0 mismatch" in out

    def test_verify_detects_tampered_metrics(self, results_path, tmp_path, capsys):
        """A report that no longer follows from the arrays must be rejected."""
        data = cli.load_results(results_path)
        # Reassign as a plain dict copy so the metric values can be altered.
        tampered = dict(data)
        tampered["standard_results"] = {
            model: dict(row) for model, row in data["standard_results"].items()
        }
        tampered["standard_results"]["baseline"]["ece"] = 0.99
        path = tmp_path / "tampered.pkl"
        path.write_bytes(pickle.dumps(tampered))

        exit_code = cli.main(["verify", "--results", str(path)])

        out = capsys.readouterr().out
        assert exit_code == 1
        assert "MISMATCH baseline.ece" in out

    def test_verify_tolerance_is_configurable(self, results_path, tmp_path, capsys):
        """Float32 storage noise is forgiven by the default, a tight atol is not."""
        data = cli.load_results(results_path)
        tampered = dict(data)
        tampered["standard_results"] = {
            model: dict(row) for model, row in data["standard_results"].items()
        }
        stored = tampered["standard_results"]["baseline"]["ece"]
        tampered["standard_results"]["baseline"]["ece"] = stored + 1e-4
        path = tmp_path / "slightly-off.pkl"
        path.write_bytes(pickle.dumps(tampered))

        assert cli.main(["verify", "--results", str(path)]) == 0  # inside default atol
        assert cli.main(["verify", "--results", str(path), "--atol", "1e-6"]) == 1
        capsys.readouterr()


class TestArtefactSafety:
    def test_loader_refuses_code_executing_globals(self):
        payload = pickle.dumps(__import__("os").system)

        with pytest.raises(pickle.UnpicklingError):
            cli._RestrictedUnpickler(io.BytesIO(payload)).load()

    def test_loader_reads_plain_data(self):
        payload = pickle.dumps({"a": [1, 2, 3]})

        assert cli._RestrictedUnpickler(io.BytesIO(payload)).load() == {"a": [1, 2, 3]}

    def test_file_sha256_matches_hashlib(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(b"uce")

        import hashlib

        assert cli.file_sha256(path) == hashlib.sha256(b"uce").hexdigest()


class TestModelRows:
    def test_rows_are_computed_from_arrays_not_copied(self):
        """Metrics must follow from the stored confidences/labels."""
        confidences = np.array([0.95, 0.9, 0.85, 0.4])
        predictions = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 3])  # third sample is wrong
        results = {
            "standard_results": {
                "toy": {
                    "accuracy": 999.0,  # deliberately wrong stored value
                    "ece": 123.0,
                    "spearman_corr": 0.0,
                    "pearson_corr": 0.0,
                    "id_confidences": confidences,
                    "predictions": predictions,
                    "labels": labels,
                }
            }
        }

        rows = cli._model_rows(results)

        assert rows["toy"]["accuracy"] == pytest.approx(75.0)
        assert rows["toy"]["ece"] != 123.0
        assert "ood_auroc" not in rows["toy"]

    def test_unsupervised_section_adds_q4_accuracy(self):
        results = {
            "standard_results": {
                "toy": {
                    "id_confidences": np.array([0.9, 0.8, 0.7, 0.6]),
                    "predictions": np.array([0, 1, 2, 3]),
                    "labels": np.array([0, 1, 2, 0]),
                }
            },
            "unsupervised_results": {
                "toy": {
                    "spearman_corr": 0.5,
                    "final_confidence": np.array([0.9, 0.8, 0.7, 0.6]),
                    "errors": np.array([0, 0, 0, 1]),
                }
            },
        }

        rows = cli._model_rows(results)

        assert rows["toy"]["q4_accuracy"] == pytest.approx(100.0)
        assert "unsupervised_spearman" in rows["toy"]
