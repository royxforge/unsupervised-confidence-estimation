"""Tests for ``uce.metrics``.

Two layers of checking:

* analytic cases with hand-computed expected values, so each definition is
  pinned independently of any data;
* recomputation of the committed ``evaluation_results.pkl`` metrics from the
  stored per-sample arrays, which is the only way to confirm the report's
  numbers are reproducible from the code in this repository.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest

from uce.metrics import (
    accuracy_in_top_quantile,
    area_under_risk_coverage,
    confidence_from_probs,
    dirichlet_confidence,
    dirichlet_probabilities,
    dirichlet_vacuity,
    error_correlation,
    evaluate_confidence,
    expected_calibration_error,
    fpr_at_95_tpr,
    mutual_information,
    ood_detection_auroc,
    pearson_correlation,
    predictive_entropy,
    reliability_bins,
    risk_coverage_curve,
    spearman_correlation,
)

_ALLOWED_PICKLE_MODULES = ("numpy", "collections")
_ALLOWED_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "str", "bytes", "int", "float"}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "builtins" and name in _ALLOWED_BUILTINS:
            import builtins

            return getattr(builtins, name)
        if any(module == p or module.startswith(p + ".") for p in _ALLOWED_PICKLE_MODULES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


@pytest.fixture(scope="module")
def results(project_root) -> dict:
    """The committed evaluation artefact, loaded with a data-only unpickler."""
    path = project_root / "evaluation_results.pkl"
    if not path.exists():
        pytest.skip("evaluation_results.pkl not present")
    return _RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()


# --------------------------------------------------------------------------
# Analytic cases
# --------------------------------------------------------------------------


class TestCalibration:
    def test_confidence_from_probs(self):
        probs = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        confidences, predictions = confidence_from_probs(probs)

        assert predictions.tolist() == [1, 0]
        assert confidences == pytest.approx([0.7, 0.6])

    def test_rejects_non_2d_probs(self):
        with pytest.raises(ValueError, match="2-D"):
            confidence_from_probs(np.array([0.5, 0.5]))

    def test_perfectly_calibrated_ece_is_zero(self):
        # Two bins, each sample's confidence equals the observed accuracy.
        confidences = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        correct = np.array([1, 0, 0, 0, 0])  # accuracy 0.2 in the low bin

        assert expected_calibration_error(confidences, correct, n_bins=2) == pytest.approx(0.0)

    def test_ece_is_confidence_accuracy_gap(self):
        confidences = np.array([0.9, 0.9, 0.9, 0.9])
        correct = np.array([1, 0, 0, 0])  # accuracy 0.25

        assert expected_calibration_error(confidences, correct, n_bins=1) == pytest.approx(0.65)

    def test_ece_weights_bins_by_population(self):
        # Bins are [0, 0.5) and [0.5, 1.0]: all four samples fall in the upper
        # bin, where accuracy is 0.75 against a mean confidence of 0.8.
        confidences = np.array([0.5, 0.9, 0.9, 0.9])
        correct = np.array([0, 1, 1, 1])

        assert expected_calibration_error(confidences, correct, n_bins=2) == pytest.approx(0.05)

    def test_ece_leaves_empty_bins_out_of_the_sum(self):
        # Naive 0 * nan arithmetic would make this nan instead of 0.1.
        confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        correct = np.array([1, 1, 1, 1, 1, 1, 1, 0])

        ece = expected_calibration_error(confidences, correct, n_bins=10)

        assert not np.isnan(ece)
        assert ece == pytest.approx(abs(7 / 8 - 0.9))

    def test_confidence_of_one_lands_in_the_last_bin(self):
        bins = reliability_bins(np.array([1.0]), np.array([1]), n_bins=5)

        assert bins.counts[-1] == 1
        assert bins.counts.sum() == 1

    def test_empty_bins_are_reported_with_counts_and_out_of_ece(self):
        bins = reliability_bins(np.array([0.95]), np.array([1]), n_bins=10)

        assert bins.counts[0] == 0
        assert np.isnan(bins.confidences[0])
        assert bins.expected_calibration_error == pytest.approx(0.05)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="correct must have shape"):
            expected_calibration_error(np.array([0.9, 0.8]), np.array([1]))


class TestOodDetection:
    def test_perfectly_separated_scores_give_auroc_one(self):
        assert ood_detection_auroc([0.9, 0.8], [0.2, 0.1]) == pytest.approx(1.0)

    def test_identical_distributions_give_auroc_half(self):
        scores = np.array([0.1, 0.4, 0.6, 0.9])

        assert ood_detection_auroc(scores, scores) == pytest.approx(0.5)

    def test_inverted_scores_give_auroc_zero(self):
        assert ood_detection_auroc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)

    def test_known_mann_whitney_value(self):
        # ID {0.3, 0.5} vs OOD {0.4}: wins are 0.3<0.4 (loss), 0.5>0.4 (win) -> 0.5
        assert ood_detection_auroc([0.3, 0.5], [0.4]) == pytest.approx(0.5)

    def test_ties_count_as_half(self):
        assert ood_detection_auroc([0.5], [0.5]) == pytest.approx(0.5)

    def test_empty_scores_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            ood_detection_auroc([], [0.5])

    def test_fpr95_is_zero_for_separated_scores(self):
        id_scores = np.linspace(0.5, 1.0, 100)
        ood_scores = np.linspace(0.0, 0.4, 100)

        assert fpr_at_95_tpr(id_scores, ood_scores) == pytest.approx(0.0)

    def test_fpr95_is_one_for_inverted_scores(self):
        assert fpr_at_95_tpr([0.1, 0.2], [0.9, 1.0]) == pytest.approx(1.0)

    def test_fpr95_rejects_bad_target(self):
        with pytest.raises(ValueError, match="target_tpr"):
            fpr_at_95_tpr([0.5], [0.5], target_tpr=0.0)


class TestPredictiveUncertainty:
    def test_entropy_of_certain_prediction_is_zero(self):
        assert predictive_entropy(np.array([[1.0, 0.0, 0.0]]))[0] == pytest.approx(0.0)

    def test_entropy_of_uniform_prediction_is_log_k(self):
        probs = np.full((1, 4), 0.25)

        assert predictive_entropy(probs)[0] == pytest.approx(np.log(4))

    def test_entropy_is_invariant_to_unnormalised_input(self):
        probs = np.array([[2.0, 2.0]])

        assert predictive_entropy(probs)[0] == pytest.approx(np.log(2))

    def test_entropy_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            predictive_entropy(np.array([0.5, 0.5]))

    def test_mutual_information_is_zero_when_passes_agree(self):
        identical = np.tile(np.array([[0.7, 0.2, 0.1]]), (5, 2, 1))

        assert mutual_information(identical) == pytest.approx(0.0, abs=1e-12)

    def test_mutual_information_grows_with_disagreement(self):
        agreeing = np.array([[[0.9, 0.1], [0.9, 0.1]]])
        disagreeing = np.array([[[1.0, 0.0], [0.0, 1.0]]])

        assert mutual_information(agreeing)[0] < mutual_information(disagreeing)[0]

    def test_mutual_information_never_negative(self):
        rng = np.random.default_rng(0)
        passes = rng.dirichlet(np.ones(3), size=(20, 4))

        assert (mutual_information(passes) >= 0).all()

    def test_mutual_information_requires_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            mutual_information(np.ones((4, 3)))


class TestSelectivePrediction:
    def test_perfect_ranking_has_minimal_risk(self):
        confidence = np.array([0.9, 0.8, 0.7, 0.6])
        correct = np.array([True, True, False, False])
        curve = risk_coverage_curve(confidence, correct)

        assert curve.risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
        assert curve.coverage[-1] == pytest.approx(1.0)

    def test_reversed_ranking_is_pessimistic(self):
        confidence = np.array([0.6, 0.7, 0.8, 0.9])
        correct = np.array([True, True, False, False])

        curve = risk_coverage_curve(confidence, correct)

        assert curve.risk[0] == pytest.approx(1.0)
        assert curve.risk[-1] == pytest.approx(0.5)

    def test_aurc_is_lower_for_better_ranking(self):
        correct = np.array([True, True, False, False])
        good = area_under_risk_coverage(np.array([0.9, 0.8, 0.7, 0.6]), correct)
        bad = area_under_risk_coverage(np.array([0.6, 0.7, 0.8, 0.9]), correct)

        assert good < bad
        assert good == pytest.approx((0.0 + 0.0 + 1 / 3 + 0.5) / 4)

    def test_risk_at_coverage(self):
        correct = np.array([True, True, False, False])
        curve = risk_coverage_curve(np.array([0.9, 0.8, 0.7, 0.6]), correct)

        assert curve.risk_at_coverage(0.5) == pytest.approx(0.0)
        assert curve.risk_at_coverage(1.0) == pytest.approx(0.5)

    def test_accuracy_in_top_quantile_perfect_ranking(self):
        confidence = np.array([0.9, 0.8, 0.7, 0.6])
        correct = np.array([True, True, False, False])

        assert accuracy_in_top_quantile(confidence, correct, quantile=0.5) == pytest.approx(100.0)

    def test_accuracy_in_top_quantile_keeps_at_least_one_sample(self):
        assert accuracy_in_top_quantile([0.9, 0.1], [True, False], quantile=0.1) == pytest.approx(
            100.0
        )

    def test_accuracy_in_top_quantile_validates_inputs(self):
        with pytest.raises(ValueError, match="same shape"):
            accuracy_in_top_quantile([0.9], [True, False])
        with pytest.raises(ValueError, match="quantile"):
            accuracy_in_top_quantile([0.9], [True], quantile=0.0)

    def test_correlations_on_monotone_data(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 3.0, 5.0, 7.0])

        assert spearman_correlation(a, b).statistic == pytest.approx(1.0)
        assert spearman_correlation(a, b[::-1]).statistic == pytest.approx(-1.0)
        assert pearson_correlation(a, b).statistic == pytest.approx(1.0)

    def test_constant_input_is_undefined_not_a_crash(self):
        result = spearman_correlation(np.ones(4), np.array([1.0, 2.0, 3.0, 4.0]))

        assert not result.defined
        assert np.isnan(result.statistic)

    def test_correlation_needs_two_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            spearman_correlation([1.0], [1.0])

    def test_error_correlation_sign_convention(self):
        confidence = np.array([0.9, 0.8, 0.7, 0.6])
        correct = np.array([True, True, False, False])

        # The report's "spearman_corr" is the confidence-vs-error coefficient:
        # for a useful score it is negative, and it is exactly the negation of
        # the (more intuitive) confidence-vs-correctness coefficient.
        error_coefficient = error_correlation(confidence, correct).statistic

        assert error_coefficient == pytest.approx(
            spearman_correlation(confidence, (~correct).astype(float)).statistic
        )
        assert error_coefficient == pytest.approx(
            -spearman_correlation(confidence, correct).statistic
        )
        assert error_coefficient < 0
        assert spearman_correlation(confidence, correct).statistic > 0


class TestEvidential:
    def test_probabilities_are_the_dirichlet_mean(self):
        alpha = np.array([[3.0, 1.0]])

        np.testing.assert_allclose(dirichlet_probabilities(alpha), [[0.75, 0.25]])

    def test_confidence_is_the_max_probability(self):
        assert dirichlet_confidence(np.array([[3.0, 1.0]]))[0] == pytest.approx(0.75)

    def test_vacuity_is_k_over_alpha_sum(self):
        # K = 2, sum(alpha) = 4 -> 0.5; near-zero evidence -> near 1.
        assert dirichlet_vacuity(np.array([[3.0, 1.0]]))[0] == pytest.approx(0.5)
        assert dirichlet_vacuity(np.array([[1.0, 1.0]]))[0] == pytest.approx(1.0)

    def test_rejects_non_positive_alpha(self):
        with pytest.raises(ValueError, match="strictly positive"):
            dirichlet_probabilities(np.array([[1.0, 0.0]]))

    def test_rejects_non_2d_alpha(self):
        with pytest.raises(ValueError, match="2-D"):
            dirichlet_confidence(np.array([1.0, 2.0]))


class TestEvaluateConfidence:
    def test_aggregate_matches_individual_metrics(self):
        confidence = np.array([0.95, 0.9, 0.6, 0.55, 0.8, 0.7, 0.65, 0.5])
        correct = np.array([True, True, False, False, True, False, True, False])

        report = evaluate_confidence(confidence, correct)

        assert report.n == 8
        assert report.accuracy == pytest.approx(correct.mean() * 100)
        assert report.ece == pytest.approx(expected_calibration_error(confidence, correct))
        assert report.q4_accuracy == pytest.approx(accuracy_in_top_quantile(confidence, correct))
        assert report.aurc == pytest.approx(area_under_risk_coverage(confidence, correct))
        assert report.spearman_corr == pytest.approx(
            error_correlation(confidence, correct).statistic
        )

    def test_as_dict_uses_report_column_names(self):
        data = evaluate_confidence([0.9, 0.1], [True, False]).as_dict()

        assert set(data) == {
            "n",
            "accuracy",
            "ece",
            "spearman_corr",
            "correctness_spearman",
            "aurc",
            "q4_accuracy",
        }

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="at least one sample"):
            evaluate_confidence([], [])


# --------------------------------------------------------------------------
# Verification against the committed artefact
# --------------------------------------------------------------------------


class TestCommittedArtefact:
    """Recompute the report's numbers from the stored arrays.

    These tests are the reason the metrics module exists: they turn the
    committed ``final_report.txt`` into a claim the code can check.
    """

    @pytest.mark.parametrize("model", ["baseline", "mc_dropout", "evidential"])
    def test_accuracy_matches_stored(self, results, model):
        row = results["standard_results"][model]
        correct = row["predictions"] == row["labels"]

        # The artefact stores accuracy as a float32-rounded percentage.
        assert correct.mean() * 100 == pytest.approx(row["accuracy"], abs=1e-4)

    @pytest.mark.parametrize("model", ["baseline", "mc_dropout", "evidential"])
    def test_ece_matches_stored(self, results, model):
        row = results["standard_results"][model]
        confidences = np.asarray(row["id_confidences"], dtype=np.float64)
        correct = row["predictions"] == row["labels"]

        assert expected_calibration_error(confidences, correct, n_bins=15) == pytest.approx(
            row["ece"], abs=5e-4
        )

    @pytest.mark.parametrize("model", ["baseline", "mc_dropout", "evidential"])
    def test_ood_auroc_and_fpr95_match_stored(self, results, model):
        row = results["standard_results"][model]

        assert ood_detection_auroc(row["id_confidences"], row["ood_confidences"]) == pytest.approx(
            row["ood_auroc"], abs=5e-4
        )
        assert fpr_at_95_tpr(row["id_confidences"], row["ood_confidences"]) == pytest.approx(
            row["ood_fpr95"], abs=5e-4
        )

    @pytest.mark.parametrize("model", ["baseline", "mc_dropout", "evidential"])
    def test_error_correlations_match_stored(self, results, model):
        row = results["standard_results"][model]
        confidences = np.asarray(row["id_confidences"], dtype=np.float64)
        correct = row["predictions"] == row["labels"]

        # The report stores both coefficients against the error signal, which
        # is why a well-behaved UQ method shows negative values here.
        assert error_correlation(confidences, correct).statistic == pytest.approx(
            row["spearman_corr"], abs=1e-3
        )
        assert pearson_correlation(confidences, (~correct).astype(float)).statistic == (
            pytest.approx(row["pearson_corr"], abs=1e-3)
        )
        assert row["spearman_corr"] < 0 and row["pearson_corr"] < 0

    @pytest.mark.parametrize(
        ("model", "expected_q4"),
        [("baseline", 99.68), ("mc_dropout", 99.92), ("evidential", 98.60)],
    )
    def test_q4_accuracy_matches_the_report(self, results, model, expected_q4):
        """The report's Q4 column is the top-quartile accuracy of the unsupervised score."""
        unsup = results["unsupervised_results"][model]
        final_confidence = np.asarray(unsup["final_confidence"], dtype=np.float64)
        correct = ~np.asarray(unsup["errors"]).astype(bool)

        assert accuracy_in_top_quantile(final_confidence, correct, quantile=0.25) == (
            pytest.approx(expected_q4, abs=0.01)
        )

    @pytest.mark.parametrize("model", ["baseline", "mc_dropout", "evidential"])
    def test_unsupervised_correlations_match_stored(self, results, model):
        unsup = results["unsupervised_results"][model]
        final_confidence = np.asarray(unsup["final_confidence"], dtype=np.float64)
        correct = ~np.asarray(unsup["errors"]).astype(bool)

        # Unlike the standard section, the unsupervised section reports the
        # score against correctness (positive when the score is useful), and it
        # uses the composite final_confidence rather than the raw softmax one.
        assert spearman_correlation(final_confidence, correct).statistic == pytest.approx(
            unsup["spearman_corr"], abs=1e-3
        )
        assert pearson_correlation(final_confidence, correct).statistic == pytest.approx(
            unsup["pearson_corr"], abs=1e-3
        )
        assert unsup["spearman_corr"] > 0 and unsup["pearson_corr"] > 0

    def test_evaluate_confidence_reproduces_a_report_row(self, results):
        row = results["standard_results"]["baseline"]
        correct = row["predictions"] == row["labels"]

        report = evaluate_confidence(np.asarray(row["id_confidences"], dtype=np.float64), correct)

        assert report.accuracy == pytest.approx(row["accuracy"], abs=1e-4)
        assert report.ece == pytest.approx(row["ece"], abs=5e-4)
        assert report.spearman_corr == pytest.approx(row["spearman_corr"], abs=1e-3)
