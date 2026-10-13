"""Tests for the UCE model architectures.

The repository previously had no model tests at all, which is how
``ResNetCIFARWithDropout.forward`` came to return ``None``: MC dropout - the
best-calibrated method in the report - could not produce logits. These tests
cover the forward pass, the MC dropout helpers and the committed checkpoints.
"""

from __future__ import annotations

import pytest
import torch

from uce.metrics import mutual_information
from uce.models import EvidentialResNet18, ResNet18, create_model, load_pretrained

BATCH_SIZE = 2


@pytest.fixture
def images() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, 3, 32, 32)


class TestForwardPass:
    def test_baseline_logits(self, images):
        model = create_model("baseline", depth=20).eval()

        logits = model(images)

        assert logits.shape == (BATCH_SIZE, 10)

    @pytest.mark.parametrize("model_type", ["mc_dropout", "ensemble"])
    def test_dropout_and_ensemble_variants_return_logits(self, images, model_type):
        """Regression guard: this used to return None, breaking MC dropout."""
        model = create_model(model_type, depth=20).eval()

        logits = model(images)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (BATCH_SIZE, 10)

    def test_evidential_returns_positive_alpha(self, images):
        model = create_model("evidential", depth=20).eval()

        alpha = model(images)

        assert alpha.shape == (BATCH_SIZE, 10)
        assert (alpha > 0).all()
        assert torch.allclose(model.get_alpha(images), alpha)
        assert torch.allclose(model.get_evidence(images), alpha - 1, atol=1e-5)

    def test_evidential_proba_is_a_distribution(self, images):
        model = create_model("evidential", depth=20).eval()

        probs = model.predict_proba(images)

        assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)

    def test_get_features_returns_penultimate_activations(self, images):
        model = create_model("baseline", depth=20, pretrained=True).eval()

        features = model.get_features(images)

        # ResNet-18 for CIFAR ends at 256 channels; the feature-space part of the
        # unsupervised score is built on these activations.
        assert features.shape == (BATCH_SIZE, 256)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model("transformer", depth=20)

    @pytest.mark.parametrize("depth", [20, 32])
    def test_depth_selects_block_counts(self, depth):
        model = create_model("baseline", depth=depth)

        blocks_per_layer = len(model.layer1)
        assert blocks_per_layer == {20: 3, 32: 5}[depth]


class TestMcDropout:
    def test_forward_is_stochastic_in_train_mode(self, images):
        model = create_model("mc_dropout", depth=20)
        model.train()

        first = model(images)
        second = model(images)

        assert not torch.allclose(first, second), "dropout should randomise the output"

    def test_mc_dropout_forward_returns_mean_and_variance(self, images):
        model = create_model("mc_dropout", depth=20)

        mean_logits, var_logits = model.mc_dropout_forward(images, n_samples=8)

        assert mean_logits.shape == (BATCH_SIZE, 10)
        assert var_logits.shape == (BATCH_SIZE, 10)
        assert (var_logits >= 0).all()
        assert (var_logits > 0).any(), "stochastic passes must disagree somewhere"

    def test_mc_dropout_forward_restores_model_mode(self, images):
        model = create_model("mc_dropout", depth=20).eval()

        model.mc_dropout_forward(images, n_samples=2)

        assert model.training is False

    def test_mc_dropout_logits_keeps_every_pass(self, images):
        model = create_model("mc_dropout", depth=20)

        logits = model.mc_dropout_logits(images, n_samples=6)

        assert logits.shape == (BATCH_SIZE, 6, 10)
        assert not torch.allclose(logits[:, 0], logits[:, 1])

    def test_mc_dropout_probs_feed_the_uncertainty_metrics(self, images):
        model = create_model("mc_dropout", depth=20)

        probs = model.mc_dropout_probs(images, n_samples=8).numpy()

        assert probs.shape == (BATCH_SIZE, 8, 10)
        assert (probs >= 0).all() and (probs <= 1).all()
        # Probabilities per pass, so the uncertainty metrics can consume them.
        assert probs.sum(axis=2) == pytest.approx(1.0, abs=1e-5)
        assert (mutual_information(probs) >= 0).all()

    def test_mc_dropout_probs_restores_model_mode(self, images):
        model = create_model("mc_dropout", depth=20).eval()

        model.mc_dropout_probs(images, n_samples=2)

        assert model.training is False


class TestCheckpoints:
    """The committed checkpoints are ResNet-18; they must load and run."""

    @pytest.mark.parametrize(
        ("model_type", "filename"),
        [
            ("baseline", "baseline_best.pth"),
            ("mc_dropout", "mcdropout_best.pth"),
            ("evidential", "evidential_best.pth"),
        ],
    )
    def test_pretrained_checkpoint_loads_and_predicts(
        self, checkpoints_dir, model_type, filename, images
    ):
        path = checkpoints_dir / filename
        if not path.exists():
            pytest.skip(f"{filename} not present")

        model = load_pretrained(model_type, str(path), device="cpu")

        output = model(images)

        assert output.shape == (BATCH_SIZE, 10)
        if model_type == "evidential":
            assert (output > 0).all(), "evidential output must be Dirichlet alpha"

    def test_ensemble_members_are_resnet18(self, ensemble_dir, images):
        members = sorted(ensemble_dir.glob("ensemble_model_*.pth"))
        if not members:
            pytest.skip("no ensemble members present")

        model = create_model("baseline", pretrained=True)
        state = torch.load(members[0], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        assert model(images).shape == (BATCH_SIZE, 10)

    def test_evidential_resnet18_has_evidence_head(self):
        model = create_model("evidential", pretrained=True, num_classes=10)

        assert isinstance(model, EvidentialResNet18)
        assert hasattr(model, "evidence_head")
        assert not hasattr(model, "fc")

    def test_baseline_pretrained_is_resnet18(self):
        model = create_model("baseline", pretrained=True)

        assert isinstance(model, ResNet18)
