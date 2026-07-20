"""Shared fixtures for UCE tests."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def final_report_path(project_root: Path) -> Path:
    return project_root / "final_report.txt"


@pytest.fixture(scope="session")
def checkpoints_dir(project_root: Path) -> Path:
    return project_root / "checkpoints"


@pytest.fixture(scope="session")
def ensemble_dir(project_root: Path) -> Path:
    return project_root / "ensemble_model"


@pytest.fixture(scope="session")
def images_dir(project_root: Path) -> Path:
    return project_root / "images"


@pytest.fixture(scope="session")
def expected_models() -> list[str]:
    return ["baseline", "mcdropout", "evidential"]


@pytest.fixture(scope="session")
def expected_ensemble_members() -> int:
    return 7
