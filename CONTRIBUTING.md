# Contributing to Unsupervised Confidence Estimation

Thank you for your interest in contributing to Unsupervised Confidence
Estimation! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code. Please report unacceptable behavior to royxforge@proton.me.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/unsupervised-confidence-estimation.git
   cd unsupervised-confidence-estimation
   ```
3. Add the upstream repository:
   ```
   git remote add upstream https://github.com/royxforge/unsupervised-confidence-estimation.git
   ```

## Development Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Jupyter Notebook or VS Code with Jupyter support

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn matplotlib seaborn tqdm tensorboard
pip install torch torchvision
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide.
- Use type annotations for function signatures where possible.
- Maximum line length: 88 characters.
- Use descriptive variable names.
- Document all classes and functions with docstrings.

### Imports

Organize imports in the following order:

1. Standard library imports
2. Third-party imports
3. Local application imports

## Testing

### Smoke Test

```bash
export FAST_DEBUG_SUBSET=1
python -c "from unsupervised_confidence_estimation import main_pipeline; main_pipeline(train_models=False, run_ablations=True, num_epochs=1)"
```

### Full Test

Open `unsupervised_confidence_estimation.ipynb` and run all cells to verify
reproducibility. Results should match those in `final_report.txt`.

## Pull Request Process

1. Create a new branch from `main`:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, descriptive commit messages.

3. Run the notebook end-to-end to verify correctness.

4. Push your branch and open a Pull Request on GitHub.

5. In your PR description, include:
   - What the change does
   - Any relevant issue numbers
   - How you tested the change
   - Results or comparison tables if applicable

6. Request review from a maintainer.

## Commit Message Conventions

We follow conventional commit format:

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(metric): add feature space dispersion to unsupervised confidence score
fix(evaluation): correct ECE calculation for OOD samples
docs(readme): update ablation study results
```

## Issue Reporting

### Bug Reports

When filing a bug report, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Environment details (OS, Python version, CUDA version)
- Relevant logs or error messages

### Feature Requests

We welcome feature suggestions! Please include:

- A clear description of the proposed feature
- The motivation or use case
- Any relevant research or references
- Whether you are willing to implement it

Thank you for helping make Unsupervised Confidence Estimation better!
