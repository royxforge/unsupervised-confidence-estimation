# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-07-20

### Added

- **Validation test suite**: Added `tests/` directory with 33 structural validation tests covering report content integrity, checkpoint file existence and size verification, ensemble member completeness, run metadata JSON validity, evaluation results pickle file verification, and image asset presence. All tests pass without requiring GPU or model inference.

---

## [1.1.0] - 2026-07-20

### Added

- **Community health files**: Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CONTRIBUTING.md` (contribution guidelines), `SECURITY.md` (vulnerability reporting policy), and `CITATION.cff` (citation metadata). These files establish project governance, community participation guidelines, and academic attribution framework.

---

## [1.0.0] - 2026-07-14

### Added

- Initial complete release of the Unsupervised Confidence Estimation framework.
- Full pipeline implementation in `unsupervised_confidence_estimation.ipynb` encompassing four uncertainty quantification paradigms: Baseline Maximum Softmax Probability (MSP), Monte Carlo Dropout (MC Dropout), Evidential Deep Learning (EDL), and Deep Ensembles.
- Novel unsupervised confidence metric that combines four label-free signals into a unified confidence score: prediction consistency across augmentations, entropy based uncertainty, feature space dispersion, and softmax temperature analysis.
- Pretrained model checkpoints for all four UQ methods stored in `checkpoints/` directory.
- Deep Ensemble member weights (7 ensemble members) stored in `ensemble_model/` directory.
- Comprehensive evaluation artifacts including `evaluation_results.pkl` and `final_report.txt` with accuracy, Expected Calibration Error (ECE), OOD AUROC, inference latency, Spearman correlation, and Q4 accuracy metrics.
- CIFAR-10 (in-distribution) and CIFAR-100 (out-of-distribution) benchmark configuration with 100 training epochs and 10 classes.
- Full visualization suite comprising comparison metrics, confidence distributions, OOD ROC curves, training curves, reliability diagrams, and per-method unsupervised analysis plots.
- Three ablation studies validating the unsupervised metric design: dropout rate sensitivity, ensemble size scaling, and component weight contribution analysis.
- TensorBoard run metadata in `runs/meta_1760506599.json`.
- MIT License documentation.
- Comprehensive `README.md` with project overview, motivation, methodology, quantitative results, installation instructions, usage guidelines, and citation information.

### Changed

- Corrected asset directory structure by relocating 17 visualization image files from `ensemble_model/images/` to the top-level `images/` directory to improve repository navigability.
- Updated repository URL in the citation block within `README.md` from `royxlead` to `royxforge` to reflect the correct GitHub organization.
- Revised author attribution in `README.md` footer from "Founding AI/ML Engineer at Yuga AI" to "Artificial Intelligence Engineer at Accure Inc." with updated profile link.

### Fixed

- Resolved file path inconsistencies where visualization assets were incorrectly nested under `ensemble_model/` rather than residing at the project root level.

[1.2.0]: https://github.com/royxforge/unsupervised-confidence-estimation/releases/tag/v1.2.0
[1.1.0]: https://github.com/royxforge/unsupervised-confidence-estimation/releases/tag/v1.1.0
[1.0.0]: https://github.com/royxforge/unsupervised-confidence-estimation/releases/tag/v1.0.0
