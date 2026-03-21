# Telco Customer Churn Prediction

[![CI](https://github.com/omarpiro/churn-ml-decision/actions/workflows/ci.yml/badge.svg)](https://github.com/omarpiro/churn-ml-decision/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Coverage 89%](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

> **I built a production-ready ML pipeline that optimizes for business value, not just accuracy. It uses Net_Value as the selection metric, a JSON-backed model registry with promote/rollback, two-tier quality gates, KS-test drift detection, and a strict CI pipeline with 89% test coverage.**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Results & Metrics](#results--metrics)
- [Challenges Solved](#challenges-solved)
- [Built By Me](#built-by-me)
- [Production Mindset](#production-mindset)
- [Tech Stack](#tech-stack)
- [Lessons Learned](#lessons-learned)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Documentation](#documentation)

---

## Problem Statement

### The Business Challenge

Telco companies lose millions annually to customer churn. The naive approach - "predict who will churn and contact everyone" - **fails because**:

| Challenge | Why It's Hard |
|-----------|---------------|
| **False positives are expensive** | Contacting non-churners wastes $50/contact |
| **Accuracy is misleading** | A model predicting "nobody churns" gets 73% accuracy on imbalanced data |
| **Default thresholds are arbitrary** | Using 0.5 ignores the cost asymmetry between FP and FN |
| **Statistical metrics miss the point** | Maximizing F1 doesn't maximize profit |

### The Solution

This pipeline reframes churn prediction as an **optimization problem**:

```
Net_Value = (True_Positives x Retained_Value) - (Total_Flagged x Contact_Cost)

Where:
- Retained_Value = CLV ($2,000) x Success_Rate (30%) = $600
- Contact_Cost = $50
```

Instead of maximizing accuracy, we **maximize Net_Value** subject to quality constraints (Recall >= 70%, Precision >= 50%), then select the optimal threshold.

**Result**: Threshold shifted from 0.50 to **0.25**, increasing estimated business value by **40%**.

---

## Architecture

```
                                 ┌─────────────────────────────────────────────────┐
                                 │              config/default.yaml                │
                                 │   (Business params, quality gates, features)    │
                                 └─────────────────────┬───────────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
                    ▼                                  ▼                                  ▼
┌───────────────────────────┐    ┌───────────────────────────────┐    ┌───────────────────────────┐
│       PREPARE STAGE       │    │         TRAIN STAGE           │    │       EVALUATE STAGE      │
│  ─────────────────────────│    │  ─────────────────────────────│    │  ─────────────────────────│
│  - Data validation        │    │  - Multi-candidate training   │    │  - Threshold grid search  │
│  - 14 engineered features │───▶│  - ROC-AUC model selection   │───▶│  - Net_Value optimization │
│  - Train/val/test splits  │    │  - Registry registration      │    │  - Quality gate checks    │
│  - Drift reference export │    │  - MLflow experiment logging  │    │  - Test set evaluation    │
└───────────────────────────┘    └───────────────────────────────┘    └───────────────────────────┘
                                                       │
                                                       ▼
                                 ┌───────────────────────────────┐
                                 │        PREDICT STAGE          │
                                 │  ─────────────────────────────│
                                 │  - Production model from      │
                                 │    registry (strict mode)     │
                                 │  - Batch CSV inference        │
                                 │  - Drift detection on input   │
                                 └───────────────────────────────┘
```

### Artifact Flow

| Stage | Input | Output |
|-------|-------|--------|
| Prepare | Raw CSV (7,043 rows) | `preprocessor.joblib`, train/val/test `.npy`, `drift_reference.json` |
| Train | Processed arrays | `model.joblib`, `registry.json` entry, `train_summary.json` |
| Evaluate | Model + val/test arrays | `threshold_analysis_val.csv`, `final_test_results.csv` |
| Predict | New CSV + production model | `predictions.csv` |

---

## Key Features

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| **Business Decision Framework** | Net_Value = TP x $600 - Flagged x $50 | Optimizes for ROI, not accuracy |
| **Threshold Optimization** | Grid search 0.20-0.85, step 0.05 | Finds optimal threshold (0.25) instead of defaulting to 0.5 |
| **Two-Level Precision Gates** | Selection: 0.50, Test gate: 0.45 | Absorbs validation-to-test variance |
| **JSON Model Registry** | Timestamped versions, promote/rollback | Production-safe model management without external services |
| **14 Engineered Features** | Spend ratios, risk indicators, service adoption | Domain-driven features that improve signal |
| **KS-Test Drift Detection** | Statistical monitoring against training distribution | Catches data distribution shifts before they impact predictions |
| **Config-Driven Pipeline** | Single YAML controls entire system | Environment overrides via `CHURN__*` env vars |
| **Strict Mode** | `--strict` flag fails fast on issues | No silent fallbacks in production |
| **DVC Pipeline** | Declarative stage dependencies | Reproducible, cached pipeline runs |
| **MLflow Tracking** | Metrics, params, artifacts per run | Experiment history and comparison |

---

## Results & Metrics

### Current Production Model

| Metric | Value | Quality Gate | Status |
|--------|-------|--------------|--------|
| **ROC-AUC** | 0.838 | >= 0.83 | PASS |
| **Recall** | 81.0% | >= 70% | PASS |
| **Precision** | 50.4% | >= 45% | PASS |
| **F1 Score** | 0.622 | - | - |
| **Optimal Threshold** | 0.25 | - | - |
| **Net Value** | **$151,750** | - | - |

> Model: `logistic_regression-v20260207134143` (L1/saga regularization)

### Threshold Analysis

```
Threshold: 0.25 (selected via Net_Value optimization)
├── True Positives:  303 churners caught
├── False Positives: 298 non-churners contacted
├── Total Flagged:   601 customers
├── Recall:          81% of churners identified
└── Net Value:       $151,750 estimated retention value
```

### Quality Gate Design

| Gate | Selection (Validation) | Final (Test) | Rationale |
|------|------------------------|--------------|-----------|
| Recall | >= 0.70 | >= 0.70 | Catch at least 70% of churners |
| Precision | >= 0.50 | >= 0.45 | 5% relaxation absorbs val→test variance |
| ROC-AUC | - | >= 0.83 | Overall model quality |

---

## Challenges Solved

Real engineering problems encountered and resolved during development:

### 1. Business vs. Statistical Optimization

**Problem:** Standard ML metrics (accuracy, F1) don't account for asymmetric costs. A high-precision model that misses churners costs more than a lower-precision model that catches them.

**Solution:** Implemented `Net_Value` as the selection metric:
```python
net_value = true_positives * retained_value - total_flagged * contact_cost
```
This shifted optimal threshold from 0.50 to 0.25, increasing business value by 40%.

### 2. Validation-to-Test Variance

**Problem:** Models passing validation precision (0.52) would sometimes fail test precision (0.48) due to normal statistical variance, causing spurious deployment failures.

**Solution:** Two-tier constraint system:
- **Selection constraint** (validation): Precision >= 0.50 (strict during optimization)
- **Quality gate** (test): Precision >= 0.45 (relaxed by 10% to absorb variance)

### 3. Registry Path Normalization

**Problem:** Legacy registry entries contained absolute paths that broke when the project was moved or cloned. Registry would point to `/Users/old_machine/models/...` instead of relative paths.

**Solution:** `ModelRegistry._normalize_model_path()` converts all paths to project-relative POSIX format on read, auto-migrating legacy entries without manual intervention.

### 4. Silent Fallback Danger

**Problem:** In early versions, `predict.py` would silently fall back to `best_model.joblib` if the registry model was missing, potentially using an untested model in production.

**Solution:** `--strict` mode that fails explicitly instead of falling back. Production deployments require `--strict` flag, while development allows graceful degradation with logged warnings.

### 5. Threshold Selection Documentation

**Problem:** Why 0.25? Without documentation, future maintainers would question or change the threshold without understanding the business rationale.

**Solution:** Created `docs/THRESHOLD_ANALYSIS_NOTE.md` documenting:
- The optimization objective (Net_Value)
- Constraint rationale (recall/precision floors)
- Why 0.25 outperforms default 0.50 by ~$60K in estimated value

---

## Built By Me

This is a **solo project** demonstrating end-to-end ML engineering capabilities:

| Capability | Evidence |
|------------|----------|
| **ML Pipeline Design** | 4-stage DVC pipeline with proper stage dependencies |
| **Feature Engineering** | 14 domain-driven features (spend ratios, risk indicators, tenure interactions) |
| **Business Alignment** | Net_Value optimization instead of accuracy chasing |
| **Production Reliability** | Quality gates, strict mode, graceful degradation |
| **Model Management** | JSON registry with promote/rollback, no silent fallbacks |
| **Monitoring** | KS-test drift detection with statistical thresholds |
| **Configuration** | Pydantic v2 validation, environment variable overrides |
| **Testing** | 89% coverage across 18 test files (~3,500 lines of tests) |
| **CI/CD** | GitHub Actions running lint + full pipeline + tests |
| **Documentation** | Production guide, threshold analysis, deployment checklist |

---

## Production Mindset

This project was built with production deployment in mind:

- [x] **No silent failures** - `--strict` mode for production, explicit errors over silent fallbacks
- [x] **Config validation** - Pydantic v2 with `extra="forbid"` catches typos immediately
- [x] **Environment overrides** - `CHURN__BUSINESS__CLV=3000` without code changes
- [x] **Retry logic** - `tenacity` decorators on model loading with exponential backoff
- [x] **Structured logging** - Contextual logs with `extra={}` metadata
- [x] **Artifact versioning** - Timestamped model files (`model_v2_20260207134143.joblib`)
- [x] **Quality gates** - Automated pass/fail on ROC-AUC, recall, precision
- [x] **Drift detection** - Statistical monitoring catches distribution shifts
- [x] **Health checks** - `churn-health-check` validates production readiness
- [x] **Rollback capability** - `churn-model-rollback` reverts to previous production model
- [x] **Canonical aliases** - `best_model.joblib` for DVC/notebooks, registry for production

---

## Tech Stack

| Category | Technology | Why This Choice |
|----------|------------|-----------------|
| **ML Framework** | scikit-learn 1.2+ | Stable, interpretable LogisticRegression with L1 sparsity |
| **Optional Models** | XGBoost, LightGBM | Configurable candidates (easy to enable) |
| **Pipeline** | DVC | Declarative stages, dependency tracking, cached runs |
| **Experiment Tracking** | MLflow | Metrics/params/artifacts logging (filesystem backend) |
| **Config Validation** | Pydantic v2 | Type-safe config with `extra="forbid"` |
| **Retry Logic** | Tenacity | Exponential backoff on transient failures |
| **Data Processing** | pandas, numpy | Standard data manipulation stack |
| **Statistical Tests** | scipy (KS-test) | Drift detection via Kolmogorov-Smirnov |
| **Testing** | pytest, pytest-cov | 89% coverage with `--cov-fail-under=88` CI gate |
| **Linting** | Ruff | Fast Python linter (replaces flake8 + isort) |
| **CI/CD** | GitHub Actions | Lint -> Pipeline -> Tests on every push |

---

## Lessons Learned

### 1. Threshold Selection is a Business Decision
The default 0.5 threshold assumes equal costs for false positives and false negatives. In churn prediction, missing a churner (lost CLV) costs more than contacting a non-churner (wasted outreach). **Always** let business costs drive threshold selection.

### 2. Two-Stage Validation Prevents Brittle Pipelines
Having identical precision requirements for selection (validation) and quality gates (test) causes spurious failures. A 5-10% relaxation for the final gate absorbs natural variance without compromising quality.

### 3. Silent Fallbacks Are Production Hazards
Early versions fell back to `best_model.joblib` when registry lookup failed. This meant production could silently use a different model than expected. Explicit `--strict` mode is essential for production safety.

### 4. Model Registries Need Path Normalization
Absolute paths break when projects move between machines. Registry entries should always use project-relative paths, with automatic migration for legacy formats.

### 5. Feature Engineering Beats Model Complexity
The winning model is LogisticRegression (L1), not XGBoost. Well-engineered features (`tenure_x_contract`, `is_mtm_fiber`, `overpay_indicator`) provide more lift than switching to complex models.

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/omarpiro/churn-ml-decision.git
cd churn-ml-decision
pip install -e ".[dev,ml,ops]"

# Verify installation
churn-validate-config
```

### Run Full Pipeline

```bash
# Prepare data (validation, feature engineering, preprocessing)
churn-prepare --config config/default.yaml --strict

# Train model (multi-candidate, register best)
churn-train --config config/default.yaml --strict

# Evaluate (threshold optimization, quality gates)
churn-evaluate --config config/default.yaml --target latest --strict

# Predict on new data
churn-predict --config config/default.yaml \
  --input data/new_customers.csv \
  --output predictions.csv
```

### Using DVC

```bash
# Run full pipeline with caching
dvc repro

# View metrics
dvc metrics show
```

---

## CLI Reference

### Pipeline Commands

| Command | Description |
|---------|-------------|
| `churn-prepare` | Data validation, feature engineering, train/val/test splits |
| `churn-train` | Train candidates, select best by ROC-AUC, register in registry |
| `churn-evaluate` | Threshold optimization, quality gates, final test metrics |
| `churn-predict` | Batch inference using production model |

### Management Commands

| Command | Description |
|---------|-------------|
| `churn-model-info` | Display current production model metadata |
| `churn-model-promote --model-id <id>` | Promote model to production |
| `churn-model-rollback` | Rollback to previous production model |
| `churn-check-drift --input <csv>` | Check data drift against training distribution |
| `churn-health-check` | Validate production readiness |
| `churn-validate-config` | Validate configuration file |

### Common Flags

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML config (default: `config/default.yaml`) |
| `--strict` | Fail fast on issues (no silent fallbacks) |
| `--target latest\|production\|local` | Model target for evaluate stage |

---

## Project Structure

```
churn-ml-decision/
├── config/
│   └── default.yaml           # Single source of truth for all configuration
├── data/
│   ├── raw/                   # Source CSV (7,043 telco customers)
│   └── processed/             # Train/val/test .npy arrays (DVC-tracked)
├── docs/
│   ├── PRODUCTION_GUIDE.md    # Pipeline flow, troubleshooting
│   ├── THRESHOLD_ANALYSIS_NOTE.md  # Selection policy rationale
│   ├── DEPLOYMENT_CHECKLIST.md     # Release steps
│   └── PROJECT_REPORT.md      # Executive summary
├── models/
│   ├── registry.json          # Model versions, metrics, promotion status (Git-tracked)
│   ├── *.joblib               # Trained models (timestamped + canonical alias)
│   ├── preprocessor.joblib    # Fitted sklearn ColumnTransformer
│   └── final_test_results.csv # Production model test metrics (Git-tracked)
├── src/churn_ml_decision/
│   ├── prepare.py             # Data validation + feature engineering
│   ├── train.py               # Multi-candidate training + registry
│   ├── evaluate.py            # Threshold optimization + quality gates
│   ├── predict.py             # Batch inference
│   ├── model_registry.py      # JSON-backed model management
│   ├── monitoring.py          # KS-test drift detection
│   ├── config.py              # Pydantic v2 configuration
│   ├── cli.py                 # Management commands
│   └── ...                    # 18 modules total
├── tests/                     # 18 test files (89% coverage)
├── notebooks/                 # 11 exploratory notebooks
├── .github/workflows/ci.yml   # Lint + Pipeline + Tests
├── dvc.yaml                   # Pipeline stage definitions
├── Makefile                   # Development shortcuts
└── pyproject.toml             # Package configuration + CLI entry points
```

---

## Roadmap

### Current State: Batch-Ready

| Capability | Status |
|------------|--------|
| Batch inference | Ready |
| Model versioning | Ready |
| Quality gates | Ready |
| Drift detection | Ready |
| CI/CD pipeline | Ready |
| Test coverage | 89% |

### Planned: Real-Time Deployment

| Item | Priority | Description |
|------|----------|-------------|
| **FastAPI wrapper** | High | REST API: `POST /predict`, `GET /model/info`, `GET /health` |
| **Dockerfile** | High | Container image for consistent deployment |
| **Health endpoint** | Medium | `/health` and `/ready` for Kubernetes probes |
| **Cloud storage** | Medium | S3/GCS integration for model artifact backup |
| **Async processing** | Medium | Queue-based inference for high throughput |
| **Drift alerting** | Low | Webhook/email when drift detected |

### Proposed API Design

```
POST /predict          -> Single customer prediction
POST /predict/batch    -> Batch predictions (JSON array)
GET  /model/info       -> Current production model metadata
GET  /health           -> Service health status
GET  /metrics          -> Prometheus metrics
```

---

## Documentation

- [Production Guide](docs/PRODUCTION_GUIDE.md) - Pipeline flow, troubleshooting, emergency rollback
- [Threshold Analysis Note](docs/THRESHOLD_ANALYSIS_NOTE.md) - Selection policy rationale
- [Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md) - Pre-deployment verification steps
- [Project Report](docs/PROJECT_REPORT.md) - Executive summary for stakeholders

---

## Author

**Omar Piro** - Machine Learning Engineer
