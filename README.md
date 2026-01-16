# Churn ML System

A production-oriented machine learning system for customer churn prediction. The goal is to predict whether a customer will churn or not based on their behavior and demographics.

## Overview

This project aims at demonstrating a clean, reproducible ML training system with:
- Structured codebase with clear module boundaries
- DataFrames data validation with Pandera schemas
- Feature engineering with sklearn pipelines
- Configuration management with Pydantic
- Comprehensive test coverage
- Relevant metrics for model evaluation
- Pre-commit hooks for code quality (ruff)

## Quick Start

### Installation

```bash
# Install dependencies
git clone git@github.com:abenki/churn-ml-system.git
uv sync --all-extras # --all-extras is used to install all optional dependencies
```

### Training

Train the model with a single command:

```bash
uv run train --config config.yaml
```

This will:
1. Load and validate the data
2. Preprocess features
3. Train a logistic regression model
4. Evaluate on unused test set
5. Save model and metrics to `artifacts/`

### Running Tests

```bash
uv run pytest tests/ -v
```

### Linting

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Project Structure

```
src/churn_prediction/
├── __init__.py
├── config.py           # Pydantic configuration
├── logging_config.py   # Logging setup
├── pipeline.py         # Main training pipeline
├── artifacts.py        # Model/metrics persistence
├── data/
│   ├── loader.py       # Data loading and cleaning
│   └── schema.py       # Pandera validation schemas
├── features/
│   └── transformer.py  # Feature preprocessing
├── training/
│   └── trainer.py      # Model training logic
└── evaluation/
    └── metrics.py      # Evaluation metrics
```

## Configuration

Configuration is managed via `config.yaml`:

```yaml
data:
  raw_data_path: data/telco_customer_churn.csv
  target_column: Churn
  id_column: customerID

training:
  test_size: 0.2
  random_seed: 42
  model_params:
    C: 1.0
    max_iter: 1000
```

## Artifacts

After training, the following artifacts are saved to `artifacts/`:
- `model.joblib` - Trained sklearn pipeline
- `metrics.json` - Evaluation metrics (ROC-AUC, F1, etc.)
- `feature_config.json` - Feature configuration used

## Development

Install pre-commit hooks:

```bash
uv run pre-commit install
```
