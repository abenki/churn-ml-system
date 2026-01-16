"""Artifact persistence for models, metrics, and configurations."""

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from churn_prediction.config import ArtifactsConfig, FeatureConfig
from churn_prediction.evaluation.metrics import EvaluationMetrics
from churn_prediction.logging_config import get_logger

logger = get_logger("artifacts")


def save_model(pipeline: Pipeline, config: ArtifactsConfig) -> Path:
    """Save trained model pipeline to disk.

    Args:
        pipeline: Trained sklearn Pipeline.
        config: Artifacts configuration.

    Returns:
        Path where model was saved.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / config.model_filename

    logger.info(f"Saving model to {model_path}")
    joblib.dump(pipeline, model_path)

    return model_path


def save_metrics(metrics: EvaluationMetrics, config: ArtifactsConfig) -> Path:
    """Save evaluation metrics to disk.

    Args:
        metrics: Evaluation metrics.
        config: Artifacts configuration.

    Returns:
        Path where metrics were saved.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / config.metrics_filename

    logger.info(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    return metrics_path


def save_feature_config(feature_config: FeatureConfig, config: ArtifactsConfig) -> Path:
    """Save feature configuration to disk.

    Args:
        feature_config: Feature configuration.
        config: Artifacts configuration.

    Returns:
        Path where feature config was saved.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    feature_config_path = config.output_dir / config.feature_config_filename

    logger.info(f"Saving feature config to {feature_config_path}")
    with open(feature_config_path, "w") as f:
        json.dump(feature_config.model_dump(), f, indent=2)

    return feature_config_path


def load_model(config: ArtifactsConfig) -> Pipeline:
    """Load trained model pipeline from disk.

    Args:
        config: Artifacts configuration.

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    model_path = config.output_dir / config.model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def load_metrics(config: ArtifactsConfig) -> EvaluationMetrics:
    """Load evaluation metrics from disk.

    Args:
        config: Artifacts configuration.

    Returns:
        Loaded EvaluationMetrics.

    Raises:
        FileNotFoundError: If metrics file doesn't exist.
    """
    metrics_path = config.output_dir / config.metrics_filename

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found at {metrics_path}")

    logger.info(f"Loading metrics from {metrics_path}")
    with open(metrics_path) as f:
        data = json.load(f)

    return EvaluationMetrics(**data)
