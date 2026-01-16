"""Model training for churn prediction."""

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_prediction.config import FeatureConfig, TrainingConfig
from churn_prediction.features import create_preprocessor
from churn_prediction.logging_config import get_logger

logger = get_logger("training.trainer")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Uses stratified splitting to preserve class distribution.

    Args:
        X: Feature matrix.
        y: Target array.
        config: Training configuration.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info(f"Splitting data with test_size={config.test_size}, seed={config.random_seed}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Train target distribution: {np.bincount(y_train).tolist()}")
    logger.info(f"Test target distribution: {np.bincount(y_test).tolist()}")

    return (
        cast(pd.DataFrame, X_train),
        cast(pd.DataFrame, X_test),
        cast(pd.Series, y_train),
        cast(pd.Series, y_test),
    )


def create_model(model_params: dict[str, Any]) -> LogisticRegression:
    """Create a logistic regression model.

    Args:
        model_params: Model hyperparameters.

    Returns:
        Configured LogisticRegression model.
    """
    logger.info(f"Creating LogisticRegression with params: {model_params}")

    # Use class_weight='balanced' to handle class imbalance
    return LogisticRegression(class_weight="balanced", **model_params)


def create_training_pipeline(
    feature_config: FeatureConfig,
    model_params: dict[str, Any],
) -> Pipeline:
    """Create a full training pipeline with preprocessing and model.

    Args:
        feature_config: Feature configuration.
        model_params: Model hyperparameters.

    Returns:
        sklearn Pipeline with preprocessor and model.
    """
    preprocessor = create_preprocessor(feature_config)
    model = create_model(model_params)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    logger.info("Training pipeline created")
    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train the model pipeline.

    Args:
        pipeline: sklearn Pipeline to train.
        X_train: Training features.
        y_train: Training target.

    Returns:
        Fitted pipeline.
    """
    logger.info("Starting model training")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed")
    return pipeline
