"""Main training pipeline orchestrating all components."""

import argparse
import logging
import sys
from pathlib import Path

from churn_prediction.artifacts import save_feature_config, save_metrics, save_model
from churn_prediction.config import Settings, load_config
from churn_prediction.data import load_and_validate
from churn_prediction.evaluation import evaluate_model
from churn_prediction.features import prepare_features
from churn_prediction.logging_config import get_logger, setup_logging
from churn_prediction.tracking import Tracker
from churn_prediction.training import create_training_pipeline, split_data, train_model

logger = get_logger("pipeline")


def run_pipeline(settings: Settings) -> None:
    """Run the full training pipeline.

    Args:
        settings: Validated settings object.
    """
    logger.info("Starting training pipeline")

    # Load and validate data
    df = load_and_validate(settings.data.raw_data_path)

    # Prepare features and target
    X, y = prepare_features(
        df,
        target_column=settings.data.target_column,
        id_column=settings.data.id_column,
    )

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, settings.training)

    # Track with MLflow if enabled
    tracker = Tracker(enabled=settings.mlflow.enabled)
    with tracker.start_run(experiment_name=settings.mlflow.experiment_name):
        # Log parameters
        tracker.log_param("test_size", settings.training.test_size)
        tracker.log_param("random_seed", settings.training.random_seed)
        tracker.log_params(settings.training.model_params)

        # Create pipeline
        pipeline = create_training_pipeline(
            feature_config=settings.features,
            model_params=settings.training.model_params,
        )
        pipeline = train_model(pipeline, X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(pipeline, X_test, y_test)

        # Log evaluation metrics
        tracker.log_metrics(metrics.to_dict())

        # Save artifacts
        model_path = save_model(pipeline, settings.artifacts)
        metrics_path = save_metrics(metrics, settings.artifacts)
        feature_config_path = save_feature_config(settings.features, settings.artifacts)

        # Log artifacts
        tracker.log_artifact(str(model_path))
        tracker.log_artifact(str(metrics_path))
        tracker.log_artifact(str(feature_config_path))

    logger.info("Training pipeline completed successfully")
    logger.info(
        f"Model saved to {settings.artifacts.output_dir / settings.artifacts.model_filename}"
    )
    logger.info(f"Final metrics: {metrics}")


def main() -> int:
    """Main entry point for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)

    try:
        settings = load_config(args.config)
        run_pipeline(settings)
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
