"""Model evaluation metrics."""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn_prediction.logging_config import get_logger

logger = get_logger("evaluation.metrics")


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""

    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "roc_auc": self.roc_auc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __str__(self) -> str:
        """Format metrics as string."""
        return (
            f"ROC-AUC: {self.roc_auc:.4f} | "
            f"Accuracy: {self.accuracy:.4f} | "
            f"Precision: {self.precision:.4f} | "
            f"Recall: {self.recall:.4f} | "
            f"F1: {self.f1:.4f}"
        )


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationMetrics:
    """Evaluate model on test data.

    Args:
        pipeline: Trained sklearn Pipeline.
        X_test: Test features.
        y_test: Test target.

    Returns:
        EvaluationMetrics with computed metrics.
    """
    logger.info("Evaluating model")

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = EvaluationMetrics(
        roc_auc=float(roc_auc_score(y_test, y_pred_proba)),
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        recall=float(recall_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
    )

    logger.info(f"Evaluation results: {metrics}")

    return metrics
