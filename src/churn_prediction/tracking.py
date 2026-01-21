"""MLflow tracking utilities."""

from contextlib import nullcontext
from typing import Any

import mlflow


class Tracker:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def start_run(self, experiment_name: str | None = None):
        if not self.enabled:
            return nullcontext()

        if experiment_name:
            mlflow.set_experiment(experiment_name)
        return mlflow.start_run()

    def log_param(self, key: str, value: Any):
        if self.enabled:
            mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]):
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]):
        if self.enabled:
            mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str):
        if self.enabled:
            mlflow.log_artifact(local_path)
