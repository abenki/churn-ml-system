"""Configuration management for the churn prediction system."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Configuration for data ingestion."""

    raw_data_path: Path = Field(description="Path to raw CSV data")
    target_column: str = Field(default="Churn", description="Name of target column")
    id_column: str = Field(default="customerID", description="Name of ID column")

    @field_validator("raw_data_path")
    @classmethod
    def validate_path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Data file does not exist: {v}")
        return v


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""

    categorical_features: list[str] = Field(
        default=[
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
        description="List of categorical feature columns",
    )
    numerical_features: list[str] = Field(
        default=["tenure", "MonthlyCharges", "TotalCharges"],
        description="List of numerical feature columns",
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    model_params: dict[str, Any] = Field(
        default={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        description="Model hyperparameters",
    )


class ArtifactsConfig(BaseModel):
    """Configuration for artifacts storage."""

    output_dir: Path = Field(
        default=Path("artifacts"), description="Output directory for artifacts"
    )
    model_filename: str = Field(default="model.joblib", description="Model filename")
    metrics_filename: str = Field(default="metrics.json", description="Metrics filename")
    feature_config_filename: str = Field(
        default="feature_config.json", description="Feature config filename"
    )


class MLflowConfig(BaseModel):
    """Configuration for MLflow."""

    experiment_name: str = Field(default="churn-prediction", description="Experiment name")
    enabled: bool = Field(default=True, description="Whether to track or not a pipeline run")


class Settings(BaseSettings):
    """Main settings class combining all configurations."""

    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


def load_config(config_path: Path | None = None) -> Settings:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default values.

    Returns:
        Settings object with validated configuration.

    Raises:
        ValueError: If config file doesn't exist or validation fails.
    """
    if config_path is None:
        raise ValueError("Config path must be provided")

    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return Settings(**config_dict)
