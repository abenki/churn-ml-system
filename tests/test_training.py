"""Tests for model training."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_prediction.config import FeatureConfig, TrainingConfig
from churn_prediction.features import prepare_features
from churn_prediction.training import create_training_pipeline, split_data, train_model


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for training tests."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "customerID": [f"{i:03d}" for i in range(n_samples)],
            "gender": np.random.choice(["Male", "Female"], n_samples),
            "SeniorCitizen": np.random.choice([0, 1], n_samples),
            "Partner": np.random.choice(["Yes", "No"], n_samples),
            "Dependents": np.random.choice(["Yes", "No"], n_samples),
            "tenure": np.random.randint(0, 72, n_samples),
            "PhoneService": np.random.choice(["Yes", "No"], n_samples),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n_samples,
            ),
            "MonthlyCharges": np.random.uniform(20, 100, n_samples),
            "TotalCharges": np.random.uniform(100, 5000, n_samples),
            "Churn": np.random.choice(["Yes", "No"], n_samples, p=[0.26, 0.74]),
        }
    )


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create training configuration for tests."""
    return TrainingConfig(test_size=0.2, random_seed=42)


@pytest.fixture
def feature_config() -> FeatureConfig:
    """Create feature configuration for tests."""
    return FeatureConfig()


class TestSplitData:
    """Tests for data splitting."""

    def test_split_data_correct_proportions(self, training_config: TrainingConfig) -> None:
        """Split should respect test_size proportion."""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))

        X_train, X_test, y_train, y_test = split_data(X, y, training_config)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_data_stratified(self, training_config: TrainingConfig) -> None:
        """Split should preserve class distribution."""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.array([0] * 70 + [1] * 30))  # 70% class 0, 30% class 1

        _, _, y_train, y_test = split_data(X, y, training_config)

        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)

        # Both should be close to 30%
        assert abs(train_ratio - 0.30) < 0.05
        assert abs(test_ratio - 0.30) < 0.05

    def test_split_data_reproducible(self, training_config: TrainingConfig) -> None:
        """Split should be reproducible with same seed."""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))

        X_train1, X_test1, _, _ = split_data(X, y, training_config)
        X_train2, X_test2, _, _ = split_data(X, y, training_config)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)


class TestTrainingPipeline:
    """Tests for training pipeline."""

    def test_create_training_pipeline_returns_pipeline(self, feature_config: FeatureConfig) -> None:
        """Should return a sklearn Pipeline."""
        pipeline = create_training_pipeline(feature_config, {"C": 1.0, "max_iter": 100})

        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_train_model_fits_pipeline(
        self,
        sample_data: pd.DataFrame,
        feature_config: FeatureConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Training should fit the pipeline."""
        X, y = prepare_features(sample_data, target_column="Churn", id_column="customerID")

        X_train, _, y_train, _ = split_data(X, y, training_config)

        pipeline = create_training_pipeline(
            feature_config, {"C": 1.0, "max_iter": 100, "solver": "lbfgs"}
        )

        trained_pipeline = train_model(pipeline, X_train, y_train)

        # Pipeline should be fitted (can make predictions)
        predictions = trained_pipeline.predict(X_train)

        assert len(predictions) == len(y_train)
        assert all(p in [0, 1] for p in predictions)

    def test_trained_model_can_predict_probabilities(
        self,
        sample_data: pd.DataFrame,
        feature_config: FeatureConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Trained model should output probability scores."""
        X, y = prepare_features(sample_data, target_column="Churn", id_column="customerID")

        X_train, X_test, y_train, _ = split_data(X, y, training_config)

        pipeline = create_training_pipeline(
            feature_config, {"C": 1.0, "max_iter": 100, "solver": "lbfgs"}
        )

        trained_pipeline = train_model(pipeline, X_train, y_train)

        probabilities = trained_pipeline.predict_proba(X_test)

        assert probabilities.shape[1] == 2  # Two classes
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_array_almost_equal(probabilities.sum(axis=1), 1.0)
