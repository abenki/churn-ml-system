"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import FeatureConfig
from churn_prediction.features import create_preprocessor, prepare_features


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for feature testing."""
    return pd.DataFrame(
        {
            "customerID": ["001", "002", "003", "004"],
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [12, 24, 36, 48],
            "PhoneService": ["Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["Yes", "No phone service", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "Yes"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No"],
            "DeviceProtection": ["Yes", "No", "No internet service", "Yes"],
            "TechSupport": ["No", "Yes", "No internet service", "No"],
            "StreamingTV": ["Yes", "No", "No internet service", "Yes"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 42.30, 89.10],
            "TotalCharges": [358.20, 1366.80, 1522.80, 4276.80],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )


@pytest.fixture
def feature_config() -> FeatureConfig:
    """Create feature configuration for testing."""
    return FeatureConfig()


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_prepare_features_returns_correct_shapes(self, sample_data: pd.DataFrame) -> None:
        """Should return feature matrix and target with correct shapes."""
        X, y = prepare_features(sample_data, target_column="Churn", id_column="customerID")

        assert X.shape[0] == 4  # 4 samples
        assert X.shape[1] == 19  # 21 columns - customerID - Churn
        assert len(y) == 4

    def test_prepare_features_excludes_target_and_id(self, sample_data: pd.DataFrame) -> None:
        """Should exclude target and ID columns from features."""
        X, _ = prepare_features(sample_data, target_column="Churn", id_column="customerID")

        assert "Churn" not in X.columns
        assert "customerID" not in X.columns

    def test_prepare_features_encodes_target_correctly(self, sample_data: pd.DataFrame) -> None:
        """Target should be encoded as binary (0/1)."""
        _, y = prepare_features(sample_data, target_column="Churn", id_column="customerID")

        assert set(y.unique()) == {0, 1}
        assert y.iloc[0] == 0  # "No" -> 0
        assert y.iloc[1] == 1  # "Yes" -> 1


class TestCreatePreprocessor:
    """Tests for create_preprocessor function."""

    def test_preprocessor_transforms_data(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Preprocessor should transform data without errors."""
        X, _ = prepare_features(sample_data, target_column="Churn", id_column="customerID")
        preprocessor = create_preprocessor(feature_config)

        X_transformed = np.asarray(preprocessor.fit_transform(X))

        assert X_transformed.shape[0] == 4
        assert X_transformed.shape[1] > 0

    def test_preprocessor_scales_numerical_features(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Numerical features should be standardized."""
        X, _ = prepare_features(sample_data, target_column="Churn", id_column="customerID")
        preprocessor = create_preprocessor(feature_config)

        X_transformed = preprocessor.fit_transform(X)

        # Numerical features are first in the transformer
        num_features = X_transformed[:, : len(feature_config.numerical_features)]

        # Standardized features should have mean ~0 and std ~1
        assert np.abs(num_features.mean(axis=0)).max() < 0.1
        assert np.abs(num_features.std(axis=0) - 1).max() < 0.5

    def test_preprocessor_one_hot_encodes_categorical_features(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Categorical features should be one-hot encoded."""
        X, _ = prepare_features(sample_data, target_column="Churn", id_column="customerID")
        preprocessor = create_preprocessor(feature_config)

        X_transformed = preprocessor.fit_transform(X)

        # One-hot encoded features should only contain 0 and 1
        cat_features = X_transformed[:, len(feature_config.numerical_features) :]
        unique_values = np.unique(cat_features)

        assert all(v in [0.0, 1.0] for v in unique_values)

    def test_preprocessor_is_deterministic(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Preprocessor should produce identical results on same data."""
        X, _ = prepare_features(sample_data, target_column="Churn", id_column="customerID")

        preprocessor1 = create_preprocessor(feature_config)
        preprocessor2 = create_preprocessor(feature_config)

        X_transformed1 = np.asarray(preprocessor1.fit_transform(X))
        X_transformed2 = np.asarray(preprocessor2.fit_transform(X))

        assert X_transformed1 is not None
        assert X_transformed2 is not None

        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)
