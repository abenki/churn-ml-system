"""Tests for data loading and validation."""

import pandas as pd
import pytest

from churn_prediction.data import DataLoadError, clean_data, validate_data


@pytest.fixture
def valid_sample_data() -> pd.DataFrame:
    """Create valid sample data for testing."""
    return pd.DataFrame(
        {
            "customerID": ["001", "002", "003"],
            "gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [12, 24, 36],
            "PhoneService": ["Yes", "No", "Yes"],
            "MultipleLines": ["Yes", "No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["Yes", "No", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "MonthlyCharges": [29.85, 56.95, 42.30],
            "TotalCharges": [358.20, 1366.80, 1522.80],
            "Churn": ["No", "Yes", "No"],
        }
    )


class TestDataValidation:
    """Tests for data validation."""

    def test_valid_data_passes_validation(self, valid_sample_data: pd.DataFrame) -> None:
        """Valid data should pass schema validation."""
        result = validate_data(valid_sample_data)
        assert len(result) == 3

    def test_invalid_gender_fails_validation(self, valid_sample_data: pd.DataFrame) -> None:
        """Invalid gender value should fail validation."""
        invalid_data = valid_sample_data.copy()
        invalid_data.loc[0, "gender"] = "Unknown"

        with pytest.raises(DataLoadError, match="Schema validation failed"):
            validate_data(invalid_data)

    def test_missing_column_fails_validation(self, valid_sample_data: pd.DataFrame) -> None:
        """Missing required column should fail validation."""
        invalid_data = valid_sample_data.drop(columns=["tenure"])

        with pytest.raises(DataLoadError, match="Schema validation failed"):
            validate_data(invalid_data)

    def test_duplicate_customer_id_fails_validation(self, valid_sample_data: pd.DataFrame) -> None:
        """Duplicate customerID should fail validation."""
        invalid_data = valid_sample_data.copy()
        invalid_data.loc[0, "customerID"] = invalid_data.loc[1, "customerID"]

        with pytest.raises(DataLoadError, match="Schema validation failed"):
            validate_data(invalid_data)

    def test_negative_tenure_fails_validation(self, valid_sample_data: pd.DataFrame) -> None:
        """Negative tenure should fail validation."""
        invalid_data = valid_sample_data.copy()
        invalid_data.loc[0, "tenure"] = -1

        with pytest.raises(DataLoadError, match="Schema validation failed"):
            validate_data(invalid_data)


class TestDataCleaning:
    """Tests for data cleaning."""

    def test_clean_data_handles_whitespace_total_charges(self) -> None:
        """Whitespace in TotalCharges should be handled."""
        data = pd.DataFrame(
            {
                "customerID": ["001"],
                "gender": ["Male"],
                "SeniorCitizen": [0],
                "Partner": ["Yes"],
                "Dependents": ["No"],
                "tenure": [0],
                "PhoneService": ["Yes"],
                "MultipleLines": ["No"],
                "InternetService": ["DSL"],
                "OnlineSecurity": ["No"],
                "OnlineBackup": ["No"],
                "DeviceProtection": ["No"],
                "TechSupport": ["No"],
                "StreamingTV": ["No"],
                "StreamingMovies": ["No"],
                "Contract": ["Month-to-month"],
                "PaperlessBilling": ["Yes"],
                "PaymentMethod": ["Electronic check"],
                "MonthlyCharges": [29.85],
                "TotalCharges": [" "],  # Whitespace value
                "Churn": ["No"],
            }
        )

        cleaned = clean_data(data)

        # Should be imputed to MonthlyCharges * tenure = 0
        assert cleaned.loc[0, "TotalCharges"] == 0.0

    def test_clean_data_preserves_valid_values(self, valid_sample_data: pd.DataFrame) -> None:
        """Valid TotalCharges values should be preserved."""
        original_values = valid_sample_data["TotalCharges"].tolist()
        cleaned = clean_data(valid_sample_data)

        assert cleaned["TotalCharges"].tolist() == original_values
