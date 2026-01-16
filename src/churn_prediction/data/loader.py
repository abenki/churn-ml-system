from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError

from churn_prediction.data.schema import RawDataSchema
from churn_prediction.logging_config import get_logger

logger = get_logger("data.loader")


class DataLoadError(Exception):
    """Raised when data loading or validation fails."""


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw CSV data from disk.

    Args:
        path: Path to CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        DataLoadError: If file cannot be read.
    """
    logger.info(f"Loading data from {path}")

    try:
        df = pd.read_csv(path, index_col=0)
        logger.info(f"Loaded {len(df)} rows")
        return df
    except Exception as e:
        raise DataLoadError(f"Failed to load data from {path}: {e}") from e


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data before validation.

    Handles known data quality issues:
    - TotalCharges contains whitespace strings that should be null

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning data")
    df = df.copy()

    # TotalCharges has whitespace values that should be treated as missing
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with MonthlyCharges * tenure (reasonable imputation)
    # For customers with tenure=0, TotalCharges should be 0
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]

    n_cleaned = mask.sum()
    if n_cleaned > 0:
        logger.info(f"Imputed {n_cleaned} missing TotalCharges values")

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DataFrame against schema.

    Args:
        df: DataFrame to validate.

    Returns:
        Validated DataFrame.

    Raises:
        DataLoadError: If validation fails.
    """
    logger.info("Validating data schema")

    try:
        validated_df = RawDataSchema.validate(df)
        logger.info("Schema validation passed")
        return validated_df
    except SchemaError as e:
        raise DataLoadError(f"Schema validation failed: {e}") from e


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load, clean, and validate data in one step.

    Args:
        path: Path to CSV file.

    Returns:
        Validated DataFrame ready for feature engineering.

    Raises:
        DataLoadError: If loading or validation fails.
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df = validate_data(df)
    return df
