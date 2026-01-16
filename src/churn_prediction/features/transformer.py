"""Feature transformation for churn prediction."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.config import FeatureConfig
from churn_prediction.logging_config import get_logger

logger = get_logger("features.transformer")


def create_preprocessor(feature_config: FeatureConfig) -> ColumnTransformer:
    """Create a sklearn preprocessor for feature transformation.

    The preprocessor includes numerical and categorical feature
    transformations. It applies standard scaling to numerical
    features and one-hot encoding to categorical features.

    Args:
        feature_config: Feature configuration specifying columns.

    Returns:
        Configured ColumnTransformer.
    """
    logger.info("Creating feature preprocessor")

    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, feature_config.numerical_features),
            ("cat", categorical_transformer, feature_config.categorical_features),
        ],
        remainder="drop",
    )

    logger.info(
        f"Preprocessor configured with {len(feature_config.numerical_features)} numerical "
        f"and {len(feature_config.categorical_features)} categorical features"
    )

    return preprocessor


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target from raw data.

    Separates features from target, excludes ID column.

    Args:
        df: Validated DataFrame.
        target_column: Name of target column.
        id_column: Name of ID column to exclude.

    Returns:
        Tuple of (feature DataFrame, target Series).
    """
    logger.info("Preparing features and target")

    # Encode target to binary
    y = (df[target_column] == "Yes").astype(int)

    # Drop target and ID columns from features
    X = df.drop(columns=[target_column, id_column])

    logger.info(f"Features shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    return X, y


def get_feature_names(preprocessor: ColumnTransformer, feature_config: FeatureConfig) -> list[str]:
    """Get feature names after transformation.

    Must be called after fitting the preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        feature_config: Feature configuration.

    Returns:
        List of transformed feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn versions
        num_features = feature_config.numerical_features
        cat_features = []
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        for i, col in enumerate(feature_config.categorical_features):
            for cat in encoder.categories_[i]:
                cat_features.append(f"{col}_{cat}")
        return num_features + cat_features
