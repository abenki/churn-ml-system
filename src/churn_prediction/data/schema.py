import pandera.pandas as pa
from pandera.typing.pandas import Series


class RawDataSchema(pa.DataFrameModel):
    """Schema for raw telco customer churn data."""

    customerID: Series[str] = pa.Field(nullable=False, unique=True)
    gender: Series[str] = pa.Field(isin=["Male", "Female"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"])
    MultipleLines: Series[str] = pa.Field(isin=["Yes", "No", "No phone service"])
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    OnlineSecurity: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    OnlineBackup: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    DeviceProtection: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    TechSupport: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    StreamingTV: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    StreamingMovies: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"])
    PaymentMethod: Series[str] = pa.Field(
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[float] = pa.Field(ge=0, nullable=True)
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
