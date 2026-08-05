from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DEFAULT_DATA_FILE = ROOT / "churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "CustomerID"

CANONICAL_COLUMNS = [
    "CustomerID",
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

NUMERIC_FEATURES = ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = [
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]
CAT_FEATURES = ["InternetService", "Contract", "PaymentMethod"]


@dataclass
class ModelArtifacts:
    """Container for trained model assets and metadata."""

    pipeline: Pipeline
    preprocessor: ColumnTransformer
    model_name: str
    metrics: dict[str, float]
    threshold: float
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    training_columns: list[str]
    background_frame: pd.DataFrame


COLUMN_ALIASES = {
    "customerid": "CustomerID",
    "customer_id": "CustomerID",
    "gender": "Gender",
    "seniorcitizen": "SeniorCitizen",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "Tenure",
    "phoneservice": "PhoneService",
    "multiplelines": "MultipleLines",
    "internetservice": "InternetService",
    "onlinesecurity": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "techsupport": "TechSupport",
    "streamingtv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "contract": "Contract",
    "paperlessbilling": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "monthlycharges": "MonthlyCharges",
    "totalcharges": "TotalCharges",
    "churn": "Churn",
}


def _coerce_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: COLUMN_ALIASES.get(str(col).strip().lower(), str(col).strip()) for col in df.columns}
    result = df.rename(columns=renamed).copy()
    ordered = [col for col in CANONICAL_COLUMNS if col in result.columns]
    remainder = [col for col in result.columns if col not in ordered]
    return result[ordered + remainder]


def clean_telco_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = canonicalize_columns(df)
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    if "Tenure" in frame.columns:
        frame["Tenure"] = pd.to_numeric(frame["Tenure"], errors="coerce")
    if "MonthlyCharges" in frame.columns:
        frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    if "SeniorCitizen" in frame.columns:
        frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    for col in [c for c in BINARY_FEATURES + CAT_FEATURES if c in frame.columns]:
        frame[col] = frame[col].astype(str).str.strip().replace({"nan": np.nan})
    if "Churn" in frame.columns:
        churn_numeric = pd.to_numeric(frame["Churn"], errors="coerce")
        churn_text = frame["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        frame["Churn"] = churn_numeric.where(churn_numeric.notna(), churn_text).astype("Float64")
    return frame


def generate_demo_telco_data(n_rows: int = 2500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    customer_id = [f"CUST-{i:05d}" for i in range(1, n_rows + 1)]
    gender = rng.choice(["Female", "Male"], size=n_rows)
    senior = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])
    tenure = np.clip(rng.gamma(shape=2.1, scale=12.0, size=n_rows).astype(int), 0, 72)
    phone = rng.choice(["Yes", "No"], size=n_rows, p=[0.91, 0.09])
    internet = rng.choice(["Fiber optic", "DSL", "No"], size=n_rows, p=[0.45, 0.38, 0.17])

    multiple_lines = []
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    contract = []
    paperless = []
    payment = []
    monthly_charges = []
    total_charges = []
    churn = []

    payment_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    contract_options = ["Month-to-month", "One year", "Two year"]

    for idx in range(n_rows):
        contract_choice = rng.choice(contract_options, p=[0.56, 0.25, 0.19])
        contract.append(contract_choice)
        paperless_choice = rng.choice(["Yes", "No"], p=[0.62, 0.38])
        paperless.append(paperless_choice)
        payment_choice = rng.choice(payment_options, p=[0.33, 0.20, 0.23, 0.24])
        payment.append(payment_choice)

        base_charge = rng.normal(68, 20)
        if internet[idx] == "Fiber optic":
            base_charge += 22
        elif internet[idx] == "No":
            base_charge -= 25
        if contract_choice == "Month-to-month":
            base_charge += 4
        monthly = float(np.clip(base_charge, 18, 120))
        monthly_charges.append(round(monthly, 2))

        if phone[idx] == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"], p=[0.42, 0.58]))

        if internet[idx] == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(rng.choice(["Yes", "No"], p=[0.38, 0.62]))
            online_backup.append(rng.choice(["Yes", "No"], p=[0.46, 0.54]))
            device_protection.append(rng.choice(["Yes", "No"], p=[0.45, 0.55]))
            tech_support.append(rng.choice(["Yes", "No"], p=[0.33, 0.67]))
            streaming_tv.append(rng.choice(["Yes", "No"], p=[0.56, 0.44]))
            streaming_movies.append(rng.choice(["Yes", "No"], p=[0.57, 0.43]))

        total = monthly * tenure[idx] + rng.normal(0, 140)
        total_charges.append(round(max(total, 0), 2))

        churn_logit = -1.8
        churn_logit += 1.25 if contract_choice == "Month-to-month" else 0.0
        churn_logit += 0.95 if tenure[idx] < 12 else 0.0
        churn_logit += 0.65 if internet[idx] == "Fiber optic" else 0.0
        churn_logit += 0.55 if payment_choice == "Electronic check" else 0.0
        churn_logit += 0.52 if tech_support[-1] == "No" and internet[idx] != "No" else 0.0
        churn_logit += 0.38 if online_security[-1] == "No" and internet[idx] != "No" else 0.0
        churn_logit += 0.28 if monthly > 85 else 0.0
        churn_logit += 0.22 if paperless_choice == "Yes" else 0.0
        churn_logit += 0.18 if senior[idx] == 1 else 0.0
        churn_prob = 1 / (1 + np.exp(-churn_logit))
        churn.append("Yes" if rng.random() < churn_prob else "No")

    return pd.DataFrame(
        {
            "CustomerID": customer_id,
            "Gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )


def load_customer_data(data_file: Path | str = DEFAULT_DATA_FILE, target_rows: int = 2500) -> pd.DataFrame:
    path = Path(data_file)
    if path.exists():
        frame = clean_telco_frame(pd.read_csv(path))
    else:
        frame = generate_demo_telco_data(target_rows)

    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(frame.get("TotalCharges"), errors="coerce")
    frame["MonthlyCharges"] = pd.to_numeric(frame.get("MonthlyCharges"), errors="coerce")
    frame["Tenure"] = pd.to_numeric(frame.get("Tenure"), errors="coerce")
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = frame["TotalCharges"].fillna(frame["MonthlyCharges"] * frame["Tenure"]) 
    for column in ["MonthlyCharges", "Tenure", "TotalCharges", "SeniorCitizen"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna(frame[column].median())
    for column in [c for c in CANONICAL_COLUMNS if c not in {"CustomerID", "SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges", "Churn"} and c in frame.columns]:
        frame[column] = frame[column].fillna("Unknown")
    if frame["CustomerID"].duplicated().any():
        frame["CustomerID"] = [f"{cid}-{idx:04d}" for idx, cid in enumerate(frame["CustomerID"].astype(str), start=1)]
    if len(frame) < target_rows:
        synthetic = generate_demo_telco_data(target_rows - len(frame), random_state=99)
        frame = pd.concat([frame, synthetic], ignore_index=True)
    frame = clean_telco_frame(frame)
    return frame[CANONICAL_COLUMNS]


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = clean_telco_frame(df)
    if "Churn" in frame.columns:
        frame = frame.dropna(subset=["Churn"]).copy()
        frame["Churn"] = frame["Churn"].astype(int)
    return frame


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = prepare_model_frame(df)
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    target = frame[TARGET_COLUMN].astype(int)
    return features, target


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [c for c in NUMERIC_FEATURES if c in feature_frame.columns]
    categorical_features = [c for c in feature_frame.columns if c not in numeric_features]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _coerce_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def build_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_model_frame(df)
    return frame.drop(columns=[TARGET_COLUMN], errors="ignore")


def risk_label(probability: float) -> str:
    if probability < 0.20:
        return "Very Low Risk"
    if probability < 0.40:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    if probability < 0.80:
        return "High Risk"
    return "Critical Risk"


def score_customers(frame: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    features = build_prediction_frame(frame)
    predicted_probability = artifacts.pipeline.predict_proba(features)[:, 1]
    scored = features.copy()
    scored["PredictedChurnProbability"] = predicted_probability
    scored["RiskLevel"] = scored["PredictedChurnProbability"].apply(risk_label)
    if "TotalCharges" in scored.columns:
        scored["RevenueAtRisk"] = scored["MonthlyCharges"] * scored["PredictedChurnProbability"] * 12
    else:
        scored["RevenueAtRisk"] = scored["PredictedChurnProbability"] * 0
    return scored


def load_artifacts(model_dir: Path | str = MODEL_DIR) -> ModelArtifacts:
    model_path = Path(model_dir) / "saved_model.pkl"
    preprocessor_path = Path(model_dir) / "preprocessor.pkl"
    payload: dict[str, Any] = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else pipeline.named_steps["preprocessor"]
    return ModelArtifacts(
        pipeline=pipeline,
        preprocessor=preprocessor,
        model_name=payload.get("model_name", getattr(pipeline.named_steps.get("model"), "__class__", type(pipeline)).__name__) if isinstance(payload, dict) else pipeline.named_steps["model"].__class__.__name__,
        metrics=payload.get("metrics", {}) if isinstance(payload, dict) else {},
        threshold=float(payload.get("threshold", 0.5)) if isinstance(payload, dict) else 0.5,
        feature_names=list(payload.get("feature_names", get_feature_names(preprocessor))) if isinstance(payload, dict) else get_feature_names(preprocessor),
        numeric_features=list(payload.get("numeric_features", NUMERIC_FEATURES)) if isinstance(payload, dict) else NUMERIC_FEATURES,
        categorical_features=list(payload.get("categorical_features", [])) if isinstance(payload, dict) else [],
        training_columns=list(payload.get("training_columns", [])) if isinstance(payload, dict) else [],
        background_frame=pd.DataFrame(payload.get("background_records", [])) if isinstance(payload, dict) and payload.get("background_records") else pd.DataFrame(),
    )


def revenue_at_risk(df: pd.DataFrame, probability_column: str = "PredictedChurnProbability") -> float:
    if probability_column not in df.columns or "MonthlyCharges" not in df.columns:
        return 0.0
    return float((df["MonthlyCharges"] * df[probability_column] * 12).sum())


def customer_key_search(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return frame.iloc[0:0]
    mask = frame["CustomerID"].astype(str).str.contains(str(query), case=False, na=False)
    return frame.loc[mask].copy()
