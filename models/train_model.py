from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import compute_classification_metrics
from utils.preprocessing import (
    MODEL_DIR,
    ModelArtifacts,
    build_preprocessor,
    build_prediction_frame,
    generate_demo_telco_data,
    get_feature_names,
    load_customer_data,
    prepare_model_frame,
    split_features_target,
)

DATA_FILE = ROOT / "churn.csv"
RANDOM_STATE = 42


def build_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
    return models


def fit_and_select_best_model() -> tuple[dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data = load_customer_data(DATA_FILE, target_rows=2500)
    features, target = split_features_target(data)
    x_train, x_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_valid_processed = preprocessor.transform(x_valid)
    feature_names = get_feature_names(preprocessor)

    candidate_models = build_models()
    results: list[tuple[str, float, dict[str, float], Pipeline]] = []
    best_name = None
    best_score = -np.inf
    best_pipeline = None
    best_metrics: dict[str, float] = {}

    for name, estimator in candidate_models.items():
        estimator.fit(x_train_processed, y_train)
        valid_prob = estimator.predict_proba(x_valid_processed)[:, 1]
        metrics = compute_classification_metrics(y_valid, valid_prob)
        score = metrics["roc_auc"]
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        results.append((name, score, metrics, pipeline))
        if score > best_score:
            best_score = score
            best_name = name
            best_metrics = metrics
            best_pipeline = pipeline

    assert best_name is not None and best_pipeline is not None
    full_preprocessor = build_preprocessor(features)
    full_preprocessor.fit(features)
    best_estimator = build_models()[best_name]
    best_estimator.fit(full_preprocessor.transform(features), target)
    final_pipeline = Pipeline([("preprocessor", full_preprocessor), ("model", best_estimator)])

    metadata = {
        "pipeline": final_pipeline,
        "model_name": best_name,
        "metrics": best_metrics,
        "threshold": 0.5,
        "feature_names": get_feature_names(full_preprocessor),
        "numeric_features": [c for c in ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges"] if c in features.columns],
        "categorical_features": [c for c in features.columns if c not in ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges"]],
        "training_columns": list(features.columns),
        "background_records": features.sample(min(len(features), 200), random_state=RANDOM_STATE).to_dict(orient="records"),
        "validation_score": float(best_score),
    }
    return metadata, features, target, x_valid, y_valid


def save_artifacts(metadata: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(metadata, MODEL_DIR / "saved_model.pkl")
    joblib.dump(metadata["pipeline"].named_steps["preprocessor"], MODEL_DIR / "preprocessor.pkl")


def main() -> None:
    metadata, _, _, _, _ = fit_and_select_best_model()
    save_artifacts(metadata)
    print(f"Best model: {metadata['model_name']}")
    print(f"Validation ROC AUC: {metadata['validation_score']:.4f}")
    print(f"Artifacts saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
