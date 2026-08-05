from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, StratifiedKFold

from .preprocessing import get_feature_names

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be blocked by environment policy
    shap = None


def compute_classification_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    y_pred = (y_prob_arr >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
    }


def confusion_matrix_figure(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> go.Figure:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    y_pred = (y_prob_arr >= threshold).astype(int)
    matrix = confusion_matrix(y_true_arr, y_pred)
    return go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[[0, "#0f172a"], [1, "#4cc9f0"]],
            showscale=False,
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>Count=%{z}<extra></extra>",
        )
    ).update_layout(title="Confusion Matrix", template="plotly_dark", height=360)


def roc_curve_figure(y_true: Iterable[int], y_prob: Iterable[float]) -> go.Figure:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    score = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {score:.3f}", line=dict(color="#4cc9f0", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(color="#64748b", dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark", height=360)
    return fig


def precision_recall_curve_figure(y_true: Iterable[int], y_prob: Iterable[float]) -> go.Figure:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    score = auc(recall, precision)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC = {score:.3f}", line=dict(color="#7c3aed", width=3)))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark", height=360)
    return fig


def learning_curve_figure(pipeline, X: pd.DataFrame, y: Iterable[int]) -> go.Figure:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_sizes, train_scores, valid_scores = learning_curve(
        clone(pipeline),
        X,
        np.asarray(list(y)),
        cv=cv,
        n_jobs=-1,
        scoring="roc_auc",
        train_sizes=np.linspace(0.2, 1.0, 5),
    )
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode="lines+markers", name="Training ROC AUC", line=dict(color="#2dd4bf", width=3)))
    fig.add_trace(go.Scatter(x=train_sizes, y=valid_mean, mode="lines+markers", name="Validation ROC AUC", line=dict(color="#f59e0b", width=3)))
    fig.update_layout(title="Learning Curve", xaxis_title="Training Examples", yaxis_title="ROC AUC", template="plotly_dark", height=360)
    return fig


def feature_importance_frame(artifacts, sample_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    model = artifacts.pipeline.named_steps["model"]
    feature_names = artifacts.feature_names or get_feature_names(artifacts.preprocessor)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        if sample_frame is None:
            sample_frame = artifacts.background_frame if not artifacts.background_frame.empty else pd.DataFrame([{}])
        transformed = np.asarray(artifacts.preprocessor.transform(sample_frame))
        importances = np.abs(transformed).mean(axis=0)
    frame = pd.DataFrame({"feature": feature_names[: len(importances)], "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def feature_importance_figure(frame: pd.DataFrame, limit: int = 15) -> go.Figure:
    top = frame.head(limit).sort_values("importance", ascending=True)
    fig = go.Figure(
        data=go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker=dict(color="#4cc9f0"),
        )
    )
    fig.update_layout(title="Feature Importance", xaxis_title="Relative Importance", yaxis_title="", template="plotly_dark", height=max(420, 28 * len(top) + 160))
    return fig


def correlation_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return go.Figure().update_layout(title="Correlation Heatmap", template="plotly_dark")
    corr = numeric_df.corr().fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(title="Correlation Heatmap", template="plotly_dark", height=600)
    return fig


def shap_explanation(artifacts, customer_frame: pd.DataFrame, background_frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = artifacts.pipeline.named_steps["model"]
    feature_names = artifacts.feature_names or get_feature_names(artifacts.preprocessor)
    background = background_frame if background_frame is not None and not background_frame.empty else artifacts.background_frame
    if background.empty:
        background = customer_frame.sample(min(len(customer_frame), 50), replace=True, random_state=42)
    background_transformed = artifacts.preprocessor.transform(background)
    customer_transformed = artifacts.preprocessor.transform(customer_frame)
    raw_values: np.ndarray
    if shap is not None:
        try:
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(model, background_transformed)
            else:
                explainer = shap.Explainer(model, background_transformed)
            values = explainer(customer_transformed)
            raw_values = np.asarray(values.values if hasattr(values, "values") else values)
            if raw_values.ndim == 3:
                raw_values = raw_values[:, :, 1]
        except Exception:
            raw_values = np.asarray(customer_transformed) - np.asarray(background_transformed).mean(axis=0)
    else:
        raw_values = np.asarray(customer_transformed) - np.asarray(background_transformed).mean(axis=0)
    local = pd.DataFrame(
        {
            "feature": feature_names[: raw_values.shape[-1]],
            "shap_value": raw_values[0],
        }
    ).sort_values("shap_value", ascending=False)
    global_frame = pd.DataFrame(
        {
            "feature": feature_names[: raw_values.shape[-1]],
            "mean_abs_shap": np.abs(raw_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return local, global_frame
