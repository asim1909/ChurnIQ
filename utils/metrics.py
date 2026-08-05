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
from sklearn.model_selection import StratifiedKFold, learning_curve

from .preprocessing import get_feature_names

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be blocked by environment policy
    shap = None


def apply_chart_theme(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    """Apply minimalist light theme and styling to Plotly figures matching reference design."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title.upper()}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#475569"),
            x=0.0,
            y=0.98,
        ),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#64748b", size=11),
        height=height,
        margin=dict(l=40, r=20, t=45, b=40),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_color="#f8fafc",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            bordercolor="#334155",
        ),
        xaxis=dict(
            gridcolor="#f1f5f9",
            griddash="dash",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=10, color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            griddash="dash",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=10, color="#64748b"),
        ),
    )
    return fig


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
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[[0, "#f8fafc"], [0.5, "#bae6fd"], [1, "#0ea5e9"]],
            showscale=False,
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#0f172a", family="JetBrains Mono"),
            hovertemplate="%{y} vs %{x}<br>Count=%{z}<extra></extra>",
        )
    )
    return apply_chart_theme(fig, "Confusion Matrix", 360)


def roc_curve_figure(y_true: Iterable[int], y_prob: Iterable[float]) -> go.Figure:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    score = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC AUC = {score:.3f}",
            line=dict(color="#0ea5e9", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.08)",
        )
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(color="#cbd5e1", dash="dash", width=1.5)))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return apply_chart_theme(fig, "ROC Curve", 360)


def precision_recall_curve_figure(y_true: Iterable[int], y_prob: Iterable[float]) -> go.Figure:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    score = auc(recall, precision)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR AUC = {score:.3f}",
            line=dict(color="#6366f1", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.08)",
        )
    )
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return apply_chart_theme(fig, "Precision-Recall Curve", 360)


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
    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode="lines+markers", name="Training ROC AUC", line=dict(color="#10b981", width=3)))
    fig.add_trace(go.Scatter(x=train_sizes, y=valid_mean, mode="lines+markers", name="Validation ROC AUC", line=dict(color="#f59e0b", width=3)))
    fig.update_layout(xaxis_title="Training Examples", yaxis_title="ROC AUC")
    return apply_chart_theme(fig, "Learning Curve", 360)


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
            marker=dict(color="#334155", cornerradius=4),
        )
    )
    fig.update_layout(xaxis_title="Relative Importance", yaxis_title="")
    return apply_chart_theme(fig, "Feature Importance", max(420, 28 * len(top) + 120))


def correlation_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return apply_chart_theme(go.Figure(), "Correlation Heatmap", 400)
    corr = numeric_df.corr().fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=[[0, "#6366f1"], [0.5, "#ffffff"], [1, "#0ea5e9"]],
            zmin=-1,
            zmax=1,
        )
    )
    return apply_chart_theme(fig, "Correlation Heatmap", 420)


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
