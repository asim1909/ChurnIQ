from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

from utils.metrics import (
    compute_classification_metrics,
    confusion_matrix_figure,
    feature_importance_figure,
    feature_importance_frame,
    learning_curve_figure,
    precision_recall_curve_figure,
    roc_curve_figure,
)
from utils.preprocessing import load_artifacts, load_customer_data, split_features_target
from utils.ui import inject_css, kpi_card, page_header

try:
        from utils.ui import section_header
except ImportError:  # pragma: no cover - compatibility with stale Streamlit reload state
        def section_header(title: str, subtitle: str | None = None, badge: str | None = None) -> None:
                badge_html = f'<span class="risk-pill risk-low">{badge}</span>' if badge else ""
                subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
                st.markdown(
                        f"""
                        <div class="section-heading">
                            <div class="section-heading-copy">
                                <div class="section-kicker">Section</div>
                                <h2 class="section-title">{title}</h2>
                                {subtitle_html}
                            </div>
                            <div class="section-heading-meta">{badge_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                )

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def load_state() -> tuple[pd.DataFrame, object]:
    data = load_customer_data()
    artifacts = load_artifacts()
    return data, artifacts


def render() -> None:
    inject_css()
    data, artifacts = load_state()
    page_header("Model Performance", "Validate model quality, inspect decision curves, and monitor explainability artifacts.", badge="Model QA")

    features, target = split_features_target(data)
    x_train, x_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    valid_prob = artifacts.pipeline.predict_proba(x_valid)[:, 1]
    metrics = compute_classification_metrics(y_valid, valid_prob)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in zip(
        [c1, c2, c3, c4, c5],
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"]],
    ):
        with col:
            kpi_card(label, f"{value:.3f}", "Validation set performance")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_matrix_figure(y_valid, valid_prob), use_container_width=True)
        st.plotly_chart(roc_curve_figure(y_valid, valid_prob), use_container_width=True)
    with right:
        st.plotly_chart(precision_recall_curve_figure(y_valid, valid_prob), use_container_width=True)
        st.plotly_chart(learning_curve_figure(artifacts.pipeline, features, target), use_container_width=True)

    section_header("Feature Importance", "Ranked drivers from the selected validation model.")
    feature_frame = feature_importance_frame(artifacts, data.head(300))
    st.plotly_chart(feature_importance_figure(feature_frame), use_container_width=True)
    st.dataframe(feature_frame.head(20), use_container_width=True, hide_index=True)

    section_header("Contract Risk Comparison", "Average churn probability by contract type.")
    comparison = data.copy()
    comparison["PredictedChurnProbability"] = artifacts.pipeline.predict_proba(features)[:, 1]
    by_contract = comparison.groupby("Contract")["PredictedChurnProbability"].mean().reset_index()
    chart = px.bar(by_contract, x="Contract", y="PredictedChurnProbability", color="PredictedChurnProbability", color_continuous_scale="Blues")
    chart.update_layout(template="plotly_dark", height=350, title="Average Churn Probability by Contract")
    st.plotly_chart(chart, use_container_width=True)


if __name__ == "__main__":
    render()
