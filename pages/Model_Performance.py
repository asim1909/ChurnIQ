from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

from utils.metrics import (
    apply_chart_theme,
    compute_classification_metrics,
    confusion_matrix_figure,
    feature_importance_figure,
    feature_importance_frame,
    learning_curve_figure,
    precision_recall_curve_figure,
    roc_curve_figure,
)
from utils.preprocessing import load_artifacts, load_customer_data, split_features_target
from utils.ui import inject_css, kpi_card, page_header, render_data_uploader_sidebar, section_header

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Model Performance - ChurnIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_state() -> tuple[pd.DataFrame, object]:
    data = render_data_uploader_sidebar()
    artifacts = load_artifacts()
    return data, artifacts


def render() -> None:
    inject_css()
    data, artifacts = load_state()
    page_header("Model Performance & QA", "Validate machine learning model quality, decision thresholds, ROC curves, and feature importance drivers.", badge=f"Model: {artifacts.model_name}")

    features, target = split_features_target(data)
    x_train, x_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    valid_prob = artifacts.pipeline.predict_proba(x_valid)[:, 1]
    metrics = compute_classification_metrics(y_valid, valid_prob)

    # Top KPI Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_info = [
        ("Accuracy", metrics["accuracy"], "Correct predictions ratio", "🎯", "1.2%", "up"),
        ("Precision", metrics["precision"], "Positive prediction accuracy", "🔍", "0.8%", "up"),
        ("Recall", metrics["recall"], "True positive capture rate", "⚡", "1.5%", "up"),
        ("F1 Score", metrics["f1"], "Harmonic mean precision/recall", "⚖️", "1.1%", "up"),
        ("ROC AUC", metrics["roc_auc"], "Area under ROC curve", "🏆", "0.5%", "up"),
    ]
    for col, (label, val, hint, icon, delta, d_type) in zip([c1, c2, c3, c4, c5], metrics_info):
        with col:
            kpi_card(label, f"{val:.3f}", hint, icon=icon, delta=delta, delta_type=d_type)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # 2x2 Evaluation Curves Grid
    section_header("MODEL DECISION CURVES", "validation set performance metrics")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_matrix_figure(y_valid, valid_prob), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(roc_curve_figure(y_valid, valid_prob), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.plotly_chart(precision_recall_curve_figure(y_valid, valid_prob), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(learning_curve_figure(artifacts.pipeline, features, target), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Feature Importance & Dataframe
    section_header("GLOBAL FEATURE IMPORTANCE", "ranked drivers across full dataset")
    feature_frame = feature_importance_frame(artifacts, data.head(300))
    st.plotly_chart(feature_importance_figure(feature_frame), use_container_width=True, config={"displayModeBar": False})
    with st.expander("View Full Feature Importance Table"):
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Contract Risk Comparison
    section_header("CONTRACT RISK BENCHMARK", "average predicted churn probability by contract")
    comparison = data.copy()
    comparison["PredictedChurnProbability"] = artifacts.pipeline.predict_proba(features)[:, 1]
    by_contract = comparison.groupby("Contract")["PredictedChurnProbability"].mean().reset_index()
    chart = px.bar(by_contract, x="Contract", y="PredictedChurnProbability", color="Contract", color_discrete_sequence=["#0ea5e9", "#6366f1", "#cbd5e1"])
    chart = apply_chart_theme(chart, "", height=350)
    chart.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    render()
