from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.insights import generate_executive_insights
from utils.metrics import (
    apply_chart_theme,
    correlation_heatmap_figure,
    feature_importance_figure,
    feature_importance_frame,
)
from utils.preprocessing import load_artifacts, load_customer_data, revenue_at_risk, score_customers
from utils.ui import inject_css, kpi_card, page_header, render_data_uploader_sidebar, render_insight_cards, section_header

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Portfolio Dashboard - ChurnIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_dashboard_state() -> tuple[pd.DataFrame, object]:
    data = render_data_uploader_sidebar()
    artifacts = load_artifacts()
    scored = score_customers(data, artifacts)
    scored["Churn"] = pd.to_numeric(data["Churn"], errors="coerce").fillna(0).astype(int)
    scored["CustomerID"] = data["CustomerID"].astype(str)
    return scored, artifacts


def _sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### 🎛️ Portfolio Filters")
    gender = st.sidebar.multiselect("Gender", sorted(df["Gender"].dropna().unique().tolist()), default=sorted(df["Gender"].dropna().unique().tolist()))
    senior = st.sidebar.multiselect("Senior Citizen", sorted(df["SeniorCitizen"].dropna().unique().tolist()), default=sorted(df["SeniorCitizen"].dropna().unique().tolist()))
    contract = st.sidebar.multiselect("Contract Type", sorted(df["Contract"].dropna().unique().tolist()), default=sorted(df["Contract"].dropna().unique().tolist()))
    payment = st.sidebar.multiselect("Payment Method", sorted(df["PaymentMethod"].dropna().unique().tolist()), default=sorted(df["PaymentMethod"].dropna().unique().tolist()))
    internet = st.sidebar.multiselect("Internet Service", sorted(df["InternetService"].dropna().unique().tolist()), default=sorted(df["InternetService"].dropna().unique().tolist()))
    risk_level = st.sidebar.multiselect("Risk Level", ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"], default=["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"])
    monthly = st.sidebar.slider("Monthly Charges ($)", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())))
    tenure = st.sidebar.slider("Tenure (Months)", int(df["Tenure"].min()), int(df["Tenure"].max()), (int(df["Tenure"].min()), int(df["Tenure"].max())))

    filtered = df[
        df["Gender"].isin(gender)
        & df["SeniorCitizen"].isin(senior)
        & df["Contract"].isin(contract)
        & df["PaymentMethod"].isin(payment)
        & df["InternetService"].isin(internet)
        & df["RiskLevel"].isin(risk_level)
        & df["MonthlyCharges"].between(monthly[0], monthly[1])
        & df["Tenure"].between(tenure[0], tenure[1])
    ].copy()
    return filtered


def render() -> None:
    inject_css()
    scored, artifacts = load_dashboard_state()
    filtered = _sidebar_filters(scored)
    page_header("Portfolio Intelligence Dashboard", "Comprehensive analytics command center for revenue risk, customer churn drivers, and explainable AI.", badge=f"Model: {artifacts.model_name}")

    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(filtered)
    churn_rate = filtered["Churn"].mean() if total_customers else 0.0
    high_risk = int(filtered["RiskLevel"].isin(["High Risk", "Critical Risk"]).sum())
    critical_risk = int((filtered["RiskLevel"] == "Critical Risk").sum())
    current_revenue_risk = revenue_at_risk(filtered)
    expected_saved = current_revenue_risk * 0.30
    avg_monthly = float(filtered["MonthlyCharges"].mean()) if total_customers else 0.0
    roi = (expected_saved - current_revenue_risk * 0.08) / (current_revenue_risk * 0.08) if current_revenue_risk else 0.0

    with col1:
        kpi_card("Filtered Accounts", f"{total_customers:,}", f"From {len(scored):,} total records", icon="👥", delta="1.8%", delta_type="up")
    with col2:
        kpi_card("Churn Rate", f"{churn_rate:.1%}", "Observed segment churn", icon="📉", delta="2.8%", delta_type="down")
    with col3:
        kpi_card("Revenue At Risk", f"${current_revenue_risk:,.0f}", "Projected annual exposure", icon="💰", delta="1.5%", delta_type="up")
    with col4:
        kpi_card("Expected Saved", f"${expected_saved:,.0f}", f"Est. ROI: {roi:.1f}x", icon="🎯", delta="4.2%", delta_type="up")

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("High Risk Accounts", f"{high_risk:,}", "Actionable save queue", icon="⚠️", delta="3.2%", delta_type="down")
    with c2:
        kpi_card("Critical Risk Accounts", f"{critical_risk:,}", "Immediate intervention needed", icon="🚨", delta="1.1%", delta_type="down")
    with c3:
        kpi_card("Average ARPU", f"${avg_monthly:.2f}", "Monthly revenue per user", icon="💵", delta="0.5%", delta_type="up")
    with c4:
        kpi_card("Campaign ROI", f"{roi:.1f}x", "Targeted strategy yield", icon="📈", delta="2.1%", delta_type="up")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Contract & Risk Breakdown
    left, right = st.columns([1.2, 0.8])
    with left:
        section_header("CHURN RATE BY CONTRACT TYPE", "vulnerability across commitment terms")
        churn_by_contract = filtered.groupby("Contract")["Churn"].mean().reset_index()
        fig_contract = px.bar(churn_by_contract, x="Contract", y="Churn", color="Contract", color_discrete_sequence=["#0ea5e9", "#6366f1", "#f59e0b"])
        fig_contract = apply_chart_theme(fig_contract, "", height=340)
        fig_contract.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig_contract, use_container_width=True, config={"displayModeBar": False})
    with right:
        section_header("RISK CATEGORY DISTRIBUTION", "portfolio risk tier breakdown")
        risk_distribution = filtered["RiskLevel"].value_counts().reindex(["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]).fillna(0)
        fig_risk = px.pie(values=risk_distribution.values, names=risk_distribution.index, hole=0.6, color_discrete_sequence=["#94a3b8", "#0ea5e9", "#f59e0b", "#fb923c", "#f43f5e"])
        fig_risk = apply_chart_theme(fig_risk, "", height=340)
        st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Feature Importance & Correlation
    c1, c2 = st.columns(2)
    with c1:
        section_header("FEATURE IMPORTANCE RANKINGS", "top model drivers")
        fig_feature = feature_importance_figure(feature_importance_frame(artifacts, filtered.head(200)))
        st.plotly_chart(fig_feature, use_container_width=True, config={"displayModeBar": False})
    with c2:
        section_header("CORRELATION HEATMAP", "numeric variable dependencies")
        st.plotly_chart(correlation_heatmap_figure(filtered), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Distributions
    c1, c2 = st.columns(2)
    with c1:
        section_header("MONTHLY CHARGES DISTRIBUTION", "billing segment breakdown")
        monthly_fig = px.histogram(filtered, x="MonthlyCharges", nbins=30, color="RiskLevel", color_discrete_map={"Very Low Risk": "#94a3b8", "Low Risk": "#0ea5e9", "Medium Risk": "#f59e0b", "High Risk": "#fb923c", "Critical Risk": "#f43f5e"})
        monthly_fig = apply_chart_theme(monthly_fig, "", height=340)
        st.plotly_chart(monthly_fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        section_header("TENURE DISTRIBUTION (MONTHS)", "customer lifetime tiers")
        tenure_fig = px.histogram(filtered, x="Tenure", nbins=24, color="RiskLevel", color_discrete_map={"Very Low Risk": "#94a3b8", "Low Risk": "#0ea5e9", "Medium Risk": "#f59e0b", "High Risk": "#fb923c", "Critical Risk": "#f43f5e"})
        tenure_fig = apply_chart_theme(tenure_fig, "", height=340)
        st.plotly_chart(tenure_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Insights
    section_header("FILTERED PORTFOLIO SIGNALS", "dynamically generated from active filters")
    render_insight_cards(generate_executive_insights(filtered))

    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)

    # High Risk Accounts Table
    section_header("PRIORITY ACCOUNT QUEUE", "top high-risk accounts requiring intervention")
    top_customers = filtered.sort_values(["PredictedChurnProbability", "RevenueAtRisk"], ascending=False).head(10).copy()
    st.dataframe(top_customers[["CustomerID", "MonthlyCharges", "Tenure", "Contract", "PredictedChurnProbability", "RiskLevel", "RevenueAtRisk"]], use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # Download Actions
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📥 Download High Risk Queue (CSV)", filtered[filtered["RiskLevel"].isin(["High Risk", "Critical Risk"])].to_csv(index=False).encode("utf-8"), "high_risk_customers.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("📥 Download Full Predictions (CSV)", filtered.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv", use_container_width=True)


if __name__ == "__main__":
    render()
