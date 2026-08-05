from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.insights import generate_executive_insights, segment_customers
from utils.metrics import correlation_heatmap_figure, feature_importance_frame, feature_importance_figure
from utils.preprocessing import load_artifacts, load_customer_data, revenue_at_risk, score_customers
from utils.ui import inject_css, kpi_card, page_header, risk_badge

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
def load_dashboard_state() -> tuple[pd.DataFrame, object]:
    data = load_customer_data()
    artifacts = load_artifacts()
    scored = score_customers(data, artifacts)
    scored["Churn"] = pd.to_numeric(data["Churn"], errors="coerce").fillna(0).astype(int)
    scored["CustomerID"] = data["CustomerID"].astype(str)
    return scored, artifacts


def _sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Portfolio Filters")
    gender = st.sidebar.multiselect("Gender", sorted(df["Gender"].dropna().unique().tolist()), default=sorted(df["Gender"].dropna().unique().tolist()))
    senior = st.sidebar.multiselect("Senior Citizen", sorted(df["SeniorCitizen"].dropna().unique().tolist()), default=sorted(df["SeniorCitizen"].dropna().unique().tolist()))
    contract = st.sidebar.multiselect("Contract", sorted(df["Contract"].dropna().unique().tolist()), default=sorted(df["Contract"].dropna().unique().tolist()))
    payment = st.sidebar.multiselect("Payment Method", sorted(df["PaymentMethod"].dropna().unique().tolist()), default=sorted(df["PaymentMethod"].dropna().unique().tolist()))
    internet = st.sidebar.multiselect("Internet Service", sorted(df["InternetService"].dropna().unique().tolist()), default=sorted(df["InternetService"].dropna().unique().tolist()))
    risk_level = st.sidebar.multiselect("Risk Level", ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"], default=["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"])
    monthly = st.sidebar.slider("Monthly Charges", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), (float(df["MonthlyCharges"].quantile(0.1)), float(df["MonthlyCharges"].quantile(0.9))))
    tenure = st.sidebar.slider("Tenure", int(df["Tenure"].min()), int(df["Tenure"].max()), (int(df["Tenure"].quantile(0.1)), int(df["Tenure"].quantile(0.9))))

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
    page_header("AI Customer Churn Intelligence Dashboard", "Executive churn command center for retention, revenue at risk, and explainable AI." , badge=f"Model: {artifacts.model_name}")

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
        kpi_card("Total Customers", f"{total_customers:,}", f"Filtered from {len(scored):,} records")
    with col2:
        kpi_card("Current Churn Rate", f"{churn_rate:.1%}", "Observed portfolio churn")
    with col3:
        kpi_card("Revenue At Risk", f"${current_revenue_risk:,.0f}", "Projected annual exposure")
    with col4:
        kpi_card("Expected Revenue Saved", f"${expected_saved:,.0f}", f"Retention ROI: {roi:.1f}x")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("High Risk Customers", f"{high_risk:,}", "Actionable save queue")
    with c2:
        kpi_card("Critical Risk Customers", f"{critical_risk:,}", "Immediate intervention")
    with c3:
        kpi_card("Average Monthly Revenue", f"${avg_monthly:.2f}", "ARPU in filtered segment")
    with c4:
        kpi_card("Retention Campaign ROI", f"{roi:.1f}x", "Illustrative upside")

    st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8])
    with left:
        churn_by_contract = filtered.groupby("Contract")["Churn"].mean().reset_index()
        fig_contract = px.bar(churn_by_contract, x="Contract", y="Churn", color="Contract", color_discrete_sequence=["#4cc9f0", "#7c3aed", "#f59e0b"])
        fig_contract.update_layout(template="plotly_dark", title="Churn by Contract", height=360, showlegend=False)
        st.plotly_chart(fig_contract, use_container_width=True)
    with right:
        risk_distribution = filtered["RiskLevel"].value_counts().reindex(["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]).fillna(0)
        fig_risk = px.pie(values=risk_distribution.values, names=risk_distribution.index, hole=0.55, color_discrete_sequence=["#2dd4bf", "#4cc9f0", "#f59e0b", "#fb923c", "#ef4444"])
        fig_risk.update_layout(template="plotly_dark", title="Risk Distribution", height=360, showlegend=True)
        st.plotly_chart(fig_risk, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_feature = feature_importance_figure(feature_importance_frame(artifacts, filtered.head(200)))
        st.plotly_chart(fig_feature, use_container_width=True)
    with c2:
        st.plotly_chart(correlation_heatmap_figure(filtered), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        monthly_fig = px.histogram(filtered, x="MonthlyCharges", nbins=30, color="RiskLevel", color_discrete_map={"Very Low Risk": "#2dd4bf", "Low Risk": "#4cc9f0", "Medium Risk": "#f59e0b", "High Risk": "#fb923c", "Critical Risk": "#ef4444"})
        monthly_fig.update_layout(template="plotly_dark", title="Monthly Charges Distribution", height=360)
        st.plotly_chart(monthly_fig, use_container_width=True)
    with c2:
        tenure_fig = px.histogram(filtered, x="Tenure", nbins=24, color="RiskLevel", color_discrete_map={"Very Low Risk": "#2dd4bf", "Low Risk": "#4cc9f0", "Medium Risk": "#f59e0b", "High Risk": "#fb923c", "Critical Risk": "#ef4444"})
        tenure_fig.update_layout(template="plotly_dark", title="Tenure Distribution", height=360)
        st.plotly_chart(tenure_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
    section_header("Executive Insights", "Narrative signals generated from the filtered portfolio.")
    for insight in generate_executive_insights(filtered):
        st.markdown(f"- {insight}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
    section_header("Top 10 High Risk Customers", "The highest-priority accounts after filtering.")
    top_customers = filtered.sort_values(["PredictedChurnProbability", "RevenueAtRisk"], ascending=False).head(10).copy()
    top_customers["RiskBadge"] = top_customers["RiskLevel"].apply(risk_badge)
    st.dataframe(top_customers[["CustomerID", "MonthlyCharges", "Tenure", "Contract", "PredictedChurnProbability", "RiskLevel", "RevenueAtRisk"]], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button("Download High Risk Customers CSV", filtered[filtered["RiskLevel"].isin(["High Risk", "Critical Risk"])].to_csv(index=False).encode("utf-8"), "high_risk_customers.csv", "text/csv")
    st.download_button("Download Predictions CSV", filtered.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")


if __name__ == "__main__":
    render()
