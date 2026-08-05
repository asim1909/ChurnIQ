from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from pages.Dashboard import render as render_dashboard
from utils.insights import generate_executive_insights
from utils.preprocessing import load_artifacts, load_customer_data, revenue_at_risk, score_customers
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

ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_state() -> tuple[pd.DataFrame, object]:
    data = load_customer_data()
    artifacts = load_artifacts()
    scored = score_customers(data, artifacts)
    scored["Churn"] = pd.to_numeric(data["Churn"], errors="coerce").fillna(0).astype(int)
    return scored, artifacts


def render() -> None:
    inject_css()
    scored, artifacts = load_state()
    page_header("AI Customer Churn Intelligence Dashboard", "Enterprise-grade retention intelligence, built for executive review and frontline action.", badge=f"Best Model: {artifacts.model_name}")

    st.sidebar.title("Navigation")
    st.sidebar.page_link("app.py", label="Executive Overview", icon="🏠")
    st.sidebar.page_link("pages/Dashboard.py", label="Portfolio Dashboard", icon="📊")
    st.sidebar.page_link("pages/Customer_Explorer.py", label="Customer Explorer", icon="🔎")
    st.sidebar.page_link("pages/Model_Performance.py", label="Model Performance", icon="📈")
    st.sidebar.page_link("pages/Retention_Simulator.py", label="Retention Simulator", icon="🧪")

    filtered = scored.copy()
    total = len(filtered)
    churn_rate = filtered["Churn"].mean() if total else 0.0
    risk_customers = int(filtered["RiskLevel"].isin(["High Risk", "Critical Risk"]).sum())
    revenue_risk = revenue_at_risk(filtered)
    avg_monthly = float(filtered["MonthlyCharges"].mean()) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", f"{total:,}", "Active customer base")
    with c2:
        kpi_card("Current Churn Rate", f"{churn_rate:.1%}", "Observed churn in data")
    with c3:
        kpi_card("High Risk Customers", f"{risk_customers:,}", "Requires immediate retention")
    with c4:
        kpi_card("Revenue At Risk", f"${revenue_risk:,.0f}", f"Average monthly revenue: ${avg_monthly:.2f}")

    left, right = st.columns([1.2, 0.8])
    with left:
        chart = px.histogram(filtered, x="PredictedChurnProbability", color="RiskLevel", nbins=25, color_discrete_map={"Very Low Risk": "#2dd4bf", "Low Risk": "#4cc9f0", "Medium Risk": "#f59e0b", "High Risk": "#fb923c", "Critical Risk": "#ef4444"})
        chart.update_layout(template="plotly_dark", title="Predicted Churn Probability Distribution", height=380)
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
        section_header("Executive Insights", "Automatically generated portfolio commentary for leadership review.")
        for insight in generate_executive_insights(filtered):
            st.markdown(f"- {insight}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
    section_header("Portfolio Snapshot", "A quick look at the customers currently driving the dashboard view.")
    st.dataframe(filtered[["CustomerID", "Contract", "MonthlyCharges", "Tenure", "PredictedChurnProbability", "RiskLevel"]].head(20), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Open the full portfolio dashboard"):
        render_dashboard()


if __name__ == "__main__":
    render()
