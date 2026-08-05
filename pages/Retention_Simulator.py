from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.preprocessing import load_artifacts, load_customer_data, score_customers
from utils.recommendations import simulate_retention_campaign
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
    scored = score_customers(data, artifacts)
    return scored, artifacts


def render() -> None:
    inject_css()
    scored, _ = load_state()
    page_header("Retention Simulator", "Stress-test retention investments with live controls for discount, success rate, and budget.", badge="What-if")

    high_risk_segment = scored[scored["RiskLevel"].isin(["High Risk", "Critical Risk"])]
    st.sidebar.header("Simulation Controls")
    discount_rate = st.sidebar.slider("Retention Discount (%)", 0, 20, 10)
    success_rate = st.sidebar.slider("Retention Success Rate (%)", 0, 100, 35)
    budget = st.sidebar.slider("Campaign Budget ($)", 1000, 500000, 50000, step=1000)

    output = simulate_retention_campaign(high_risk_segment, discount_rate, success_rate, budget)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Campaign Cost", f"${output['campaign_cost']:,.0f}", "Projected spend")
    with c2:
        kpi_card("Revenue Saved", f"${output['revenue_saved']:,.0f}", "Expected retained revenue")
    with c3:
        kpi_card("Profit", f"${output['profit']:,.0f}", "Revenue saved less cost")
    with c4:
        kpi_card("ROI", f"{output['roi']:.2f}x", "Campaign efficiency")

    c5, c6 = st.columns(2)
    with c5:
        kpi_card("Customers Saved", f"{output['customers_saved']:,.0f}", "Expected retained accounts")
    with c6:
        kpi_card("Revenue At Risk", f"${output['revenue_at_risk']:,.0f}", "Addressable high-risk base")

    section_header("Target Segment", "High-risk accounts selected for scenario analysis.")
    st.write(f"High-risk customers selected: {len(high_risk_segment):,}")
    st.dataframe(high_risk_segment[["CustomerID", "MonthlyCharges", "Tenure", "Contract", "PredictedChurnProbability", "RiskLevel"]].head(25), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
