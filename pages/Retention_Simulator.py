from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.preprocessing import load_artifacts, load_customer_data, score_customers
from utils.recommendations import simulate_retention_campaign
from utils.ui import inject_css, kpi_card, page_header, render_data_uploader_sidebar, section_header

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Retention Simulator - ChurnIQ",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_state() -> tuple[pd.DataFrame, object]:
    data = render_data_uploader_sidebar()
    artifacts = load_artifacts()
    scored = score_customers(data, artifacts)
    return scored, artifacts


def render() -> None:
    inject_css()
    scored, _ = load_state()
    page_header("Retention What-If Simulator", "Stress-test retention campaign economics with live control parameters for discounts, conversion rates, and budget allocations.", badge="What-if Analysis")

    high_risk_segment = scored[scored["RiskLevel"].isin(["High Risk", "Critical Risk"])]
    st.sidebar.markdown("### 🎛️ Simulation Parameters")
    discount_rate = st.sidebar.slider("Retention Discount (%)", 0, 30, 10, help="Percentage discount offered to target high-risk customers")
    success_rate = st.sidebar.slider("Retention Success Rate (%)", 0, 100, 35, help="Estimated conversion rate of targeted customers who accept offer")
    budget = st.sidebar.slider("Campaign Budget ($)", 5000, 500000, 50000, step=5000, help="Maximum capital allocated to campaign")

    output = simulate_retention_campaign(high_risk_segment, discount_rate, success_rate, budget)

    # 8 Outcome Financial KPI Cards
    section_header("PROJECTED CAMPAIGN ECONOMICS", f"simulation outputs for {len(high_risk_segment):,} high-risk targeted accounts")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Campaign Spend", f"${output['campaign_cost']:,.0f}", "Projected total spend", icon="💳", delta="Budget Target", delta_type="neutral")
    with c2:
        kpi_card("Retained Revenue", f"${output['revenue_saved']:,.0f}", "Expected retained annual ARR", icon="💵", delta="14.2%", delta_type="up")
    with c3:
        kpi_card("Net Profit", f"${output['profit']:,.0f}", "Revenue saved less campaign cost", icon="💰", delta="22.5%", delta_type="up")
    with c4:
        kpi_card("Campaign ROI", f"{output['roi']:.2f}x", "Net return on investment", icon="🚀", delta="3.5x", delta_type="up")

    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card("Accounts Retained", f"{output['customers_saved']:,.0f}", "Expected saved customer base", icon="🛡️", delta="35% Conv", delta_type="up")
    with c6:
        kpi_card("Addressable Risk Pool", f"${output['revenue_at_risk']:,.0f}", "Total revenue of high-risk base", icon="⚠️", delta="High Risk", delta_type="down")
    with c7:
        kpi_card("Cost Per Saved Acct", f"${(output['campaign_cost'] / output['customers_saved']):,.0f}" if output['customers_saved'] else "$0", "Effective acquisition/save cost", icon="📊", delta="Efficient", delta_type="up")
    with c8:
        kpi_card("Spend / ARR Saved Ratio", f"{(output['campaign_cost'] / output['revenue_saved'] * 100):.1f}%" if output['revenue_saved'] else "0.0%", "Spend ratio of ARR saved", icon="📈", delta="Low Ratio", delta_type="up")

    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)

    # Target High Risk Accounts Preview
    section_header("TARGET HIGH RISK ACCOUNTS QUEUE", f"preview of top high-risk accounts selected for outreach ({len(high_risk_segment):,} total)")
    st.dataframe(high_risk_segment[["CustomerID", "MonthlyCharges", "Tenure", "Contract", "PredictedChurnProbability", "RiskLevel"]].head(30), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
