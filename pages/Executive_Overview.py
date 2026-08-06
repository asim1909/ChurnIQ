from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.insights import generate_executive_insights
from utils.metrics import apply_chart_theme
from utils.preprocessing import load_artifacts, load_customer_data, revenue_at_risk, score_customers
from utils.ui import (
    inject_css,
    kpi_card,
    list_item_row,
    page_header,
    render_data_uploader_sidebar,
    render_insight_cards,
    section_header,
)

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Executive Overview - ChurnIQ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_state() -> tuple[pd.DataFrame, object]:
    data = render_data_uploader_sidebar()
    artifacts = load_artifacts()
    scored = score_customers(data, artifacts)
    scored["Churn"] = pd.to_numeric(data["Churn"], errors="coerce").fillna(0).astype(int)
    return scored, artifacts


def render() -> None:
    inject_css()
    scored, artifacts = load_state()
    source_name = st.session_state.get("data_source_name", "Default Portfolio")
    page_header(
        "Executive Overview",
        "Minimalist executive command center for portfolio retention, revenue exposure, and next-best actions.",
        badge=f"Data: {source_name} | Model: {artifacts.model_name}",
    )

    filtered = scored.copy()
    total = len(filtered)
    churn_rate = filtered["Churn"].mean() if total else 0.0
    risk_customers = int(filtered["RiskLevel"].isin(["High Risk", "Critical Risk"]).sum())
    revenue_risk = revenue_at_risk(filtered)
    avg_monthly = float(filtered["MonthlyCharges"].mean()) if total else 0.0

    # Top KPI Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", f"{total:,}", "Active customer base", icon="👥", delta="1.8%", delta_type="up")
    with c2:
        kpi_card("Current Churn Rate", f"{churn_rate:.1%}", "Observed historical churn", icon="📉", delta="2.8%", delta_type="down")
    with c3:
        kpi_card("High Risk Accounts", f"{risk_customers:,}", "Immediate save queue", icon="⚠️", delta="3.2%", delta_type="down")
    with c4:
        kpi_card("Revenue At Risk", f"${revenue_risk:,.0f}", f"Avg monthly ARR: ${avg_monthly * 12:.0f}", icon="💰", delta="1.5%", delta_type="up")

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # Main Grid (2 Columns matching reference layout)
    left, right = st.columns([1.25, 0.75])
    with left:
        # Chart 1: Modern Color-Coded Bar Graph by Risk Tier
        section_header("PORTFOLIO RISK DISTRIBUTION", "Account volume categorized by risk tier.")

        risk_counts = (
            filtered["RiskLevel"]
            .value_counts()
            .reindex(["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"])
            .fillna(0)
            .reset_index()
        )
        risk_counts.columns = ["RiskCategory", "Accounts"]
        risk_counts["Share"] = (risk_counts["Accounts"] / total * 100).apply(lambda x: f"{x:.1f}%")

        fig_prob = px.bar(
            risk_counts,
            x="RiskCategory",
            y="Accounts",
            color="RiskCategory",
            text="Accounts",
            color_discrete_map={
                "Very Low Risk": "#10b981",
                "Low Risk": "#0ea5e9",
                "Medium Risk": "#f59e0b",
                "High Risk": "#f97316",
                "Critical Risk": "#f43f5e",
            },
        )
        fig_prob.update_traces(
            texttemplate="%{y:,}",
            textposition="outside",
            marker=dict(cornerradius=8),
            textfont=dict(size=12, color="#0f172a", family="Plus Jakarta Sans"),
            hovertemplate="<b>%{x}</b><br>Accounts: %{y:,}<br>Portfolio Share: %{customdata[0]}<extra></extra>",
            customdata=risk_counts[["Share"]],
        )
        fig_prob = apply_chart_theme(fig_prob, "", height=280)
        fig_prob.update_layout(
            showlegend=False,
            bargap=0.35,
            xaxis_title="",
            yaxis_title="Accounts",
        )
        st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # Chart 2: Clean rounded bar chart for contract summary
        section_header("ACCOUNTS BY CONTRACT TYPE", "Contract distribution driving portfolio commitment.")
        contract_summary = filtered.groupby("Contract")["CustomerID"].count().reset_index()
        contract_summary.columns = ["Contract", "Accounts"]

        fig_bar = px.bar(
            contract_summary,
            x="Contract",
            y="Accounts",
            text_auto=",",
            color="Contract",
            color_discrete_sequence=["#1e293b", "#475569", "#cbd5e1"],
        )
        fig_bar = apply_chart_theme(fig_bar, "", height=240)
        fig_bar.update_traces(
            textposition="outside",
            marker=dict(cornerradius=6),
            textfont=dict(size=12, color="#0f172a", family="Plus Jakarta Sans"),
        )
        fig_bar.update_layout(
            showlegend=False,
            bargap=0.45,
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with right:
        # Executive Insights Callout Cards
        section_header("EXECUTIVE INSIGHTS", "Automated business signals & commentary.")
        insights = generate_executive_insights(filtered)
        render_insight_cards(insights[:4])

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # Recent Vulnerable Accounts List
        section_header("RECENT VULNERABLE ACCOUNTS", "Highest priority accounts requiring outreach.")
        top_vulnerable = filtered.sort_values("PredictedChurnProbability", ascending=False).head(4)
        list_html = ""
        for _, row in top_vulnerable.iterrows():
            list_html += list_item_row(
                title=f"Account {row['CustomerID']}",
                meta=f"{row['Contract']} • {row['Tenure']} mo tenure",
                value=f"${row['MonthlyCharges']:.0f}/mo",
                icon="🚨" if row["RiskLevel"] == "Critical Risk" else "⚠️",
            )
        st.markdown(list_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)

    # Portfolio Snapshot Table
    section_header("PORTFOLIO SNAPSHOT", "Detailed view of high risk accounts.")
    snapshot_df = filtered.sort_values("PredictedChurnProbability", ascending=False)[
        ["CustomerID", "Contract", "MonthlyCharges", "Tenure", "PredictedChurnProbability", "RiskLevel"]
    ].head(15)
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
