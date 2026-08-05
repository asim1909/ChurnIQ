from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.metrics import shap_explanation
from utils.preprocessing import customer_key_search, load_artifacts, load_customer_data, score_customers
from utils.recommendations import recommend_retention_actions
from utils.ui import inject_css, page_header, risk_badge

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
    scored["CustomerID"] = data["CustomerID"].astype(str)
    scored["Churn"] = pd.to_numeric(data["Churn"], errors="coerce").fillna(0).astype(int)
    return scored, artifacts


def render() -> None:
    inject_css()
    scored, artifacts = load_state()
    page_header("Customer Explorer", "Search any customer, inspect churn probability, read model drivers, and trigger next-best-action recommendations.", badge="Drill-down")

    query = st.text_input("Search Customer ID", placeholder="Enter a customer ID or partial match")
    matches = customer_key_search(scored, query)
    if matches.empty:
        st.info("Search for a customer ID to view the detailed retention dossier.")
        return

    selected = matches.iloc[0].copy()
    customer_frame = pd.DataFrame([selected[artifacts.training_columns or [c for c in scored.columns if c not in {"Churn", "PredictedChurnProbability", "RiskLevel", "RevenueAtRisk"}]]])
    local_drivers, global_drivers = shap_explanation(artifacts, customer_frame)
    recommendation = recommend_retention_actions(selected, local_drivers)

    c1, c2, c3 = st.columns([1.3, 1.3, 1.2])
    with c1:
        st.markdown(f"### {selected['CustomerID']}")
        st.markdown(risk_badge(selected["RiskLevel"]), unsafe_allow_html=True)
        st.metric("Predicted Churn Probability", f"{selected['PredictedChurnProbability']:.1%}")
        st.metric("Risk Score", f"{selected['PredictedChurnProbability'] * 100:.0f}/100")
    with c2:
        st.metric("Monthly Charges", f"${selected['MonthlyCharges']:.2f}")
        st.metric("Total Charges", f"${selected['TotalCharges']:.2f}")
        st.metric("Tenure", f"{int(selected['Tenure'])} months")
    with c3:
        st.metric("Contract", selected.get("Contract", "Unknown"))
        st.metric("Payment Method", selected.get("PaymentMethod", "Unknown"))
        st.metric("Recommended Action", recommendation["primary_action"])

    left, right = st.columns([1.1, 0.9])
    with left:
        section_header("SHAP Explanation", "The strongest local drivers behind this churn score.")
        if local_drivers.empty:
            st.write("Model explanation could not be generated for this customer.")
        else:
            st.dataframe(local_drivers.head(8), use_container_width=True, hide_index=True)
        section_header("Recommended Retention Actions", "Actions ranked by likely business impact.")
        for action, rationale in zip(recommendation["actions"], recommendation["rationales"]):
            st.markdown(f"- **{action}**: {rationale}")
    with right:
        section_header("Global Feature Importance", "The model's broader view across the portfolio.")
        st.dataframe(global_drivers.head(10), use_container_width=True, hide_index=True)

    section_header("Customer Snapshot", "Core account attributes used by the scoring engine.")
    snapshot_fields = ["Gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "InternetService", "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    st.dataframe(selected[snapshot_fields].to_frame(name="Value"), use_container_width=True)


if __name__ == "__main__":
    render()
