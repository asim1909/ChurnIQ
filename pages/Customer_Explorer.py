from __future__ import annotations

from pathlib import Path
import textwrap

import pandas as pd
import streamlit as st

from utils.metrics import shap_explanation
from utils.preprocessing import customer_key_search, load_artifacts, load_customer_data, score_customers
from utils.recommendations import recommend_retention_actions
from utils.ui import inject_css, kpi_card, page_header, risk_badge, section_header

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Customer Explorer - ChurnIQ",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    page_header("Customer Intelligence Dossier", "Search any account ID to inspect churn risk, read SHAP model drivers, and view next-best retention actions.", badge="Drill-down")

    query = st.text_input("🔎 Search Customer Account ID", placeholder="Enter exact Customer ID (e.g. 7590-VHVEG) or partial match")
    matches = customer_key_search(scored, query)
    if matches.empty:
        st.info("💡 Enter a customer ID above to open their interactive retention dossier.")
        st.markdown("##### Sample High Risk Customer Accounts:")
        sample_ids = scored[scored["RiskLevel"].isin(["High Risk", "Critical Risk"])]["CustomerID"].head(5).tolist()
        st.write(" | ".join([f"`{cid}`" for cid in sample_ids]))
        return

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    selected = matches.iloc[0].copy()
    customer_frame = pd.DataFrame([selected[artifacts.training_columns or [c for c in scored.columns if c not in {"Churn", "PredictedChurnProbability", "RiskLevel", "RevenueAtRisk"}]]])
    local_drivers, global_drivers = shap_explanation(artifacts, customer_frame)
    recommendation = recommend_retention_actions(selected, local_drivers)

    # Customer Overview Banner
    b1, b2, b3 = st.columns([1.5, 1.2, 1.3])
    with b1:
        st.markdown(f"### Account: `{selected['CustomerID']}`")
        st.markdown(risk_badge(selected["RiskLevel"]), unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        st.caption(f"Contract: **{selected.get('Contract', 'Unknown')}** | Payment: **{selected.get('PaymentMethod', 'Unknown')}**")
    with b2:
        prob = selected['PredictedChurnProbability']
        st.metric("Predicted Churn Probability", f"{prob:.1%}")
        st.metric("Risk Score Index", f"{prob * 100:.0f}/100")
    with b3:
        st.metric("Monthly Revenue (ARPU)", f"${selected['MonthlyCharges']:.2f}")
        st.metric("Account Tenure", f"{int(selected['Tenure'])} months")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Explainability & Recommendations
    left, right = st.columns([1.1, 0.9])
    with left:
        section_header("SHAP MODEL DRIVERS", "local features driving churn score")
        if local_drivers.empty:
            st.write("Model explanation could not be generated for this customer.")
        else:
            st.dataframe(local_drivers.head(8), use_container_width=True, hide_index=True)

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

        section_header("RECOMMENDED RETENTION PLAYBOOK", "ranked next-best actions")
        actions = recommendation["actions"]
        rationales = recommendation["rationales"]
        actions_list = list(actions) if isinstance(actions, (list, tuple)) else [str(actions)]
        rationales_list = list(rationales) if isinstance(rationales, (list, tuple)) else [str(rationales)]
        for act, rat in zip(actions_list, rationales_list):
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="insight-card success">
                      <div class="insight-icon">🎯</div>
                      <div class="insight-text"><strong>{act}</strong>: {rat}</div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
    with right:
        section_header("GLOBAL FEATURE BENCHMARKS", "broader portfolio drivers")
        st.dataframe(global_drivers.head(10), use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)

    # Customer Attributes Snapshot
    section_header("ACCOUNT ATTRIBUTE SNAPSHOT", "raw feature values used during scoring")
    snapshot_fields = ["Gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "InternetService", "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    available_fields = [f for f in snapshot_fields if f in selected.index]
    st.dataframe(selected[available_fields].to_frame(name="Value"), use_container_width=True)


if __name__ == "__main__":
    render()
