from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ChurnIQ - Minimalist Enterprise Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page("pages/Executive_Overview.py", title="Executive Overview", icon="🏠", default=True)
dashboard = st.Page("pages/Dashboard.py", title="Portfolio Dashboard", icon="📊")
explorer = st.Page("pages/Customer_Explorer.py", title="Customer Explorer", icon="🔎")
performance = st.Page("pages/Model_Performance.py", title="Model Performance", icon="📈")
simulator = st.Page("pages/Retention_Simulator.py", title="Retention Simulator", icon="🧪")

pg = st.navigation(
    {
        "Intelligence Platform": [overview, dashboard, explorer],
        "Analytics & QA": [performance, simulator],
    }
)

pg.run()
