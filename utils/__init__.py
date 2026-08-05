"""Utility helpers for the AI Customer Churn Intelligence Dashboard."""

from .preprocessing import (
    ModelArtifacts,
    build_preprocessor,
    canonicalize_columns,
    generate_demo_telco_data,
    load_artifacts,
    load_customer_data,
    prepare_model_frame,
    risk_label,
    score_customers,
)
from .recommendations import recommend_retention_actions, simulate_retention_campaign
from .insights import generate_executive_insights, segment_customers
