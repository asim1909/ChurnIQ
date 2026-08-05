from __future__ import annotations

import numpy as np
import pandas as pd


def recommend_retention_actions(customer_row: pd.Series, driver_frame: pd.DataFrame | None = None) -> dict[str, str | list[str]]:
    """Generate rule-based retention actions informed by customer data and drivers."""

    actions: list[str] = []
    rationales: list[str] = []

    monthly = float(customer_row.get("MonthlyCharges", 0) or 0)
    tenure = float(customer_row.get("Tenure", 0) or 0)
    contract = str(customer_row.get("Contract", ""))
    internet = str(customer_row.get("InternetService", ""))
    tech_support = str(customer_row.get("TechSupport", ""))
    online_security = str(customer_row.get("OnlineSecurity", ""))
    payment = str(customer_row.get("PaymentMethod", ""))

    if contract == "Month-to-month" or tenure < 12:
        actions.append("Upgrade to Annual Contract")
        rationales.append("Short commitment periods are a primary churn driver.")
    if monthly >= 80:
        actions.append("Offer Discount")
        rationales.append("High recurring charges materially increase churn risk.")
    if tech_support in {"No", "No internet service"}:
        actions.append("Offer Free Tech Support")
        rationales.append("Service support gaps are correlated with dissatisfaction.")
    if online_security in {"No", "No internet service"} and internet != "No":
        actions.append("Bundle Security Add-On")
        rationales.append("Security add-ons reduce perceived risk and improve stickiness.")
    if payment == "Electronic check":
        actions.append("Incentivize Auto-Pay")
        rationales.append("Electronic check customers churn more often than auto-pay customers.")
    if not actions:
        actions = ["Offer Loyalty Reward"]
        rationales = ["Customer is not showing a single dominant risk driver, so a goodwill gesture is appropriate."]

    if driver_frame is not None and not driver_frame.empty:
        top_driver = driver_frame.iloc[0]["feature"]
        rationales.insert(0, f"Top model driver: {top_driver}.")

    return {
        "primary_action": actions[0],
        "actions": actions[:3],
        "rationales": rationales[:3],
    }


def simulate_retention_campaign(customers: pd.DataFrame, discount_rate: float, success_rate: float, budget: float) -> dict[str, float]:
    """Estimate campaign economics for a selected risk segment."""

    if customers.empty:
        return {"campaign_cost": 0.0, "revenue_saved": 0.0, "profit": 0.0, "roi": 0.0, "customers_saved": 0.0}

    target_customers = customers.copy()
    avg_monthly = float(target_customers["MonthlyCharges"].mean()) if "MonthlyCharges" in target_customers else 0.0
    churned_revenue = float((target_customers["MonthlyCharges"] * 12).sum()) if "MonthlyCharges" in target_customers else 0.0
    cost_per_customer = avg_monthly * 12 * (discount_rate / 100.0)
    if cost_per_customer <= 0:
        cost_per_customer = max(avg_monthly * 12 * 0.05, 1.0)

    budget_limited_targets = int(min(len(target_customers), budget / cost_per_customer if cost_per_customer else len(target_customers)))
    customer_count = max(budget_limited_targets, 0)
    campaign_cost = customer_count * cost_per_customer
    customers_saved = customer_count * (success_rate / 100.0)
    revenue_saved = customers_saved * avg_monthly * 12
    profit = revenue_saved - campaign_cost
    roi = profit / campaign_cost if campaign_cost else 0.0

    return {
        "campaign_cost": float(campaign_cost),
        "revenue_saved": float(revenue_saved),
        "profit": float(profit),
        "roi": float(roi),
        "customers_saved": float(customers_saved),
        "revenue_at_risk": float(churned_revenue),
    }
