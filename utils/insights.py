from __future__ import annotations

import pandas as pd


def _share_text(count: int, total: int) -> str:
    return f"{(count / total * 100):.1f}%" if total else "0.0%"


def generate_executive_insights(df: pd.DataFrame) -> list[str]:
    """Create business-ready insights from the scored customer base."""

    insights: list[str] = []
    total = len(df)
    if total == 0:
        return ["No customer records are available for insight generation."]

    churn_col = "Churn" if "Churn" in df.columns else None
    churn_rate = df[churn_col].mean() if churn_col else 0.0
    if "Contract" in df.columns and churn_col:
        churn_by_contract = df.groupby("Contract")[churn_col].mean().sort_values(ascending=False)
        top_contract = churn_by_contract.index[0]
        top_rate = churn_by_contract.iloc[0]
        insights.append(f"{top_rate:.0%} churn is observed in {top_contract} contracts, which is the most vulnerable segment.")
    if "Tenure" in df.columns and churn_col:
        short_tenure = df.loc[df["Tenure"] < 12, churn_col].mean() if (df["Tenure"] < 12).any() else 0.0
        insights.append(f"Customers with tenure below 12 months churn at {short_tenure:.0%}, far above the portfolio average of {churn_rate:.0%}.")
    if "InternetService" in df.columns and "MonthlyCharges" in df.columns:
        revenue_risk = df.groupby("InternetService")["MonthlyCharges"].sum().sort_values(ascending=False)
        insights.append(f"{revenue_risk.index[0]} customers contribute the highest revenue pool and require the strongest save plan.")
    if "PaymentMethod" in df.columns and churn_col:
        payment_churn = df.groupby("PaymentMethod")[churn_col].mean().sort_values(ascending=False)
        insights.append(f"{payment_churn.index[0]} users show the highest churn probability and should be prioritized for digital retention offers.")
    if "RiskLevel" in df.columns:
        high_risk_share = _share_text(int(df["RiskLevel"].isin(["High Risk", "Critical Risk"]).sum()), total)
        insights.append(f"{high_risk_share} of the book is in high or critical risk, creating an immediate retention opportunity.")
    return insights[:5]


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple business segmentation for dashboards."""

    frame = df.copy()
    tenure_bin = pd.cut(frame["Tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    charge_bin = pd.cut(frame["MonthlyCharges"], bins=[0, 40, 70, 100, 200], labels=["Low", "Mid", "High", "Premium"])
    frame["Segment"] = tenure_bin.astype(str) + " / " + charge_bin.astype(str)
    return frame[["CustomerID", "Segment"] + [c for c in ["RiskLevel", "PredictedChurnProbability", "MonthlyCharges", "Tenure"] if c in frame.columns]]
