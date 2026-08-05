# AI Customer Churn Intelligence Dashboard

An enterprise-style, AI-powered customer churn intelligence platform built with Python, Streamlit, Pandas, scikit-learn, Plotly, and SHAP.

## Project Overview

This dashboard predicts customer churn, quantifies revenue at risk, explains churn drivers, recommends retention actions, and simulates campaign economics in a format designed to feel like an internal analytics product used by large consulting and technology organizations.

## Business Problem

Churn is not only a customer experience issue. It is a direct revenue leakage problem. Teams need a single command center to identify which customers are likely to leave, understand why they are at risk, and prioritize retention spend against the highest-value accounts.

## Architecture

- `models/train_model.py` trains and compares candidate classifiers.
- `models/saved_model.pkl` stores the best trained pipeline and metadata after training.
- `models/preprocessor.pkl` stores the fitted preprocessing transformer after training.
- `utils/preprocessing.py` handles data loading, normalization, feature engineering, scoring, and artifact loading.
- `utils/metrics.py` handles performance metrics, ROC/PR curves, learning curves, feature importance, and SHAP explainability helpers.
- `utils/recommendations.py` powers retention actions and campaign simulation.
- `utils/insights.py` generates executive insights and customer segmentation.
- `pages/` contains the Streamlit multipage experience.
- `app.py` provides the executive overview and navigation hub.

## Features

- Churn probability scoring with five risk levels.
- Executive KPI cards for churn, revenue at risk, and retention ROI.
- Interactive Plotly visualizations for churn, risk, revenue, and customer segmentation.
- Model comparison across Logistic Regression, Decision Tree, Random Forest, and optional XGBoost.
- SHAP-based global and local explainability.
- Customer Explorer for searching and inspecting a single customer.
- Retention Simulator for discount, success-rate, and budget scenarios.
- Downloadable CSV outputs for leadership and operations.
- Streamlit caching for responsive performance.

## Technology Stack

Python, Pandas, NumPy, scikit-learn, Plotly, Streamlit, Joblib, SHAP, Matplotlib, and optional XGBoost.

## Screenshots

Add dashboard screenshots to the `screenshots/` folder after running the app locally.

## Installation

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model and generate the persisted artifacts:

```bash
python models/train_model.py
```

4. Launch the dashboard:

```bash
streamlit run app.py
```

## Usage

- Use the sidebar filters on the dashboard to segment the portfolio.
- Open Customer Explorer to search a single customer and view SHAP explanations.
- Open Model Performance to review classification metrics and curves.
- Open Retention Simulator to test retention budgets and discount strategies.

## Data

The project expects a Telco Customer Churn style dataset in `churn.csv`. If the file is missing or very small, the training pipeline creates a realistic demo dataset so the dashboard can still run end-to-end.

## Future Improvements

- Add automated model monitoring and drift detection.
- Persist retention scenarios and campaign results to a database.
- Introduce scheduled retraining and model registry integration.
- Add role-based access control and audit logging.
- Extend explainability with segmented SHAP summaries and cohort drift analysis.
