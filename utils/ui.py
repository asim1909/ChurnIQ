from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
STYLE_FILE = ASSETS_DIR / "style.css"


def inject_css() -> None:
    """Load the shared enterprise theme into Streamlit."""

    if STYLE_FILE.exists():
        st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render a consistent hero banner."""

    badge_html = f'<span class="risk-pill risk-medium">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-topline">AI CHURN INTELLIGENCE PLATFORM</div>
          <div class="hero-body">
            <div class="hero-copy">
              <h1 class="hero-title">{title}</h1>
              <p class="hero-subtitle">{subtitle}</p>
            </div>
            <div class="hero-meta">{badge_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None, badge: str | None = None) -> None:
    """Render a polished section header for dashboard content blocks."""

    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    badge_html = f'<span class="risk-pill risk-low">{badge}</span>' if badge else ""
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


def kpi_card(label: str, value: str, hint: str | None = None) -> None:
    """Render a single KPI card."""

    hint_html = f'<div class="kpi-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_class(level: str) -> str:
    return {
        "Very Low Risk": "risk-very-low",
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
        "Critical Risk": "risk-critical",
    }.get(level, "risk-medium")


def risk_badge(level: str) -> str:
    return f'<span class="risk-pill {risk_class(level)}">{level}</span>'
