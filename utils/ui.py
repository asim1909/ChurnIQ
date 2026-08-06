from __future__ import annotations

from pathlib import Path
import textwrap

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
STYLE_FILE = ASSETS_DIR / "style.css"


def inject_css() -> None:
    """Load the shared minimalist enterprise theme into Streamlit."""
    if STYLE_FILE.exists():
        st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render a minimalist enterprise hero header."""
    badge_html = f'<span class="trend-pill neutral">{badge}</span>' if badge else ""
    html = textwrap.dedent(
        f"""
        <div class="hero-shell">
          <div class="hero-topline">⚡ CHURNIQ ENTERPRISE PLATFORM</div>
          <div class="hero-body">
            <div class="hero-copy">
              <h1 class="hero-title">{title}</h1>
              <p class="hero-subtitle">{subtitle}</p>
            </div>
            <div class="hero-meta">{badge_html}</div>
          </div>
        </div>
        """
    ).strip()
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None, badge: str | None = None) -> None:
    """Render a clean card-style header."""
    badge_html = f'<span class="trend-pill neutral">{badge}</span>' if badge else ""
    subtitle_html = f'<div style="color:#64748b; font-size:0.85rem; margin-top:0.2rem;">{subtitle}</div>' if subtitle else ""
    html = textwrap.dedent(
        f"""
        <div class="card-header-row">
          <div class="card-title-group">
            <div>
              <div class="card-kicker">{title}</div>
              {subtitle_html}
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:0.5rem;">
            {badge_html}
          </div>
        </div>
        """
    ).strip()
    st.markdown(html, unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str,
    hint: str | None = None,
    icon: str | None = None,
    delta: str | None = None,
    delta_type: str = "up",
) -> None:
    """Render a minimalist white KPI card with optional trend pill and icon."""
    delta_class = "up" if delta_type == "up" else ("down" if delta_type == "down" else "neutral")
    arrow = "↗" if delta_type == "up" else ("↘" if delta_type == "down" else "•")
    delta_html = f'<span class="trend-pill {delta_class}">{arrow} {delta}</span>' if delta else ""
    hint_html = f'<div class="kpi-hint"><span>💡</span> {hint}</div>' if hint else ""
    icon_html = f'<span style="font-size:1.1rem; opacity:0.85;">{icon}</span>' if icon else ""

    html = textwrap.dedent(
        f"""
        <div class="kpi-card">
          <div class="kpi-card-top">
            <div class="kpi-label">{label}</div>
            <div style="display:flex; align-items:center; gap:0.4rem;">
              {delta_html}
              {icon_html}
            </div>
          </div>
          <div class="kpi-value-group">
            <div class="kpi-value">{value}</div>
          </div>
          {hint_html}
        </div>
        """
    ).strip()
    st.markdown(html, unsafe_allow_html=True)


def list_item_row(title: str, meta: str, value: str, icon: str = "📦") -> str:
    """Return clean HTML snippet for a list row."""
    return textwrap.dedent(
        f"""
        <div class="list-item-row">
          <div class="list-item-left">
            <div class="list-item-thumb">{icon}</div>
            <div>
              <div class="list-item-title">{title}</div>
              <div class="list-item-meta">{meta}</div>
            </div>
          </div>
          <div class="list-item-value">{value}</div>
        </div>
        """
    ).strip()


def render_insight_cards(insights: list[str]) -> None:
    """Render executive insights as minimal visual cards."""
    icons = ["⚡", "💡", "📈", "🛡️", "🎯"]
    cards_html = ""
    for idx, insight in enumerate(insights):
        icon = icons[idx % len(icons)]
        card_class = "insight-card"
        if "churn" in insight.lower() or "vulnerable" in insight.lower() or "risk" in insight.lower():
            card_class += " warn"
        if "critical" in insight.lower() or "high" in insight.lower():
            card_class += " danger"
        cards_html += textwrap.dedent(
            f"""
            <div class="{card_class}">
              <div class="insight-icon">{icon}</div>
              <div class="insight-text">{insight}</div>
            </div>
            """
        ).strip()
    st.markdown(cards_html, unsafe_allow_html=True)


def risk_class(level: str) -> str:
    return {
        "Very Low Risk": "trend-pill up",
        "Low Risk": "trend-pill up",
        "Medium Risk": "trend-pill neutral",
        "High Risk": "trend-pill down",
        "Critical Risk": "trend-pill down",
    }.get(level, "trend-pill neutral")


def risk_badge(level: str) -> str:
    return f'<span class="{risk_class(level)}">{level}</span>'


def render_data_uploader_sidebar() -> pd.DataFrame:
    """Render a sidebar section for uploading custom CSV dataset and downloading template."""
    from utils.preprocessing import load_customer_data

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Data Source & Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom Customer CSV",
        type=["csv"],
        help="Upload your own customer dataset CSV to score churn risk and analyze your portfolio.",
    )

    if uploaded_file is not None:
        try:
            data = load_customer_data(uploaded_file, target_rows=0)
            st.session_state["custom_data"] = data
            st.session_state["data_source_name"] = uploaded_file.name
            st.sidebar.success(f"Active: **{uploaded_file.name}** ({len(data):,} rows)")
        except Exception as err:
            st.sidebar.error(f"Error parsing CSV: {err}")

    if "custom_data" in st.session_state:
        if st.sidebar.button("🔄 Reset to Default Dataset", use_container_width=True):
            del st.session_state["custom_data"]
            if "data_source_name" in st.session_state:
                del st.session_state["data_source_name"]
            st.rerun()

    # Template Download
    default_df = load_customer_data()
    template_csv = default_df.head(20).to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "📄 Download CSV Schema Template",
        template_csv,
        "churniq_csv_template.csv",
        "text/csv",
        use_container_width=True,
    )

    if "custom_data" in st.session_state:
        return st.session_state["custom_data"]
    return load_customer_data()
