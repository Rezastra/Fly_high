"""
FLY HIGH — Engine Health Console
Streamlit frontend for the CMAPSS Predictive Maintenance FastAPI backend.

Run with:
    streamlit run app.py
"""

import io
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fly High · Engine Health Console",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "Low": "#3ADD9A",
    "Medium": "#FFB020",
    "High": "#FF5C5C",
}
RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}
DEFAULT_COLOR = "#8CA0C4"

# --------------------------------------------------------------------------
# Custom CSS — cockpit / instrument-panel aesthetic
# --------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(255,176,32,0.06), transparent 40%),
                radial-gradient(circle at 85% 10%, rgba(61,221,151,0.05), transparent 35%),
                #0B1220;
        }

        /* Runway-strip divider — the signature motif */
        .runway {
            height: 2px;
            width: 100%;
            margin: 0.4rem 0 1.6rem 0;
            background-image: repeating-linear-gradient(
                90deg,
                #FFB020 0px, #FFB020 22px,
                transparent 22px, transparent 40px
            );
            opacity: 0.55;
        }

        .callsign {
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: #8CA0C4;
            text-transform: uppercase;
        }

        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 2.6rem;
            color: #E8EDF4;
            margin-bottom: 0.1rem;
            letter-spacing: -0.01em;
        }

        .hero-sub {
            font-family: 'Inter', sans-serif;
            color: #8CA0C4;
            font-size: 1.0rem;
            margin-bottom: 0.2rem;
        }

        .panel {
            background: #121A2B;
            border: 1px solid #1E2A44;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
        }

        .metric-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            color: #8CA0C4;
            text-transform: uppercase;
        }

        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: #E8EDF4;
        }

        .badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            display: inline-block;
            letter-spacing: 0.06em;
        }

        section[data-testid="stSidebar"] {
            background-color: #0E1524;
            border-right: 1px solid #1E2A44;
        }

        .streamlit-expanderHeader {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
        }

        div[data-testid="stExpander"] {
            background: #121A2B;
            border: 1px solid #1E2A44;
            border-radius: 10px;
        }

        .stButton > button {
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.05em;
            border-radius: 6px;
            border: 1px solid #FFB020;
            color: #FFB020;
            background: transparent;
        }
        .stButton > button:hover {
            background: rgba(255,176,32,0.12);
            border-color: #FFB020;
            color: #FFB020;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def runway_divider():
    st.markdown('<div class="runway"></div>', unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return (
        f'<span class="badge" style="background:{color}22;'
        f'color:{color};border:1px solid {color}66;">{text}</span>'
    )


# --------------------------------------------------------------------------
# Backend calls
# --------------------------------------------------------------------------
def api_health(base_url: str):
    try:
        r = requests.get(f"{base_url}/api/health", timeout=5)
        return r.status_code == 200, r.json() if r.ok else r.text
    except requests.RequestException as e:
        return False, str(e)


def api_model_info(base_url: str):
    try:
        r = requests.get(f"{base_url}/api/model-info", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


def api_predict(base_url: str, file_bytes: bytes, filename: str):
    try:
        files = {"file": (filename, file_bytes)}
        r = requests.post(f"{base_url}/api/predict-dataset", files=files, timeout=120)
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        detail = ""
        try:
            detail = f" — {e.response.json()}"
        except Exception:
            pass
        return None, f"{e}{detail}"


def api_feature_importance(base_url: str, top_n: int = 10):
    try:
        r = requests.get(
            f"{base_url}/api/feature-importance", params={"top_n": top_n}, timeout=15
        )
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


# --------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------
def gauge_figure(rul: float, risk: str, max_scale: float = 150):
    color = RISK_COLORS.get(risk, DEFAULT_COLOR)
    max_scale = max(max_scale, rul * 1.3, 40)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul,
            number={
                "suffix": " cyc",
                "font": {"family": "JetBrains Mono", "size": 30, "color": "#E8EDF4"},
            },
            gauge={
                "axis": {
                    "range": [0, max_scale],
                    "tickcolor": "#8CA0C4",
                    "tickfont": {"family": "JetBrains Mono", "size": 10, "color": "#8CA0C4"},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#0B1220",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_scale * 0.2], "color": "#2A1418"},
                    {"range": [max_scale * 0.2, max_scale * 0.5], "color": "#2A2214"},
                    {"range": [max_scale * 0.5, max_scale], "color": "#12241C"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.9,
                    "value": rul,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EDF4", "family": "Inter"},
    )
    return fig


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert '#RRGGBB' to an 'rgba(r,g,b,a)' string Plotly accepts."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def history_figure(history: list, risk: str):
    color = RISK_COLORS.get(risk, DEFAULT_COLOR)
    cycles = [h["cycle"] for h in history]
    ruls = [h["predicted_rul"] for h in history]
    fig = go.Figure(
        go.Scatter(
            x=cycles,
            y=ruls,
            mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.1),
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Cycle", gridcolor="#1E2A44", color="#8CA0C4"),
        yaxis=dict(title="Predicted RUL", gridcolor="#1E2A44", color="#8CA0C4"),
        font={"family": "Inter", "color": "#E8EDF4", "size": 11},
    )
    return fig


def risk_bar_figure(counts: dict):
    order = ["High", "Medium", "Low"]
    order = [r for r in order if r in counts]
    fig = go.Figure(
        go.Bar(
            x=[counts[r] for r in order],
            y=order,
            orientation="h",
            marker_color=[RISK_COLORS.get(r, DEFAULT_COLOR) for r in order],
            text=[counts[r] for r in order],
            textposition="outside",
            textfont={"family": "JetBrains Mono", "color": "#E8EDF4"},
        )
    )
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(color="#E8EDF4", tickfont={"family": "JetBrains Mono", "size": 12}),
    )
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def sidebar():
    st.sidebar.markdown('<div class="callsign">Fly High · Control Panel</div>', unsafe_allow_html=True)
    st.sidebar.markdown("### Backend connection")
    base_url = st.sidebar.text_input(
        "API base URL", value=st.session_state.get("base_url", "http://127.0.0.1:8000")
    )
    st.session_state["base_url"] = base_url.rstrip("/")

    if st.sidebar.button("Check connection", width="stretch"):
        ok, info = api_health(st.session_state["base_url"])
        st.session_state["health_ok"] = ok
        st.session_state["health_info"] = info

    if "health_ok" in st.session_state:
        if st.session_state["health_ok"]:
            st.sidebar.success("Backend online")
        else:
            st.sidebar.error(f"Unreachable: {st.session_state['health_info']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload flight data")
    uploaded = st.sidebar.file_uploader(
        "CMAPSS .txt or .csv", type=["txt", "csv"], accept_multiple_files=False
    )
    analyze = st.sidebar.button("Analyze fleet", type="primary", width="stretch")

    st.sidebar.markdown("---")
    with st.sidebar.expander("Model info"):
        if st.button("Fetch model info", width="stretch"):
            info, err = api_model_info(st.session_state["base_url"])
            if err:
                st.error(err)
            else:
                st.json(info)

    return uploaded, analyze


# --------------------------------------------------------------------------
# Main sections
# --------------------------------------------------------------------------
def render_hero():
    st.markdown('<div class="callsign">CMAPSS · Turbofan Fleet Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Fly High — Engine Health Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Upload sensor logs, get remaining useful life, risk, '
        'and degradation trend for every engine in the fleet.</div>',
        unsafe_allow_html=True,
    )
    runway_divider()


def render_fleet_overview(data: dict):
    engines = data["engines"]
    df = pd.DataFrame(
        [
            {
                "engine_id": e["engine_id"],
                "current_cycle": e["current_cycle"],
                "predicted_rul": e["predicted_rul"],
                "risk": e["risk"],
                "recommendation": e["recommendation"],
            }
            for e in engines
        ]
    )
    counts = df["risk"].value_counts().to_dict()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.6])
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Engines in fleet</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{data["engine_count"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg predicted RUL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{df["predicted_rul"].mean():.1f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        worst = df.loc[df["predicted_rul"].idxmin()]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Most urgent engine</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">#{int(worst["engine_id"])}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Risk distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_bar_figure(counts), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    return df


def render_engine_cards(data: dict, df: pd.DataFrame):
    st.markdown("#### Engine detail")
    risk_filter = st.multiselect(
        "Filter by risk", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )
    engines = [e for e in data["engines"] if e["risk"] in risk_filter]
    engines.sort(key=lambda e: (RISK_ORDER.get(e["risk"], 9), e["predicted_rul"]))

    if not engines:
        st.info("No engines match the current filter.")
        return

    for e in engines:
        color = RISK_COLORS.get(e["risk"], DEFAULT_COLOR)
        header = f'Engine #{e["engine_id"]} · cycle {e["current_cycle"]} · {e["risk"]} risk'
        with st.expander(header, expanded=(e["risk"] == "High")):
            gcol, hcol = st.columns([1, 1.4])
            with gcol:
                st.plotly_chart(
                    gauge_figure(e["predicted_rul"], e["risk"]),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"gauge_{e['engine_id']}",
                )
                st.markdown(badge(e["risk"].upper(), color), unsafe_allow_html=True)
                st.markdown(f"**Recommendation:** {e['recommendation']}")
            with hcol:
                st.markdown('<div class="metric-label">Degradation trend</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    history_figure(e["history"], e["risk"]),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"hist_{e['engine_id']}",
                )

    runway_divider()
    st.markdown("#### Fleet summary table")
    st.dataframe(
        df.sort_values("predicted_rul")[["engine_id", "current_cycle", "predicted_rul", "risk", "recommendation"]],
        width="stretch",
        hide_index=True,
    )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download summary as CSV", data=csv, file_name="fleet_summary.csv", mime="text/csv")


def render_feature_importance():
    runway_divider()
    st.markdown("#### Feature importance")
    top_n = st.slider("Top N features", min_value=3, max_value=21, value=10)
    if st.button("Load feature importance"):
        info, err = api_feature_importance(st.session_state["base_url"], top_n)
        if err:
            st.error(err)
            return
        items = info.get("feature_importance") or info.get("importances") or info
        if isinstance(items, dict):
            names = list(items.keys())
            vals = list(items.values())
        elif isinstance(items, list):
            names = [i.get("feature", i.get("name")) for i in items]
            vals = [i.get("importance", i.get("value")) for i in items]
        else:
            st.json(info)
            return
        fig = go.Figure(
            go.Bar(
                x=vals,
                y=names,
                orientation="h",
                marker_color="#FFB020",
            )
        )
        fig.update_layout(
            height=max(220, 28 * len(names)),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1E2A44", color="#8CA0C4"),
            yaxis=dict(color="#E8EDF4", autorange="reversed"),
            font={"family": "Inter", "color": "#E8EDF4"},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --------------------------------------------------------------------------
# App entry point
# --------------------------------------------------------------------------
def main():
    inject_css()
    uploaded, analyze = sidebar()
    render_hero()

    if analyze:
        if uploaded is None:
            st.warning("Upload a CMAPSS .txt or .csv file first.")
        else:
            with st.spinner("Running inference on the fleet..."):
                result, err = api_predict(
                    st.session_state["base_url"], uploaded.getvalue(), uploaded.name
                )
            if err:
                st.error(f"Prediction failed: {err}")
            else:
                st.session_state["result"] = result

    if "result" in st.session_state:
        df = render_fleet_overview(st.session_state["result"])
        runway_divider()
        render_engine_cards(st.session_state["result"], df)
        render_feature_importance()
    else:
        st.markdown(
            '<div class="panel">Upload a dataset in the sidebar and click '
            '<b>Analyze fleet</b> to see engine health, risk, and degradation trends.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()