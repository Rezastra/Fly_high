"""
FLY HIGH — Engine Health Console
Streamlit frontend for the CMAPSS Predictive Maintenance FastAPI backend.

Run with:
    streamlit run app.py
"""

import html as html_escape
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
import streamlit.components.v1 as components

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
    "Low": "#12B76A",
    "Medium": "#F79009",
    "High": "#F04438",
}
RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}
DEFAULT_COLOR = "#5F6368"
ACCENT_BLUE = "#20BEFF"
DARK_TEXT = "#202124"
TEXT_SECONDARY = "#5F6368"
GRID_COLOR = "#E1E3E6"

# --------------------------------------------------------------------------
# Custom CSS — cockpit / instrument-panel aesthetic
# --------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background: #FFFFFF;
        }

        /* Section divider — clean thin accent-blue hairline */
        .runway {
            height: 3px;
            width: 100%;
            margin: 0.4rem 0 1.6rem 0;
            background: #20BEFF;
            border-radius: 3px;
            opacity: 1;
        }

        .callsign {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            letter-spacing: 0.1em;
            font-size: 0.72rem;
            color: #5F6368;
            text-transform: uppercase;
        }

        .hero-title {
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            font-size: 2.6rem;
            margin-bottom: 0.1rem;
            letter-spacing: -0.02em;
            color: #202124;
        }

        .hero-sub {
            font-family: 'Inter', sans-serif;
            color: #5F6368;
            font-size: 1.0rem;
            margin-bottom: 0.2rem;
        }

        .panel {
            background: #FFFFFF;
            border: 1px solid #E1E3E6;
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 1px 2px rgba(32,33,36,0.04);
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .panel:hover {
            border-color: #20BEFF;
            box-shadow: 0 4px 14px rgba(32,33,36,0.08);
        }

        .metric-label {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            color: #5F6368;
            text-transform: uppercase;
        }

        .metric-value {
            font-family: 'Inter', sans-serif;
            font-size: 2.1rem;
            font-weight: 800;
            color: #202124;
        }

        .badge {
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            display: inline-block;
            letter-spacing: 0.03em;
            background: #FFFFFF;
        }

        section[data-testid="stSidebar"] {
            background-color: #F7F7F8;
            border-right: 1px solid #E1E3E6;
        }

        .streamlit-expanderHeader {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            color: #202124 !important;
        }

        div[data-testid="stExpander"] {
            background: #FFFFFF;
            border: 1px solid #E1E3E6;
            border-radius: 14px;
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
        }
        div[data-testid="stExpander"]:hover {
            border-color: #20BEFF;
            box-shadow: 0 4px 14px rgba(32,33,36,0.08);
        }

        /* Headings & body text */
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #202124;
        }

        /* Primary action buttons — pill, black, bold white text */
        .stButton > button[kind="primary"] {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: 0.01em;
            border-radius: 50px;
            border: none;
            background: #202124;
            color: #FFFFFF;
            padding: 0.5rem 1.4rem;
            transition: background 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button[kind="primary"]:hover {
            background: #3C4043;
            box-shadow: 0 2px 10px rgba(32,33,36,0.25);
            color: #FFFFFF;
        }

        /* Secondary buttons — pill, white, thin gray border */
        .stButton > button[kind="secondary"] {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            letter-spacing: 0.01em;
            border-radius: 50px;
            border: 1px solid #E1E3E6;
            background: #FFFFFF;
            color: #202124;
            padding: 0.5rem 1.4rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: #20BEFF;
            color: #20BEFF;
            box-shadow: 0 2px 8px rgba(32,190,255,0.18);
        }

        /* Text inputs — pill-shaped, white, thin gray border */
        .stTextInput > div > div > input {
            border-radius: 50px !important;
            border: 1px solid #E1E3E6 !important;
            background: #FFFFFF !important;
            color: #202124 !important;
            padding: 0.5rem 1.1rem !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #20BEFF !important;
            box-shadow: 0 0 0 1px #20BEFF !important;
        }

        /* File uploader — rounded, subtle border, dark text */
        [data-testid="stFileUploaderDropzone"] {
            background: #FFFFFF;
            border: 1.5px dashed #E1E3E6;
            border-radius: 16px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #20BEFF;
        }
        [data-testid="stFileUploaderDropzone"] button {
            border-radius: 50px !important;
            border: 1px solid #E1E3E6 !important;
            background: #FFFFFF !important;
            color: #202124 !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            border-color: #20BEFF !important;
            color: #20BEFF !important;
        }

        /* Multiselect — filter-chip style tags */
        span[data-baseweb="tag"] {
            background: #FFFFFF !important;
            border: 1px solid #E1E3E6 !important;
            color: #202124 !important;
            border-radius: 999px !important;
            font-weight: 600;
        }
        div[data-baseweb="select"] > div {
            border-radius: 16px !important;
            border-color: #E1E3E6 !important;
            background: #FFFFFF !important;
        }
        div[data-baseweb="select"] > div:focus-within {
            border-color: #20BEFF !important;
            box-shadow: 0 0 0 1px #20BEFF !important;
        }

        /* Slider — accent blue */
        div[data-testid="stSlider"] [role="slider"] {
            background-color: #20BEFF !important;
            border-color: #20BEFF !important;
        }
        div[data-testid="stSlider"] .st-emotion-cache-1dx1gwv,
        div[data-testid="stSlider"] div[style*="background-color: rgb(255"] {
            background-color: #20BEFF !important;
        }

        /* Links / small accents */
        a { color: #20BEFF !important; }

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
        f'<span class="badge" style="background:#FFFFFF;'
        f'color:{color};border:1.5px solid {color};">{text}</span>'
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
                "font": {"family": "Inter", "size": 30, "color": DARK_TEXT},
            },
            gauge={
                "axis": {
                    "range": [0, max_scale],
                    "tickcolor": TEXT_SECONDARY,
                    "tickfont": {"family": "Inter", "size": 10, "color": TEXT_SECONDARY},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#F1F3F4",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_scale * 0.2], "color": "#FCE8E6"},
                    {"range": [max_scale * 0.2, max_scale * 0.5], "color": "#FEF7E0"},
                    {"range": [max_scale * 0.5, max_scale], "color": "#E6F4EA"},
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
        font={"color": DARK_TEXT, "family": "Inter"},
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
            fillcolor=hex_to_rgba(color, 0.14),
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Cycle", gridcolor=GRID_COLOR, color=TEXT_SECONDARY),
        yaxis=dict(title="Predicted RUL", gridcolor=GRID_COLOR, color=TEXT_SECONDARY),
        font={"family": "Inter", "color": DARK_TEXT, "size": 11},
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
            textfont={"family": "Inter", "color": DARK_TEXT},
        )
    )
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(color=DARK_TEXT, tickfont={"family": "Inter", "size": 12}),
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


def _engine_card_grid_html(engines: list) -> str:
    """Build one self-contained HTML/CSS/JS document: a responsive grid of
    compact cards that expand on hover to reveal the full gauge + trend
    chart. Plotly is loaded from CDN inside this document because Streamlit
    components can't be nested inside a custom :hover container otherwise.
    """
    cards, data_blocks, render_calls = [], [], []

    for e in engines:
        eid = e["engine_id"]
        risk = e["risk"]
        color = RISK_COLORS.get(risk, DEFAULT_COLOR)
        rul = e["predicted_rul"]
        pct = max(0.0, min(rul / 150.0, 1.0)) * 100
        reco = html_escape.escape(e["recommendation"])

        gauge_json = pio.to_json(gauge_figure(rul, risk))
        line_json = pio.to_json(history_figure(e["history"], risk))
        gauge_div, line_div = f"gauge-{eid}", f"line-{eid}"
        gauge_data_id, line_data_id = f"gdata-{eid}", f"ldata-{eid}"

        cards.append(f"""
        <div class="engine-card">
          <div class="card-row">
            <span class="engine-id">Engine #{eid}</span>
            <span class="risk-pill" style="color:{color};border-color:{color};">{risk.upper()}</span>
          </div>
          <div class="cycle-text">Cycle {e["current_cycle"]} &middot; {rul:.1f} cyc RUL</div>
          <div class="progress-track">
            <div class="progress-fill" style="width:{pct:.1f}%;background:{color};"></div>
          </div>
          <div class="card-detail">
            <div class="detail-label">Remaining useful life</div>
            <div id="{gauge_div}" class="gauge-div"></div>
            <div class="detail-label">Degradation trend</div>
            <div id="{line_div}" class="line-div"></div>
            <div class="detail-reco"><b>Recommendation:</b> {reco}</div>
          </div>
        </div>
        """)
        data_blocks.append(f'<script type="application/json" id="{gauge_data_id}">{gauge_json}</script>')
        data_blocks.append(f'<script type="application/json" id="{line_data_id}">{line_json}</script>')
        render_calls.append(
            f"renderChart('{gauge_data_id}', '{gauge_div}'); "
            f"renderChart('{line_data_id}', '{line_div}');"
        )

    return f"""
    <html>
    <head>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0; padding: 4px;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow: visible;
        }}
        .engine-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
            padding-bottom: 24px;
        }}
        .engine-card {{
            position: relative;
            background: #FFFFFF;
            border: 1px solid {GRID_COLOR};
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(32,33,36,0.04);
            transition: box-shadow .2s ease, border-color .2s ease;
        }}
        .engine-card:hover {{
            border-color: {ACCENT_BLUE};
            box-shadow: 0 8px 24px rgba(32,33,36,0.14);
            z-index: 20;
        }}
        .card-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
        .engine-id {{ font-weight: 700; color: {DARK_TEXT}; font-size: 0.95rem; }}
        .cycle-text {{ color: {TEXT_SECONDARY}; font-size: 0.78rem; margin-bottom: 10px; }}
        .risk-pill {{
            font-size: 0.66rem; font-weight: 700; padding: 2px 10px;
            border-radius: 999px; border: 1.5px solid; letter-spacing: .03em;
            background: #FFFFFF;
        }}
        .progress-track {{ background: #F1F3F4; border-radius: 999px; height: 6px; overflow: hidden; }}
        .progress-fill {{ height: 100%; border-radius: 999px; transition: width .2s ease; }}
        .card-detail {{
            position: absolute; top: calc(100% + 8px); left: 0; right: 0;
            background: #FFFFFF; border: 1px solid {GRID_COLOR}; border-radius: 14px;
            padding: 14px 16px 16px; box-shadow: 0 12px 32px rgba(32,33,36,0.18);
            opacity: 0; pointer-events: none; transform: translateY(-6px);
            transition: opacity .18s ease, transform .18s ease; z-index: 30;
        }}
        .engine-card:hover .card-detail {{
            opacity: 1; pointer-events: auto; transform: translateY(0);
        }}
        .detail-label {{
            font-size: 0.66rem; font-weight: 700; letter-spacing: .08em;
            text-transform: uppercase; color: {TEXT_SECONDARY}; margin: 8px 0 2px;
        }}
        .detail-reco {{ font-size: 0.84rem; color: {DARK_TEXT}; margin-top: 8px; line-height: 1.4; }}
        .gauge-div {{ width: 100%; height: 170px; }}
        .line-div {{ width: 100%; height: 130px; }}
    </style>
    </head>
    <body>
        <div class="engine-grid">
            {''.join(cards)}
        </div>
        {''.join(data_blocks)}
        <script>
            function renderChart(dataId, targetId) {{
                var raw = document.getElementById(dataId).textContent;
                var fig = JSON.parse(raw);
                Plotly.newPlot(targetId, fig.data, fig.layout, {{displayModeBar: false, responsive: true}});
            }}
            {''.join(render_calls)}

            function resizeFrame() {{
                try {{
                    var h = document.body.scrollHeight;
                    window.frameElement.style.height = (h + 40) + 'px';
                }} catch (e) {{}}
            }}
            window.addEventListener('load', resizeFrame);
            new ResizeObserver(resizeFrame).observe(document.body);
            setTimeout(resizeFrame, 300);
            setTimeout(resizeFrame, 800);
        </script>
    </body>
    </html>
    """


def render_engine_cards(data: dict, df: pd.DataFrame):
    st.markdown("#### Engine detail")
    st.caption("Hover a card to see the full RUL gauge, degradation trend, and recommendation.")
    risk_filter = st.multiselect(
        "Filter by risk", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )
    engines = [e for e in data["engines"] if e["risk"] in risk_filter]
    engines.sort(key=lambda e: (RISK_ORDER.get(e["risk"], 9), e["predicted_rul"]))

    if not engines:
        st.info("No engines match the current filter.")
        return

    components.html(_engine_card_grid_html(engines), height=420, scrolling=False)

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
                marker_color=ACCENT_BLUE,
            )
        )
        fig.update_layout(
            height=max(220, 28 * len(names)),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor=GRID_COLOR, color=TEXT_SECONDARY),
            yaxis=dict(color=DARK_TEXT, autorange="reversed"),
            font={"family": "Inter", "color": DARK_TEXT},
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