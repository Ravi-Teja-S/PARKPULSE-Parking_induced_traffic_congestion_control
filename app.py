import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
import pydeck as pdk
import joblib
import altair as alt
import shap
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION & VIEWPORT CSS
# ==========================================
st.set_page_config(page_title="ParkPulse Command Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* 1. PREVENT GLOBAL SCROLL BAR JUMPING BUT ALLOW SCROLLING */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        overflow-y: auto !important; /* Changed from hidden to allow smaller screens to scroll */
    }

    /* 2. SECURE PADDING TO KEEP HEADER VISIBLE AT THE VERY TOP */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* 3. HIDE NATIVE STREAMLIT HEADERS */
    header {display: none !important;}
    footer {display: none !important;}

    /* 4. CUSTOM CYBERPUNK SCROLLBARS */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #31333F; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00a4ff; }

    /* 5. METRIC CARDS */
    .metric-card {
        background-color: #1e212b; border-top: 3px solid #31333F;
        padding: 4px; /* Reduced from 10px */
        border-radius: 5px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title { color: #8a8d93; font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;}
    .metric-value { color: #ffffff; font-size: 18px; /* Reduced from 24px */ font-weight: 700; margin: 0px; /* Removed margin */ font-family: monospace;}
    .accent-red { border-top-color: #ff4b4b; }
    .accent-blue { border-top-color: #00a4ff; }
    .accent-orange { border-top-color: #ffa421; }
    .accent-green { border-top-color: #21c354; }
    .sub-text { color: #666; font-size: 8px; font-family: monospace; line-height: 1;}

    /* 6. COMPONENT STYLING */
    div.stButton > button:first-child {
        font-family: monospace; font-weight: bold; font-size: 12px;
        border-radius: 4px; border: 1px solid #31333F;
    }

    /* --- SLEEK PILL-SHAPED RADIO TOGGLE --- */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: inline-flex;
        background-color: #0e1117; /* Dark background container */
        border-radius: 50px;
        padding: 4px;
        border: 1px solid #31333F; /* Subtle outer border */
        gap: 4px;
    }

    /* Individual Toggle Options */
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: transparent;
        padding: 6px 18px !important;
        border-radius: 50px !important;
        cursor: pointer;
        margin: 0;
        transition: all 0.2s ease;
        border: 1px solid transparent; /* Prevents layout shift on active state */
    }

    /* Hide the native radio circles */
    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Style the text */
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        margin: 0 !important;
        font-family: monospace;
        font-size: 12px;
        font-weight: 700;
        color: #8a8d93; /* Dim inactive text */
    }

    /* ACTIVE STATE: When the hidden input is checked */
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #1e212b !important; /* Lighter active background */
        border: 1px solid #31333F;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* ACTIVE TEXT: Make the selected text pop */
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE INITIALIZATION
# ==========================================
if "map_center" not in st.session_state: st.session_state.map_center = [12.9716, 77.5946]
if "map_zoom" not in st.session_state: st.session_state.map_zoom = 12
if "is_animating" not in st.session_state: st.session_state.is_animating = False
if "advisor_view" not in st.session_state: st.session_state.advisor_view = "brief"

def set_map_view(lat, lon, zoom):
    st.session_state.map_center = [lat, lon]
    st.session_state.map_zoom = zoom

def toggle_animation():
    st.session_state.is_animating = not st.session_state.is_animating

# ==========================================
# LOAD ARTIFACTS
# ==========================================
@st.cache_data
def load_data():
    # Adding safe fallbacks in case files are missing during testing
    try:
        df = pd.read_parquet("data/processed/hotspot_features.parquet")
        history = pd.read_csv("data/processed/hotspot_history.csv")
        bias = pd.read_csv("data/processed/reporting_bias.csv")
    except Exception as e:
        st.error(f"Data loading error: {e}. Please ensure data pipeline is run.")
        st.stop()

    try:
        shap_df = pd.read_csv("data/processed/feature_importance.csv")
    except:
        shap_df = pd.DataFrame({
            "Feature": ["rolling_3h_avg", "avg_impact", "historical_risk", "lag_1h", "hour"],
            "Importance_Score": [0.45, 0.28, 0.11, 0.06, 0.03]
        })

    # Create a mapping from grid_lat/lon to a representative location string
    # Group by grid_lat, grid_lon and get the first non-null location
    grid_location_map = df.groupby(['grid_lat', 'grid_lon'])['location'].first().reset_index()
    grid_location_map = grid_location_map.rename(columns={'location': 'resolved_location'})

    return df, history, bias, shap_df, grid_location_map

@st.cache_resource
def load_model():
    try:
        return joblib.load("models/forecast_model.pkl")
    except Exception as e:
        st.error(f"Model loading error: {e}. Ensure model is trained.")
        st.stop()

df, history, bias, shap_df, grid_location_map = load_data()
xgb_model = load_model()

# ==========================================
# TOP RIBBON & CONTROLS
# ==========================================
top1, top2 = st.columns([6, 4])
with top1:
    # Changed from <h3> to <h4> and forced margins to 0
    st.markdown("<h4 style='color: #00a4ff; margin: 0px; padding: 0px; font-family: monospace;'>PARKPULSE · COMMAND CENTER</h4>", unsafe_allow_html=True)
    st.markdown("<div style='color: #8a8d93; font-size: 10px; font-family: monospace; letter-spacing: 1px; margin-top: -5px;'>BENGALURU TRAFFIC POLICE · AI ENFORCEMENT CONSOLE</div>", unsafe_allow_html=True)
with top2:
    # Removed the <br> tag that was pushing the radio button down
    app_view = st.radio("VIEW", ["🔵 OPERATE", "🧠 EXPLAIN"], horizontal=True, label_visibility="collapsed")

# Tightened the horizontal line margins
st.markdown("<hr style='margin: 2px 0px 8px 0px; border-color: #333;'>", unsafe_allow_html=True)

col_anim, col_date, col_slider, col_sim = st.columns([1.5, 2, 6, 2.5])
with col_anim:
    st.button("▷ ANIMATE" if not st.session_state.is_animating else "⏹ STOP", on_click=toggle_animation, use_container_width=True)
with col_date:
    selected_date = st.date_input("DATE", value=df['date'].min(), min_value=df['date'].min(), max_value=df['date'].max(), label_visibility="collapsed")
with col_slider:
    selected_hour = st.slider("HOUR", 0, 23, 10, label_visibility="collapsed")
with col_sim:
    simulated_now = pd.to_datetime(f"{selected_date} {selected_hour}:00:00").tz_localize("Asia/Kolkata")
    st.markdown(f"<div style='text-align: right; color: #8a8d93; font-family: monospace; font-size: 12px; margin-top: 5px;'>SIM · {simulated_now.strftime('%Y-%m-%dT%H:%M:%S')}</div>", unsafe_allow_html=True)
cutoff_time = simulated_now - pd.Timedelta(minutes=60)
future_hour = (selected_hour + 1) % 24

# Handle bias confidence logic safely
confidence_arr = bias[bias["hour"] == selected_hour]["reporting_confidence"].values
confidence = confidence_arr[0] if len(confidence_arr) > 0 else 0.5

# ==========================================
# BACKEND ML PIPELINE
# ==========================================
live_df = df[(df["created_ist"] <= simulated_now) & (df["created_ist"] >= cutoff_time)].copy()
live_stats = pd.DataFrame()

if len(live_df) > 10:
    coords = np.radians(live_df[["latitude", "longitude"]].dropna().values)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="haversine")
    live_df["live_cluster"] = clusterer.fit_predict(coords)
    active = live_df[live_df["live_cluster"] != -1]
    if not active.empty:
        live_stats = active.groupby("live_cluster").agg(
            live_violations=("device_id", "count"),
            live_impact=("base_impact_score", "sum"),
            grid_lat=("grid_lat", "first"),
            grid_lon=("grid_lon", "first")
        ).reset_index()
        live_stats["live_density_score"] = (live_stats["live_impact"] / 500 * 100).clip(upper=100)

dispatch_grid = history[["grid_lat", "grid_lon", "historical_risk_score"]].copy()
if not live_stats.empty:
    dispatch_grid = dispatch_grid.merge(live_stats[["grid_lat", "grid_lon", "live_density_score"]], on=["grid_lat", "grid_lon"], how="left")

dispatch_grid["live_density_score"] = dispatch_grid.get("live_density_score", pd.Series([0]*len(dispatch_grid))).fillna(0)
dispatch_grid["hour"] = future_hour
dispatch_grid["dow"] = simulated_now.dayofweek if future_hour > selected_hour else (simulated_now.dayofweek + 1) % 7
dispatch_grid["lag_1h_violations"] = dispatch_grid["live_density_score"] * 0.1
dispatch_grid["rolling_3h_avg"] = dispatch_grid["historical_risk_score"] * 0.3
dispatch_grid["avg_impact"] = 25.0

xgb_features = ["hour", "dow", "historical_risk_score", "lag_1h_violations", "rolling_3h_avg", "avg_impact"]

# Predict
dispatch_grid["xgb_probability"] = xgb_model.predict_proba(dispatch_grid[xgb_features])[:, 1]
W_live, W_hist, W_xgb = (0.4, 0.2, 0.4) if confidence >= 0.30 else (0.0, 0.5, 0.5)
dispatch_grid["final_risk_score"] = ((dispatch_grid["live_density_score"] * W_live) + (dispatch_grid["historical_risk_score"] * W_hist) + ((dispatch_grid["xgb_probability"] * 100) * W_xgb))
top_threats = dispatch_grid.sort_values(
    "final_risk_score",
    ascending=False
).head(20)
# Merge resolved locations into top_threats
top_threats = top_threats.merge(grid_location_map, on=['grid_lat', 'grid_lon'], how='left')
top_threats['resolved_location'] = top_threats['resolved_location'].fillna('Unknown Location')

# ==========================================
# GLOBAL KPIs
# ==========================================
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.markdown(f"<div class='metric-card accent-blue'><div class='metric-title'>Violations (Slice)</div><div class='metric-value'>⚠ {len(live_df):,}</div><div class='sub-text'>Last 60 Mins</div></div>", unsafe_allow_html=True)
with m2: st.markdown(f"<div class='metric-card accent-orange'><div class='metric-title'>Active Hotspots</div><div class='metric-value'>◎ {len(live_stats) if not live_stats.empty else 0}</div><div class='sub-text'>Spatio-Temporal</div></div>", unsafe_allow_html=True)
with m3: st.markdown(f"<div class='metric-card accent-red'><div class='metric-title'>Predicted Zones</div><div class='metric-value'>⭕ {len(top_threats)}</div><div class='sub-text'>Next Hour</div></div>", unsafe_allow_html=True)
with m4: st.markdown(f"<div class='metric-card accent-blue'><div class='metric-title'>Confidence</div><div class='metric-value'>⏱ {confidence*100:.0f}%</div><div class='sub-text'>{'⚠️ Failover' if confidence < 0.3 else 'Sensors OK'}</div></div>", unsafe_allow_html=True)
with m5: st.markdown(f"<div class='metric-card accent-green'><div class='metric-title'>Σ Delay (Min)</div><div class='metric-value'>∿ {len(live_df)*2.5:,.0f}</div><div class='sub-text'>Commuter-Min</div></div>", unsafe_allow_html=True)
with m6: st.markdown(f"<div class='metric-card accent-green'><div class='metric-title'>Peak Hour</div><div class='metric-value'>📍 {df['hour'].value_counts().idxmax()}:00</div><div class='sub-text'>Historical Mode</div></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

CONTAINER_HEIGHT = 560

# ==========================================
# VIEW ROUTING
# ==========================================
if app_view == "🔵 OPERATE":
    col_left, col_center, col_right = st.columns([2.5, 5.5, 3])

    # ---- LEFT COLUMN (SCROLLABLE PANE) ----
    with col_left:
        with st.container(height=CONTAINER_HEIGHT):
            # st.markdown("<div style='background-color: #1e212b; padding: 15px; border-radius: 5px; border-top: 2px solid #31333F;'>", unsafe_allow_html=True)
            st.markdown("<b style='color:#8a8d93; font-size:11px; font-family:monospace; letter-spacing:1px;'>VIOLATIONS · HOUR OF DAY</b>", unsafe_allow_html=True)

            hourly_dist = df.groupby('hour').size().reset_index(name='count')
            hourly_dist['color'] = np.where((hourly_dist['hour'] >= 9) & (hourly_dist['hour'] <= 12), '#ff4b4b', '#00a4ff')
            chart = alt.Chart(hourly_dist).mark_bar().encode(
                x=alt.X('hour:O', title=None, axis=alt.Axis(labelAngle=0, labelColor='#8a8d93')),
                y=alt.Y('count:Q', title=None, axis=alt.Axis(labelColor='#8a8d93')),
                color=alt.Color('color:N', scale=None)
            ).properties(height=140).configure_view(strokeWidth=0).configure_axis(grid=False)

            st.altair_chart(chart, use_container_width=True)

            st.markdown("<hr style='margin: 5px 0; border-color: #31333F;'>", unsafe_allow_html=True)
            st.markdown("<b style='color:#8a8d93; font-size:11px; font-family:monospace; letter-spacing:1px;'>VEHICLE-TYPE IMPACT</b>", unsafe_allow_html=True)

            v_stats = df["true_vehicle_type"].value_counts().head(5).reset_index()
            v_stats.columns = ['Type', 'Count']
            v_chart = alt.Chart(v_stats).mark_bar(color='#ffa421', size=12).encode(
                x=alt.X('Count:Q', title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y('Type:N', title=None, sort='-x', axis=alt.Axis(labelColor='#8a8d93', labelFont='monospace'))
            ).properties(height=130).configure_view(strokeWidth=0).configure_axis(grid=False)

            st.altair_chart(v_chart, use_container_width=True)

            # FIX: Combine the closing div and use a negative top margin (-30px)
            # to pull the title up over Streamlit's invisible chart padding.
            st.markdown("""
            </div>
            <div style='margin-top: -30px; margin-bottom: 0px;'>
                <b style='color:#21c354; font-size:12px; font-family:monospace; letter-spacing:1px;'>⚡ LIVE ALERTS FEED</b>
            </div>
            """, unsafe_allow_html=True)

            # --- THE NESTED SCROLLING LIVE FEED ---
            feed_html = "<div style='height: 200px; overflow-y: auto; padding-right: 5px; margin-top: 5px;'>"
            for _, row in live_df.sort_values("created_ist", ascending=False).head(30).iterrows():
                viol_type = str(row.get('violation_type', 'UNKNOWN')).upper().replace('[', '').replace(']', '').replace('"', '').replace("'", '')
                if len(viol_type) > 15: viol_type = viol_type[:15] + "..."
                loc = str(row.get('location', 'Unknown Location'))
                if len(loc) > 35: loc = loc[:35] + "..."
                v_type = str(row.get('true_vehicle_type', 'VEHICLE')).upper()
                time_str = row['created_ist'].strftime('%I:%M %p')

                block = f"<div style='margin-bottom: 12px; border-bottom: 1px solid #1e212b; padding-bottom: 8px;'><div style='display:flex; justify-content:space-between; font-family:monospace; font-size:10px;'><span style='color:#00a4ff; font-weight:bold;'>MONITOR • {viol_type}</span><span style='color:#8a8d93;'>{time_str}</span></div><div style='color:#fff; font-size:11px; margin: 4px 0; font-family:sans-serif;'>{loc}</div><div style='display:flex; justify-content:space-between; font-family:monospace; font-size:10px; color:#8a8d93;'><span>{v_type}</span><span>{row.get('base_impact_score', 10) * 0.2:.1f}m delay</span></div></div>"
                feed_html += block
            feed_html += "</div>"
            st.markdown(feed_html, unsafe_allow_html=True)

    # ---- CENTER COLUMN (HIGH-PERFORMANCE PYDECK MAP) ----
    with col_center:

        def get_color(score):
            if score >= 75: return [255, 75, 75, 200]       # Red
            elif score >= 50: return [255, 164, 33, 200]    # Orange
            else: return [0, 164, 255, 200]                 # Blue

        def get_action(score):
            if score >= 75: return "TOW IMMEDIATELY"
            elif score >= 50: return "DEPLOY PATROL"
            else: return "MONITOR"

        if not top_threats.empty:
            top_threats["color"] = top_threats["final_risk_score"].apply(get_color)
            top_threats["action_text"] = top_threats["final_risk_score"].apply(get_action)
            top_threats["radius"] = top_threats["final_risk_score"].apply(lambda x: max(150, x * 4))

            layer = pdk.Layer(
                'ScatterplotLayer',
                data=top_threats,
                get_position='[grid_lon, grid_lat]',
                get_color='color',
                get_radius='radius',
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=1,
                radius_min_pixels=5,
                radius_max_pixels=30,
            )

            view_state = pdk.ViewState(
                latitude=st.session_state.map_center[0],
                longitude=st.session_state.map_center[1],
                zoom=st.session_state.map_zoom,
                pitch=0
            )

            tooltip = {
                "html": "<b>Risk Score:</b> {final_risk_score} <br/>"
                        "<b>AI Probability:</b> {xgb_probability} <br/>"
                        "<b>Location:</b> {resolved_location} <br/>"
                        "<b>Action:</b> <span style='color: white;'>{action_text}</span>",
                "style": {"backgroundColor": "#1e212b", "color": "#8a8d93", "fontFamily": "monospace"}
            }

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            )

            # Map fills the container cleanly
            st.pydeck_chart(r, use_container_width=True)

        else:
            st.info("No active threats detected in this time slice.")

    # ---- RIGHT COLUMN (SCROLLABLE PANE) ----
    with col_right:
        with st.container(height=CONTAINER_HEIGHT):

            target = top_threats.iloc[0] if not top_threats.empty else None
            threat_level = "CRITICAL" if len(top_threats) > 5 else "ELEVATED" if len(top_threats) > 0 else "NOMINAL"

            # ==========================================
            # 🔧 WHAT-IF SIMULATOR (INTERACTIVE)
            # ==========================================
            st.markdown("""
            <div style='padding: 0px 0 5px 0;'>
                <b style='color:#00a4ff; font-size:12px; font-family:monospace;'>🔧 WHAT-IF SIMULATOR</b><br>
                <span style='color:#8a8d93; font-size:10px; font-family:monospace;'>Adjust deployment to model enforcement impact.</span>
            </div>
            """, unsafe_allow_html=True)

            # Custom styled sliders
            st.markdown("<div style='margin-top: 5px;'><b style='color:#8a8d93; font-size:10px; font-family:monospace;'>TOW TRUCKS DISPATCHED</b></div>", unsafe_allow_html=True)
            tow_trucks = st.slider("tow", 0, 5, 0, label_visibility="collapsed")

            st.markdown("<div style='margin-top: -5px;'><b style='color:#8a8d93; font-size:10px; font-family:monospace;'>PATROL UNITS DEPLOYED</b></div>", unsafe_allow_html=True)
            patrol_units = st.slider("patrol", 0, 10, 0, label_visibility="collapsed")

            # Simulation Logic
            if target is not None:
                base_risk = target['final_risk_score']
                # Tows reduce risk heavily, patrols reduce it moderately
                new_risk = max(5.0, base_risk - (tow_trucks * 18.5) - (patrol_units * 6.2))

                # Clearance time in minutes
                est_clearance = max(10, 65 - (tow_trucks * 12) - (patrol_units * 4))

                # Commuter delay minutes saved
                delay_saved = (base_risk - new_risk) * 2.4
            else:
                base_risk, new_risk, est_clearance, delay_saved = 0, 0, 0, 0

            # Dynamic Color Formatting for New Risk
            risk_color = "accent-green" if new_risk < 40 else "accent-orange" if new_risk < 75 else "accent-red"

            # Mini Simulated Metric Cards
            sim1, sim2, sim3 = st.columns(3)
            with sim1: st.markdown(f"<div class='metric-card {risk_color}' style='padding: 4px;'><div class='metric-title' style='font-size: 8px;'>NEW RISK</div><div class='metric-value' style='font-size: 16px;'>{new_risk:.1f}</div></div>", unsafe_allow_html=True)
            with sim2: st.markdown(f"<div class='metric-card accent-blue' style='padding: 4px;'><div class='metric-title' style='font-size: 8px;'>CLEARANCE</div><div class='metric-value' style='font-size: 16px;'>{int(est_clearance)}m</div></div>", unsafe_allow_html=True)
            with sim3: st.markdown(f"<div class='metric-card accent-green' style='padding: 4px;'><div class='metric-title' style='font-size: 8px;'>DELAY SAVED</div><div class='metric-value' style='font-size: 16px;'>{int(delay_saved)}m</div></div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 15px 0 10px 0; border-color: #31333F;'>", unsafe_allow_html=True)

            # ==========================================
            # ✨ AI ENFORCEMENT ADVISOR
            # ==========================================
            st.markdown("""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <b style='color:#00a4ff; font-size:13px; font-family:monospace;'>✨ AI ENFORCEMENT ADVISOR</b>
                <span style='color:#666; font-size:10px; font-family:monospace; text-align:right;'>GEMINI • FLASH</span>
            </div>
            """, unsafe_allow_html=True)

            # Initialize states for the advisor
            if "advisor_view" not in st.session_state: st.session_state.advisor_view = "brief"
            if "show_terminal" not in st.session_state: st.session_state.show_terminal = False
            if "briefing_text" not in st.session_state: st.session_state.briefing_text = ""

            # Callback: Reset the terminal when changing tabs
            def set_advisor_view(view):
                st.session_state.advisor_view = view
                st.session_state.show_terminal = False
                st.session_state.briefing_text = ""

            # Row 1: The three view buttons side-by-side
            cb1, cb2, cb3 = st.columns(3)
            cb1.button("PATROL BRIEF", type="primary" if st.session_state.advisor_view == "brief" else "secondary", on_click=set_advisor_view, args=("brief",), use_container_width=True)
            cb2.button("SHIFT PLAN", type="primary" if st.session_state.advisor_view == "shift" else "secondary", on_click=set_advisor_view, args=("shift",), use_container_width=True)
            cb3.button("EXEC SUMMARY", type="primary" if st.session_state.advisor_view == "exec" else "secondary", on_click=set_advisor_view, args=("exec",), use_container_width=True)

            # Row 2: The Generate button spanning the full width
            generate_clicked = st.button("⚡ GENERATE NEW BRIEFING", type="primary", use_container_width=True)

            # CSS for the terminal readout box
            st.markdown("""
            <style>
                .terminal-box {
                    background-color: #0e1117;
                    padding: 15px;
                    border-radius: 5px;
                    border: 1px solid #31333F;
                    border-left: 3px solid #00a4ff;
                    min-height: 150px;
                    margin-top: 10px;
                    margin-bottom: 10px;
                    font-size: 13px;
                    font-family: monospace;
                    color: #e0e0e0;
                    box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
                    overflow-y: auto;
                }
            </style>
            """, unsafe_allow_html=True)

            # Only show the terminal if Generate was clicked or if it was already showing
            if generate_clicked:
                st.session_state.show_terminal = True

                # --- PREPARE GEMINI PROMPT ---
                if st.session_state.advisor_view == "brief":
                    if target is not None:
                        prompt = f"""
                        Act as the AI Enforcement Advisor for the Bengaluru Traffic Police.
                        A severe gridlock threat has been detected at: {target['resolved_location']}.

                        Task: Generate a concise, tactical patrol briefing.
                        Format with bold headers and bullet points. Include Priority, Location (Full Address), Action, and Impact. Keep it highly tactical, under 100 words.
                        """
                    else:
                        prompt = "Act as the AI Enforcement Advisor for the Bengaluru Traffic Police. The city grid is stable. Give a 2-sentence confirmation that routine monitoring is active."

                elif st.session_state.advisor_view == "shift":
                    peak = df['hour'].value_counts().idxmax()
                    prompt = f"""
                    Act as the AI Enforcement Advisor for the Bengaluru Traffic Police.
                    Generate a Shift Optimization Plan. Peak demand is detected at {peak}:00 hours. The current threat level is {threat_level}.
                    Provide a tactical breakdown for Shift A, B, and C deployments. Format strictly with bolding and bullet points. Keep under 100 words.
                    """

                elif st.session_state.advisor_view == "exec":
                    prompt = f"""
                    Act as the AI Enforcement Advisor for the Bengaluru Traffic Police.
                    Generate an Executive Summary.
                    Grid Status: {threat_level}. Sensor Confidence: {confidence*100:.0f}%.
                    Active Load: {len(live_df)} violations. Forecast (+1H): {len(top_threats)} predicted critical zones.
                    Provide a high-level strategic note explaining the current traffic state in Bengaluru. Format tightly with bullet points. Under 100 words.
                    """

                # --- CALL GEMINI API ---
                try:
                    import google.generativeai as genai
                    import os

                    # Fetch API key securely
                    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

                    if not api_key:
                        st.error("⚠️ GEMINI_API_KEY is missing. Please add it to your environment variables or Streamlit secrets.")
                    else:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash') # Currently the fastest supported flash model

                        with st.spinner("Generating briefing..."):
                            # Create a single placeholder for the entire terminal box content
                            terminal_box_placeholder = st.empty()

                            # Stream the response live to the UI
                            full_response = ""
                            response = model.generate_content(prompt, stream=True)
                            for chunk in response:
                                full_response += chunk.text
                                # Update the placeholder with the content wrapped in the terminal box div
                                terminal_box_placeholder.markdown(f"<div class='terminal-box'>{full_response}▌</div>", unsafe_allow_html=True)

                            # Final render without the block cursor
                            terminal_box_placeholder.markdown(f"<div class='terminal-box'>{full_response}</div>", unsafe_allow_html=True)

                        # Save to state so it persists if the user clicks the map or a slider
                        st.session_state.briefing_text = full_response

                except Exception as e:
                    st.error(f"AI Generation Failed: {e}. Make sure `google-generativeai` is installed and your API key is valid.")

            # Keep displaying the generated text if the state is active
            elif st.session_state.show_terminal:
                st.markdown(f"<div class='terminal-box'>{st.session_state.briefing_text}</div>", unsafe_allow_html=True)
            # ==========================================
            # 🚨 DISPATCH QUEUE +1H
            # ==========================================
            st.markdown(f"<hr style='margin: 15px 0; border-color: #31333F;'><b style='color:#ff4b4b; font-size:13px; font-family:monospace;'>🚨 DISPATCH QUEUE +1H ({future_hour}:00)</b><hr style='margin: 10px 0; border-color: #31333F;'>", unsafe_allow_html=True)

            if not top_threats.empty:
                for idx, row in top_threats.head(4).iterrows():
                    action_color = "#ff4b4b" if row['final_risk_score'] >= 75 else "#ffa421" if row['final_risk_score'] >= 50 else "#00a4ff"
                    action_label = "TOW" if row['final_risk_score'] >= 75 else "PATROL" if row['final_risk_score'] >= 50 else "MONITOR"

                    cq_text, cq_btn = st.columns([8, 2])
                    with cq_text:
                        # Display the resolved location
                        st.markdown(f"<b style='color: {action_color}; font-size:12px; font-family:monospace;'>{action_label}</b> <span style='color: #8a8d93; font-size:12px;'>| Risk: {row['final_risk_score']:.1f}</span><br><span style='color: #fff; font-size:11px; font-family: monospace;'>Location: {row['resolved_location']}</span>", unsafe_allow_html=True)
                    with cq_btn:
                        st.button("🎯", key=f"fbtn_{idx}", on_click=set_map_view, args=(row['grid_lat'], row['grid_lon'], 16))
                    st.markdown("<hr style='margin: 5px 0; border-color: #31333F;'>", unsafe_allow_html=True)

            st.button("🌍 Reset Map", on_click=set_map_view, args=(12.9716, 77.5946, 12), use_container_width=True)

elif app_view == "🧠 EXPLAIN":
    # ==========================================
    # SHAP EXPLAINABILITY TAB
    # (Removed fixed container height so the long list of explanations can flow naturally)
    # ==========================================
    st.markdown("<div style='background-color: #1e212b; padding: 15px; border-radius: 5px; border-top: 2px solid #00a4ff;'>", unsafe_allow_html=True)
    st.markdown("<b style='color:#00a4ff; font-size:14px; font-family:monospace;'>🧠 MODEL EXPLAINABILITY (XGBoost · SHAP)</b><hr style='margin: 10px 0; border-color: #333;'>", unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    with e1: st.markdown("<span style='color:#8a8d93; font-size:11px; font-weight:bold; font-family:monospace;'>ACCURACY</span><br><span style='color:#21c354; font-size:24px; font-weight:bold; font-family:monospace;'>83.58%</span>", unsafe_allow_html=True)
    with e2: st.markdown("<span style='color:#8a8d93; font-size:11px; font-weight:bold; font-family:monospace;'>ROC-AUC</span><br><span style='color:#00a4ff; font-size:24px; font-weight:bold; font-family:monospace;'>0.896</span>", unsafe_allow_html=True)
    with e3: st.markdown(f"<span style='color:#8a8d93; font-size:11px; font-weight:bold; font-family:monospace;'>ROWS USED</span><br><span style='color:#fff; font-size:24px; font-weight:bold; font-family:monospace;'>1,84,411</span>", unsafe_allow_html=True)
    with e4: st.markdown("<span style='color:#8a8d93; font-size:11px; font-weight:bold; font-family:monospace;'>POSITIVE RATE</span><br><span style='color:#fff; font-size:24px; font-weight:bold; font-family:monospace;'>14.2%</span>", unsafe_allow_html=True)

    st.markdown("<br><b style='color:#fff; font-size:12px; font-family:monospace;'>GLOBAL FEATURE IMPORTANCE · MEAN(|SHAP|)</b>", unsafe_allow_html=True)

    shap_df['color'] = np.where(shap_df['Importance_Score'] == shap_df['Importance_Score'].max(), '#ff4b4b', '#00a4ff')
    chart = alt.Chart(shap_df).mark_bar().encode(
        x=alt.X('Importance_Score:Q', title=None, axis=alt.Axis(grid=False, labelColor='#8a8d93')),
        y=alt.Y('Feature:N', title=None, sort='-x', axis=alt.Axis(labelColor='#8a8d93', labelFont='monospace')),
        color=alt.Color('color:N', scale=None)
    ).properties(height=250).configure_view(strokeWidth=0).configure_axis(grid=False)

    st.altair_chart(chart, use_container_width=True)
    st.markdown("<span class='sub-text'>Higher bars = stronger average influence on the model's 'next-hour hotspot' prediction.</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- LOCAL SHAP EXPLANATION SCROLLING LIST ---
    st.markdown("<b style='color:#fff; font-size:14px; font-family:monospace; text-transform:uppercase;'>TOP-10 HIGHEST-PROBABILITY PREDICTIONS (LOCAL SHAP)</b><hr style='margin: 5px 0; border-color: #333;'>", unsafe_allow_html=True)

    if not top_threats.empty:
        
        @st.cache_resource
        def load_explainer(_model):
            return shap.TreeExplainer(_model)

        explainer = load_explainer(xgb_model)
        top_10 = top_threats.head(10).copy()
        local_shap_values = explainer.shap_values(top_10[xgb_features])

        for i, (idx, row) in enumerate(top_10.iterrows()):
            prob = row['xgb_probability']
            shap_row = local_shap_values[i]

            # FIX 1: Flattened the header HTML to prevent Markdown code block rendering
            html_block = f"<div style='margin-bottom: 20px;'>"
            html_block += f"<div style='display:flex; justify-content:space-between; font-family:monospace; color:#8a8d93; font-size:12px; margin-bottom: 5px;'>"
            html_block += f"<span><b>#{i+1} · {row['grid_lat']}_{row['grid_lon']}</b> | <span style='font-size:10px;'>hour={row['hour']} dow={row['dow']}</span></span>"
            html_block += f"<span><b style='color:#ffa421;'>P={prob:.3f}</b></span></div>"

            feat_shap = []
            for j, f in enumerate(xgb_features):
                feat_shap.append({'feat': f, 'val': row[f], 'shap': shap_row[j]})
            feat_shap = sorted(feat_shap, key=lambda x: abs(x['shap']), reverse=True)

            max_abs_shap = max(abs(s) for s in shap_row) if max(abs(s) for s in shap_row) > 0 else 1
            for fs in feat_shap:
                bar_color = "#ff4b4b" if fs['shap'] > 0 else "#00a4ff"
                bar_width = (abs(fs['shap']) / max_abs_shap) * 80
                sign = "+" if fs['shap'] > 0 else ""
                val_str = f"{fs['val']:.2f}" if isinstance(fs['val'], float) else str(fs['val'])

                # FIX 2: Flattened the SHAP bar HTML to prevent Markdown code block rendering
                html_block += f"<div style='display:flex; align-items:center; font-family:monospace; font-size:11px; margin: 2px 0;'>"
                html_block += f"<div style='width: 25%; color:#8a8d93;'>{fs['feat']}</div>"
                html_block += f"<div style='width: 10%; color:#666; text-align:right; padding-right:10px;'>{val_str}</div>"
                html_block += f"<div style='width: 55%;'><div style='height: 8px; background-color: {bar_color}; width: {bar_width}%; border-radius: 2px;'></div></div>"
                html_block += f"<div style='width: 10%; color:{bar_color}; text-align:right;'>{sign}{fs['shap']:.4f}</div></div>"

            html_block += "<hr style='margin: 10px 0; border-color: #1e212b;'></div>"
            st.markdown(html_block, unsafe_allow_html=True)
    else:
        st.info("No active threats predicted for the next hour to explain.")
