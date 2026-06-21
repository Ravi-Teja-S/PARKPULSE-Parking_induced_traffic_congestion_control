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
import folium
from folium.plugins import HeatMapWithTime
import streamlit.components.v1 as components


warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION & VIEWPORT CSS
# ==========================================
st.set_page_config(page_title="ParkPulse Command Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* 1. PREVENT GLOBAL SCROLL BAR JUMPING BUT ALLOW SCROLLING */
    [data-testid="stAppViewContainer"] { background-color: #0e1117; overflow-y: auto !important; }
    .block-container { padding-top: 0rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }
    header {display: none !important;} footer {display: none !important;}
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #31333F; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00a4ff; }

    /* 2. METRIC CARDS */
    .metric-card { background-color: #1e212b; border-top: 3px solid #31333F; padding: 4px; border-radius: 5px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-title { color: #8a8d93; font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;}
    .metric-value { color: #ffffff; font-size: 18px; font-weight: 700; margin: 0px; font-family: monospace;}
    .accent-red { border-top-color: #ff4b4b; } .accent-blue { border-top-color: #00a4ff; }
    .accent-orange { border-top-color: #ffa421; } .accent-green { border-top-color: #21c354; }
    .sub-text { color: #666; font-size: 8px; font-family: monospace; line-height: 1;}

    /* 3. COMPONENT STYLING */
    div.stButton > button:first-child { font-family: monospace; font-weight: bold; font-size: 12px; border-radius: 4px; border: 1px solid #31333F; }
    div[data-testid="stRadio"] > div[role="radiogroup"] { display: inline-flex; background-color: #0e1117; border-radius: 50px; padding: 4px; border: 1px solid #31333F; gap: 4px; }
    div[data-testid="stRadio"] div[role="radiogroup"] label { background-color: transparent; padding: 6px 18px !important; border-radius: 50px !important; cursor: pointer; margin: 0; transition: all 0.2s ease; border: 1px solid transparent; }
    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] label p { margin: 0 !important; font-family: monospace; font-size: 12px; font-weight: 700; color: #8a8d93; }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background-color: #1e212b !important; border: 1px solid #31333F; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color: #ffffff !important; }
    
    .terminal-box { background-color: #0e1117; padding: 15px; border-radius: 5px; border: 1px solid #31333F; border-left: 3px solid #00a4ff; min-height: 150px; margin-top: 10px; margin-bottom: 10px; font-size: 13px; font-family: monospace; color: #e0e0e0; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); overflow-y: auto; }
</style>
""", unsafe_allow_html=True)



# ==========================================
# STATE INITIALIZATION
# ==========================================
if "map_center" not in st.session_state: st.session_state.map_center = [12.9716, 77.5946]
if "map_zoom" not in st.session_state: st.session_state.map_zoom = 12
if "is_animating" not in st.session_state: st.session_state.is_animating = False
if "advisor_view" not in st.session_state: st.session_state.advisor_view = "brief"
# New Integrated States
if "selected_target" not in st.session_state: st.session_state.selected_target = None
if "sim_data" not in st.session_state: st.session_state.sim_data = {}
if "queue_status" not in st.session_state: st.session_state.queue_status = {}
# FIX: Add these two lines to initialize the terminal state!
if "show_terminal" not in st.session_state: st.session_state.show_terminal = False
if "briefing_text" not in st.session_state: st.session_state.briefing_text = ""
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
    try:
        df = pd.read_parquet("data/processed/hotspot_features.parquet")
        history = pd.read_csv("data/processed/hotspot_history.csv")
        bias = pd.read_csv("data/processed/reporting_bias.csv")
    except Exception as e:
        st.error(f"Data loading error: {e}. Please ensure data pipeline is run.")
        st.stop()
    try: shap_df = pd.read_csv("data/processed/feature_importance.csv")
    except: shap_df = pd.DataFrame({"Feature": ["rolling_3h_avg", "avg_impact", "historical_risk", "lag_1h", "hour"], "Importance_Score": [0.45, 0.28, 0.11, 0.06, 0.03]})
    
    grid_location_map = df.groupby(['grid_lat', 'grid_lon'])['location'].first().reset_index()
    grid_location_map = grid_location_map.rename(columns={'location': 'resolved_location'})
    return df, history, bias, shap_df, grid_location_map

@st.cache_resource
def load_model():
    try: return joblib.load("models/forecast_model.pkl")
    except Exception as e: st.error(f"Model loading error: {e}"); st.stop()

df, history, bias, shap_df, grid_location_map = load_data()
xgb_model = load_model()

# ==========================================
# TOP RIBBON & CONTROLS
# ==========================================
top1, top2 = st.columns([6, 4])
with top1:
    st.markdown("<h4 style='color: #00a4ff; margin: 0px; padding: 0px; font-family: monospace;'>PARKPULSE · COMMAND CENTER</h4>", unsafe_allow_html=True)
    st.markdown("<div style='color: #8a8d93; font-size: 10px; font-family: monospace; letter-spacing: 1px; margin-top: -5px;'>BENGALURU TRAFFIC POLICE · AI ENFORCEMENT CONSOLE</div>", unsafe_allow_html=True)
with top2:
    app_view = st.radio("VIEW", ["🔵 OPERATE", "🧠 EXPLAIN"], horizontal=True, label_visibility="collapsed")

st.markdown("<hr style='margin: 2px 0px 8px 0px; border-color: #333;'>", unsafe_allow_html=True)

col_anim, col_date, col_slider, col_sim = st.columns([1.5, 2, 6, 2.5])
with col_anim: st.button("▷ ANIMATE" if not st.session_state.is_animating else "⏹ STOP", on_click=toggle_animation, use_container_width=True)
with col_date: selected_date = st.date_input("DATE", value=df['date'].min(), min_value=df['date'].min(), max_value=df['date'].max(), label_visibility="collapsed")
with col_slider: selected_hour = st.slider("HOUR", 0, 23, 10, label_visibility="collapsed")
with col_sim:
    simulated_now = pd.to_datetime(f"{selected_date} {selected_hour}:00:00").tz_localize("Asia/Kolkata")
    st.markdown(f"<div style='text-align: right; color: #8a8d93; font-family: monospace; font-size: 12px; margin-top: 5px;'>SIM · {simulated_now.strftime('%Y-%m-%dT%H:%M:%S')}</div>", unsafe_allow_html=True)
cutoff_time = simulated_now - pd.Timedelta(minutes=60)
future_hour = (selected_hour + 1) % 24

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
            live_violations=("device_id", "count"), live_impact=("base_impact_score", "sum"),
            grid_lat=("grid_lat", "first"), grid_lon=("grid_lon", "first")
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
dispatch_grid["xgb_probability"] = xgb_model.predict_proba(dispatch_grid[xgb_features])[:, 1]
W_live, W_hist, W_xgb = (0.4, 0.2, 0.4) if confidence >= 0.30 else (0.0, 0.5, 0.5)
dispatch_grid["final_risk_score"] = ((dispatch_grid["live_density_score"] * W_live) + (dispatch_grid["historical_risk_score"] * W_hist) + ((dispatch_grid["xgb_probability"] * 100) * W_xgb))
top_threats = dispatch_grid.sort_values("final_risk_score", ascending=False).head(20)
top_threats = top_threats.merge(grid_location_map, on=['grid_lat', 'grid_lon'], how='left')
top_threats['resolved_location'] = top_threats['resolved_location'].fillna('Unknown Location')

# ==========================================
# GLOBAL KPIs
# ==========================================
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.markdown(f"<div class='metric-card accent-blue'><div class='metric-title'>Violations</div><div class='metric-value'>⚠ {len(live_df):,}</div></div>", unsafe_allow_html=True)
with m2: st.markdown(f"<div class='metric-card accent-orange'><div class='metric-title'>Active Hotspots</div><div class='metric-value'>◎ {len(live_stats) if not live_stats.empty else 0}</div></div>", unsafe_allow_html=True)
with m3: st.markdown(f"<div class='metric-card accent-red'><div class='metric-title'>Predicted Zones</div><div class='metric-value'>⭕ {len(top_threats)}</div></div>", unsafe_allow_html=True)
with m4: st.markdown(f"<div class='metric-card accent-blue'><div class='metric-title'>Confidence</div><div class='metric-value'>⏱ {confidence*100:.0f}%</div></div>", unsafe_allow_html=True)
with m5: st.markdown(f"<div class='metric-card accent-green'><div class='metric-title'>Σ Delay (Min)</div><div class='metric-value'>∿ {len(live_df)*2.5:,.0f}</div></div>", unsafe_allow_html=True)
with m6: st.markdown(f"<div class='metric-card accent-green'><div class='metric-title'>Peak Hour</div><div class='metric-value'>📍 {df['hour'].value_counts().idxmax()}:00</div></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

CONTAINER_HEIGHT = 450

# ==========================================
# VIEW ROUTING
# ==========================================
if app_view == "🔵 OPERATE":
    col_left, col_center, col_right = st.columns([2.5, 5.5, 3.5])

    # ---- LEFT COLUMN ----
    with col_left:
        with st.container(height=CONTAINER_HEIGHT):
            st.markdown("<b style='color:#8a8d93; font-size:11px; font-family:monospace; letter-spacing:1px;'>VIOLATIONS · HOUR OF DAY</b>", unsafe_allow_html=True)
            hourly_dist = df.groupby('hour').size().reset_index(name='count')
            hourly_dist['color'] = np.where((hourly_dist['hour'] >= 9) & (hourly_dist['hour'] <= 12), '#ff4b4b', '#00a4ff')
            chart = alt.Chart(hourly_dist).mark_bar().encode(x=alt.X('hour:O', title=None, axis=alt.Axis(labelAngle=0, labelColor='#8a8d93')), y=alt.Y('count:Q', title=None, axis=alt.Axis(labelColor='#8a8d93')), color=alt.Color('color:N', scale=None)).properties(height=140).configure_view(strokeWidth=0).configure_axis(grid=False)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("<hr style='margin: 5px 0; border-color: #31333F;'>", unsafe_allow_html=True)
            st.markdown("<b style='color:#8a8d93; font-size:11px; font-family:monospace; letter-spacing:1px;'>VEHICLE-TYPE IMPACT</b>", unsafe_allow_html=True)
            v_stats = df["true_vehicle_type"].value_counts().head(5).reset_index()
            v_stats.columns = ['Type', 'Count']
            v_chart = alt.Chart(v_stats).mark_bar(color='#ffa421', size=12).encode(x=alt.X('Count:Q', title=None, axis=alt.Axis(labels=False, ticks=False)), y=alt.Y('Type:N', title=None, sort='-x', axis=alt.Axis(labelColor='#8a8d93', labelFont='monospace'))).properties(height=130).configure_view(strokeWidth=0).configure_axis(grid=False)
            st.altair_chart(v_chart, use_container_width=True)

            st.markdown("""</div><div style='margin-top: -25px; margin-bottom: 5px;'><b style='color:#21c354; font-size:12px; font-family:monospace; letter-spacing:1px;'>⚡ LIVE ALERTS FEED</b></div>""", unsafe_allow_html=True)

            # Native Streamlit Scrollable Feed
            with st.container(height=230):
                for idx, row in live_df.sort_values("created_ist", ascending=False).head(20).iterrows():
                    viol_type = str(row.get('violation_type', 'UNKNOWN')).upper()[:15]
                    time_str = row['created_ist'].strftime('%I:%M %p')
                    # Instant offline lookup!
                    raw_loc = str(row.get('location', f"Grid {row['latitude']:.4f}, {row['longitude']:.4f}"))
                    short_addr = ", ".join(raw_loc.split(',')[:2]).strip()
                    
                    st.markdown(f"<div style='display:flex; justify-content:space-between; font-family:monospace; font-size:10px;'><span style='color:#00a4ff; font-weight:bold;'>MONITOR • {viol_type}</span><span style='color:#8a8d93;'>{time_str}</span></div>", unsafe_allow_html=True)
                    
                    c1, c2 = st.columns([7, 3])
                    with c1: st.markdown(f"<div style='color:#fff; font-size:11px; font-family:sans-serif;'>{short_addr}</div>", unsafe_allow_html=True)
                    with c2:
                        if st.button("🎯", key=f"feed_{idx}", use_container_width=True):
                            st.session_state.selected_target = {
                                'id': f"live_{idx}", 'lat': row['latitude'], 'lon': row['longitude'],
                                'risk': row.get('base_impact_score', 10) * 5, # Estimated raw risk for live items
                                'address': short_addr
                            }
                            set_map_view(row['latitude'], row['longitude'], 16)
                            st.rerun()
                    st.markdown("<hr style='margin: 4px 0; border-color: #1e212b;'>", unsafe_allow_html=True)

    # ---- CENTER COLUMN ----
    with col_center:
        if not st.session_state.is_animating:
            # ==========================================
            # STATIC STATE (NATIVE PYDECK)
            # ==========================================
            def get_color(score):
                if score >= 75: return [255, 75, 75, 200]
                elif score >= 50: return [255, 164, 33, 200]
                else: return [0, 164, 255, 200]

            if not top_threats.empty:
                top_threats["color"] = top_threats["final_risk_score"].apply(get_color)
                top_threats["radius"] = top_threats["final_risk_score"].apply(lambda x: max(150, x * 4))
                
                map_layers = []

                # Highlight target with Concentric Rings
                if st.session_state.selected_target:
                    tgt_lat, tgt_lon = st.session_state.selected_target['lat'], st.session_state.selected_target['lon']
                    target_df = pd.DataFrame([{'lat': tgt_lat, 'lon': tgt_lon}])
                    
                    inner_ring = pdk.Layer('ScatterplotLayer', data=target_df, get_position='[lon, lat]', get_fill_color=[0, 0, 0, 0], get_line_color=[255, 255, 0, 255], get_radius=150, stroked=True, filled=False, line_width_min_pixels=3)
                    outer_ring = pdk.Layer('ScatterplotLayer', data=target_df, get_position='[lon, lat]', get_fill_color=[0, 0, 0, 0], get_line_color=[255, 255, 0, 150], get_radius=300, stroked=True, filled=False, line_width_min_pixels=1)
                    map_layers.extend([inner_ring, outer_ring])

                # Pass a lightweight dataframe to the map
                map_df = top_threats[['grid_lat', 'grid_lon', 'color', 'radius', 'final_risk_score', 'resolved_location']]
                base_layer = pdk.Layer('ScatterplotLayer', data=map_df, get_position='[grid_lon, grid_lat]', get_color='color', get_radius='radius', pickable=True, opacity=0.8, stroked=True, filled=True, radius_scale=1, radius_min_pixels=5, radius_max_pixels=30)
                map_layers.insert(0, base_layer)

                view_state = pdk.ViewState(latitude=st.session_state.map_center[0], longitude=st.session_state.map_center[1], zoom=st.session_state.map_zoom, pitch=0)
                r = pdk.Deck(layers=map_layers, initial_view_state=view_state, tooltip={"html": "<b>Risk:</b> {final_risk_score}<br/><b>Loc:</b> {resolved_location}"})
                st.pydeck_chart(r, use_container_width=True)
            else:
                st.info("No active threats detected in this time slice.")
                
        else:
            # ==========================================
            # ANIMATED TIME-LAPSE STATE (NATIVE PYDECK)
            # ==========================================
            map_placeholder = st.empty()
            time_label = st.empty()
            
            # Loop through 60 minutes in 10-minute intervals
            for step in range(7):
                if not st.session_state.is_animating:
                    break # Allow user to stop animation midway
                
                fraction = step / 6.0
                step_risk = dispatch_grid["live_density_score"] + (dispatch_grid["final_risk_score"] - dispatch_grid["live_density_score"]) * fraction
                
                active_step = dispatch_grid[step_risk > 15].copy()
                active_step["weight"] = step_risk[active_step.index] / 100.0
                
                # Prevent map crashing if dataframe is empty
                if active_step.empty:
                    active_step = pd.DataFrame([{"grid_lat": st.session_state.map_center[0], "grid_lon": st.session_state.map_center[1], "weight": 0}])
                
                # Native PyDeck Heatmap for smooth blending
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=active_step,
                    get_position=["grid_lon", "grid_lat"],
                    get_weight="weight",
                    radiusPixels=50,
                    intensity=1.5,
                    threshold=0.05
                )
                
                view_state = pdk.ViewState(latitude=st.session_state.map_center[0], longitude=st.session_state.map_center[1], zoom=st.session_state.map_zoom, pitch=0)
                r = pdk.Deck(layers=[layer], initial_view_state=view_state)
                
                step_time = simulated_now + pd.Timedelta(minutes=step*10)
                
                # Update UI elements dynamically
                time_label.markdown(f"<div style='text-align:center; color:#00a4ff; font-size:14px; font-weight:bold; font-family:monospace; margin-top:10px;'>⏱ FORECAST: {step_time.strftime('%H:%M')}</div>", unsafe_allow_html=True)
                map_placeholder.pydeck_chart(r, use_container_width=True)
                
                # Pause to let the user see the frame
                time.sleep(0.6) 

            # Auto-reset animation state when finished
            if st.session_state.is_animating:
                st.session_state.is_animating = False
                st.rerun()
    # ---- RIGHT COLUMN (INTEGRATED STATE) ----
    with col_right:
        with st.container(height=CONTAINER_HEIGHT):

            # ==========================================
            # 1. WHAT-IF SIMULATOR
            # ==========================================
            
            st.markdown("""<div style='display: flex; align-items: center; gap: 5px; margin-bottom: 5px;'><span style='color:#00a4ff; font-size:16px;'>🔧</span><b style='color:#8a8d93; font-size:13px; font-family:monospace; letter-spacing: 1px;'>WHAT-IF SIMULATOR</b></div>""", unsafe_allow_html=True)
            @st.fragment
            def render_simulator():
                if st.session_state.selected_target is not None:
                    tgt = st.session_state.selected_target
                    st.markdown(f"<div style='color:#e0e0e0; font-size:11px; font-family:sans-serif; margin-bottom: 10px;'>📍 {tgt['address']}</div>", unsafe_allow_html=True)
                    
                    tow_pct = st.slider("TOW %", 0, 100, 50, label_visibility="collapsed")
                    patrol_intensity = st.slider("PATROL", 0.0, 1.0, 0.50, label_visibility="collapsed")

                    col_sim_btn, col_disp_btn = st.columns(2)
                    with col_sim_btn:
                        if st.button("SIMULATE", type="primary", use_container_width=True):
                            base_risk = tgt['risk']
                            base_delay = base_risk * 3.2 
                            delay_reduction = (tow_pct / 100.0 * 0.45) + (patrol_intensity * 0.25)
                            risk_reduction = (tow_pct / 100.0 * 0.35) + (patrol_intensity * 0.30)
                            
                            st.session_state.sim_data[tgt['id']] = {
                                'base_delay': base_delay, 'sim_delay': base_delay * (1 - delay_reduction),
                                'base_risk': base_risk, 'sim_risk': base_risk * (1 - risk_reduction),
                                'cleared': int(base_delay * delay_reduction * 0.06),
                                'del_pct': delay_reduction * -100, 'rsk_pct': risk_reduction * -100
                            }
                    
                    with col_disp_btn:
                        if st.button("DISPATCH", type="primary", use_container_width=True):
                            st.session_state.queue_status[tgt['id']] = {
                                'status': 'DISPATCHED', 'address': tgt['address'],
                                'lat': tgt['lat'], 'lon': tgt['lon']
                            }
                            st.rerun()

                    # Render Simulation Cards if they exist for this target
                    if tgt['id'] in st.session_state.sim_data:
                        s = st.session_state.sim_data[tgt['id']]
                        val_color = "#21c354" if s['del_pct'] < 0 else "#fff"
                        st.markdown(f"""
                        <div style="border: 1px solid #31333F; border-radius: 4px; background-color: #0e1117; font-family: monospace; margin-bottom: 5px;">
                          <div style="display: flex; border-bottom: 1px solid #31333F;">
                              <div style="width: 50%; padding: 5px; border-right: 1px solid #31333F;"><span style="color:#8a8d93; font-size:9px;">BASE DELAY</span><br><span style="color:#fff; font-size:14px;">{s['base_delay']:.1f}m</span></div>
                              <div style="width: 50%; padding: 5px;"><span style="color:#8a8d93; font-size:9px;">SIM DELAY</span><br><span style="color:{val_color}; font-size:14px;">{s['sim_delay']:.1f}m</span></div>
                          </div>
                          <div style="display: flex; border-bottom: 1px solid #31333F;">
                              <div style="width: 50%; padding: 5px; border-right: 1px solid #31333F;"><span style="color:#8a8d93; font-size:9px;">BASE RISK</span><br><span style="color:#fff; font-size:14px;">{s['base_risk']:.1f}</span></div>
                              <div style="width: 50%; padding: 5px;"><span style="color:#8a8d93; font-size:9px;">SIM RISK</span><br><span style="color:{val_color}; font-size:14px;">{s['sim_risk']:.1f}</span></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Select an item from Live Alerts or Predicted Zones to Simulate.")

            render_simulator()

            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

            # ==========================================
            # 2. PREDICTED ZONES (+1H)
            # ==========================================
            st.markdown(f"<b style='color:#ff4b4b; font-size:13px; font-family:monospace;'>🚨 PREDICTED ZONES (+1H)</b>", unsafe_allow_html=True)
            if not top_threats.empty:
                for idx, row in top_threats.head(3).iterrows():
                    action_color = "#ff4b4b" if row['final_risk_score'] >= 75 else "#ffa421" if row['final_risk_score'] >= 50 else "#00a4ff"
                    action_label = "TOW" if row['final_risk_score'] >= 75 else "PATROL" if row['final_risk_score'] >= 50 else "MONITOR"
                    # Instant offline lookup using the resolved_location column!
                    raw_loc = str(row.get('resolved_location', f"Grid {row['grid_lat']:.4f}, {row['grid_lon']:.4f}"))
                    short_addr = ", ".join(raw_loc.split(',')[:2]).strip()

                    cq_text, cq_btn = st.columns([8, 2])
                    with cq_text:
                        st.markdown(f"<b style='color: {action_color}; font-size:11px; font-family:monospace;'>{action_label}</b> <span style='color: #8a8d93; font-size:11px;'>| Risk: {row['final_risk_score']:.1f}</span><br><span style='color: #fff; font-size:10px; font-family: sans-serif;'>{short_addr}</span>", unsafe_allow_html=True)
                    with cq_btn:
                        if st.button("🎯", key=f"pz_{idx}", use_container_width=True):
                            st.session_state.selected_target = {
                                'id': f"pz_{idx}", 'lat': row['grid_lat'], 'lon': row['grid_lon'],
                                'risk': row['final_risk_score'], 'address': short_addr
                            }
                            set_map_view(row['grid_lat'], row['grid_lon'], 16)
                            st.rerun()
                    st.markdown("<hr style='margin: 4px 0; border-color: #31333F;'>", unsafe_allow_html=True)

            # ==========================================
            # 3. DISPATCH QUEUE (ACTIVE LIFECYCLE)
            # ==========================================
            st.markdown(f"<b style='color:#ffa421; font-size:13px; font-family:monospace;'>🚚 ACTIVE DISPATCH QUEUE</b>", unsafe_allow_html=True)
            
            def update_q(qid, stat):
                st.session_state.queue_status[qid]['status'] = stat
            
            if st.session_state.queue_status:
                for qid, qdata in list(st.session_state.queue_status.items()):
                    stat = qdata['status']
                    s_color = "#ff4b4b" if stat == "DISPATCHED" else "#00a4ff" if stat == "EN_ROUTE" else "#21c354"
                    
                    st.markdown(f"""<div style="display:flex; justify-content:space-between; font-size:10px; font-family:monospace;"><span style="color:#fff;">UNIT ACTIVE</span><span style="color:{s_color}; font-weight:bold;">{stat}</span></div><div style="color:#e0e0e0; font-size:10px; font-family:sans-serif; margin-bottom:5px;">{qdata['address']}</div>""", unsafe_allow_html=True)
                    
                    qb1, qb2 = st.columns([7, 3])
                    with qb1:
                        if stat == "DISPATCHED": st.button("MARK EN ROUTE", key=f"enr_{qid}", on_click=update_q, args=(qid, "EN_ROUTE"), use_container_width=True)
                        elif stat == "EN_ROUTE": st.button("MARK RESOLVED", key=f"res_{qid}", on_click=update_q, args=(qid, "RESOLVED"), use_container_width=True)
                        elif stat == "RESOLVED": st.markdown("<div style='color:#21c354; font-size:11px; font-family:monospace;'>✔️ INCIDENT CLEARED</div>", unsafe_allow_html=True)
                    with qb2:
                        if st.button("🎯", key=f"qf_{qid}", use_container_width=True):
                            set_map_view(qdata['lat'], qdata['lon'], 16)
                            st.rerun()
                    st.markdown("<hr style='margin: 4px 0; border-color: #1e212b;'>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#8a8d93; font-size:11px; font-family:monospace;'>No active units deployed.</span><hr style='margin: 10px 0; border-color: #31333F;'>", unsafe_allow_html=True)

            # ==========================================
            # 4. AI ENFORCEMENT ADVISOR
            # ==========================================
            st.markdown("""<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'><b style='color:#00a4ff; font-size:13px; font-family:monospace;'>✨ AI ADVISOR</b><span style='color:#666; font-size:9px; font-family:monospace;'>GEMINI</span></div>""", unsafe_allow_html=True)
            def set_adv(v): st.session_state.advisor_view = v; st.session_state.show_terminal = False
            
            cb1, cb2, cb3 = st.columns(3)
            cb1.button("BRIEF", type="primary" if st.session_state.advisor_view == "brief" else "secondary", on_click=set_adv, args=("brief",), use_container_width=True)
            cb2.button("SHIFT", type="primary" if st.session_state.advisor_view == "shift" else "secondary", on_click=set_adv, args=("shift",), use_container_width=True)
            cb3.button("EXEC", type="primary" if st.session_state.advisor_view == "exec" else "secondary", on_click=set_adv, args=("exec",), use_container_width=True)

            if st.button("⚡ GENERATE", type="primary", use_container_width=True):
                st.session_state.show_terminal = True
                if st.session_state.advisor_view == "brief":
                    prompt = f"Act as AI Enforcement Advisor. Gridlock threat at: {st.session_state.selected_target['address'] if st.session_state.selected_target else 'Unknown'}. Generate a tactical patrol briefing under 100 words. Bullet points."
                elif st.session_state.advisor_view == "shift":
                    prompt = f"Generate Shift Optimization Plan. Peak demand: {df['hour'].value_counts().idxmax()}:00. Bullet points, under 100 words."
                else:
                    prompt = f"Generate Executive Summary. Threat level elevated. {len(live_df)} active violations. Bullet points, under 100 words."
                
                try:
                    import google.generativeai as genai
                    import os
                    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
                    if not api_key: st.error("⚠️ API Key missing.")
                    else:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        t_box = st.empty()
                        full_res = ""
                        for chunk in model.generate_content(prompt, stream=True):
                            full_res += chunk.text
                            t_box.markdown(f"<div class='terminal-box'>{full_res}▌</div>", unsafe_allow_html=True)
                        t_box.markdown(f"<div class='terminal-box'>{full_res}</div>", unsafe_allow_html=True)
                        st.session_state.briefing_text = full_res
                except Exception as e: st.error(f"Failed: {e}")

            elif st.session_state.show_terminal:
                st.markdown(f"<div class='terminal-box'>{st.session_state.briefing_text}</div>", unsafe_allow_html=True)

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
        explainer = shap.TreeExplainer(xgb_model)
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