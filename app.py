# app.py — Gujarat Soil Salinity Intelligence Platform
# GitHub repo root file — run with: streamlit run app.py

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(
    page_title="Gujarat Soil Salinity Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════
DARK = {
    "bg":"#0a1628","card":"#0f2040","border":"#1e3a6e",
    "text_pri":"#e8f0fe","text_sec":"#90afd4","accent":"#4fc3f7",
    "sidebar":"#0f2040","input_bg":"#0f2040",
    "chart_bg":"#0f2040","chart_fig":"#0a1628",
    "grid":"#1e3a6e","tick":"#90afd4",
    "hdr_grad":"linear-gradient(135deg,#0f2040,#1a3a6e)",
    "mode_label":"🌙 Dark","toggle_label":"Switch to ☀️ Light Mode",
}
LIGHT = {
    "bg":"#f4f7fb","card":"#ffffff","border":"#d0dce8",
    "text_pri":"#1a2a3a","text_sec":"#5a7a9a","accent":"#0288d1",
    "sidebar":"#e8f0f8","input_bg":"#ffffff",
    "chart_bg":"#ffffff","chart_fig":"#f4f7fb",
    "grid":"#d0dce8","tick":"#5a7a9a",
    "hdr_grad":"linear-gradient(135deg,#e8f0f8,#c8dff0)",
    "mode_label":"☀️ Light","toggle_label":"Switch to 🌙 Dark Mode",
}
if "theme" not in st.session_state: st.session_state.theme = "dark"
T = DARK if st.session_state.theme == "dark" else LIGHT

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CLR  = {"Critical":"#e53935","High":"#fb8c00","Moderate":"#c6d400","Low":"#43a047"}
ZONE_ICON = {"coastal":"🌊","canal":"💧","inland":"🔩","hilly":"🌲"}
ZONE_LIST = [("🌊","coastal"),("💧","canal"),("🔩","inland"),("🌲","hilly")]
DEFAULT_CENTER = [22.5,71.5]
DEFAULT_ZOOM   = 7

COORDS = {
    "Kachchh":(23.73,69.86),"Jamnagar":(22.47,70.06),"Bhavnagar":(21.76,72.15),
    "Surat":(21.17,72.83),"Bharuch":(21.70,72.98),"Porbandar":(21.64,69.61),
    "DevbhumiDwarka":(22.23,68.97),"GirSomnath":(20.90,70.37),"Morbi":(22.82,70.83),
    "Anand":(22.55,72.95),"Kheda":(22.75,72.68),"Surendranagar":(22.72,71.65),
    "Narmada":(21.87,73.49),"Navsari":(20.95,72.92),"Mahesana":(23.59,72.37),
    "Patan":(23.84,72.11),"BanasKantha":(24.17,72.43),"SabarKantha":(23.35,73.02),
    "Aravalli":(23.70,73.00),"Ahmadabad":(23.02,72.57),"Gandhinagar":(23.22,72.65),
    "Vadodara":(22.30,73.18),"Rajkot":(22.30,70.78),"Amreli":(21.60,71.22),
    "Junagadh":(21.52,70.45),"Botad":(22.17,71.67),"Dahod":(22.83,74.25),
    "Mahisagar":(23.08,73.55),"PanchMahals":(22.78,73.52),"ChhotaUdaipur":(22.30,74.01),
    "Tapi":(21.14,73.41),"Valsad":(20.59,72.93),"TheDangs":(20.75,73.69)
}

ZONE_MASTER = {
    "Kachchh":"coastal","Jamnagar":"coastal","DevbhumiDwarka":"coastal",
    "Junagadh":"coastal","GirSomnath":"coastal","Bhavnagar":"coastal",
    "Amreli":"coastal","Porbandar":"coastal","Morbi":"coastal","Rajkot":"coastal",
    "Anand":"canal","Kheda":"canal","Surendranagar":"canal",
    "Bharuch":"canal","Narmada":"canal","Botad":"canal",
    "Dahod":"hilly","Mahisagar":"hilly","PanchMahals":"hilly",
    "ChhotaUdaipur":"hilly","Tapi":"hilly","Valsad":"hilly","TheDangs":"hilly",
    "Ahmadabad":"inland","Aravalli":"inland","BanasKantha":"inland",
    "Gandhinagar":"inland","Mahesana":"inland","Patan":"inland",
    "SabarKantha":"inland","Vadodara":"inland","Navsari":"inland","Surat":"inland",
}

CAUSES = {
    "coastal":"🌊 Seawater intrusion · Arabian Sea coastal erosion",
    "canal":"💧 Narmada canal waterlogging · secondary salinization",
    "inland":"🔩 Borewell saline upconing · groundwater over-extraction",
    "hilly":"🌲 Natural low-salinity zone · good terrain drainage"
}

SMART_POLICY = {
    ("coastal","Critical"):"🚨 URGENT — Emergency coastal bund construction + mangrove belt restoration. Ban new borewells within 5 km. Immediate gypsum application.",
    ("coastal","High"):"🔴 HIGH — Install subsurface drainage + coastal windbreaks. Monitor groundwater EC monthly.",
    ("coastal","Moderate"):"⚠️ MONITOR — Seasonal ECe testing. Limit irrigation near tidal zones. Salt-tolerant crops.",
    ("coastal","Low"):"✅ STABLE — Annual coastal soil health check. Maintain bund structures.",
    ("canal","Critical"):"🚨 URGENT — Stop canal supply to affected fields. Apply gypsum + leaching. Install subsurface tile drains.",
    ("canal","High"):"🔴 HIGH — Reduce canal allocation by 30%. Field-level drainage. Switch to drip/sprinkler.",
    ("canal","Moderate"):"⚠️ MONITOR — Improve irrigation scheduling. Quarterly ECe monitoring.",
    ("canal","Low"):"✅ STABLE — Maintain current practices. Annual soil health card.",
    ("inland","Critical"):"🚨 URGENT — Ban new borewell drilling. Bio-drainage plantations. Gypsum + deep ploughing + leaching.",
    ("inland","High"):"🔴 HIGH — Restrict groundwater extraction depth. Switch to micro-irrigation.",
    ("inland","Moderate"):"⚠️ MONITOR — Monitor groundwater depth/EC quarterly. Reduce irrigation frequency.",
    ("inland","Low"):"✅ STABLE — Annual groundwater quality check. Rainwater harvesting.",
    ("hilly","Critical"):"⚠️ INVESTIGATE — Unusual for hilly zone. Likely industrial effluent. Investigate immediately.",
    ("hilly","High"):"⚠️ INVESTIGATE — Check upstream land-use change. Soil sampling recommended.",
    ("hilly","Moderate"):"✅ MONITOR — Likely temporary. Natural drainage should self-correct.",
    ("hilly","Low"):"✅ STABLE — Natural drainage active. Maintain forest cover.",
}

DEFAULT_METRICS = pd.DataFrame({
    "model":["Linear Regression","Random Forest","XGBoost","TFT Transformer"],
    "mae":[7.81,7.26,6.65,4.91],"rmse":[9.01,8.45,8.53,7.25],
    "r2":[0.745,0.776,0.771,0.831],"mape":[np.nan,np.nan,np.nan,9.3]
})

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def sc(cls): return CLR.get(cls,T["accent"])

def compute_sspi_class(v):
    if v>=75: return "Critical"
    elif v>=50: return "High"
    elif v>=25: return "Moderate"
    return "Low"

def ax_style(ax,fig):
    fig.patch.set_facecolor(T["chart_fig"]); ax.set_facecolor(T["chart_bg"])
    ax.tick_params(colors=T["tick"],labelsize=8)
    ax.xaxis.label.set_color(T["text_sec"]); ax.yaxis.label.set_color(T["text_sec"])
    ax.title.set_color(T["text_pri"])
    for sp in ax.spines.values(): sp.set_edgecolor(T["border"])
    ax.grid(alpha=0.18,color=T["grid"],linewidth=0.5)

def zb(z): return f"<span class='zb zb-{z}'>{ZONE_ICON.get(z,'')} {z.upper()}</span>"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db = os.path.join(BASE_DIR, "data", "Processed", "salinity_db.sqlite")
    if not os.path.exists(db):
        st.error(f"❌ Database not found: {db}\nRun scripts 01–11 first.")
        st.stop()
    conn = sqlite3.connect(db)
    h = pd.read_sql("SELECT * FROM sspi_history", conn)
    f = pd.read_sql("SELECT * FROM sspi_forecast", conn)
    try:
        m = pd.read_sql("SELECT * FROM model_metrics", conn)
        if len(m)==0 or "r2" not in m.columns: m = DEFAULT_METRICS.copy()
    except Exception: m = DEFAULT_METRICS.copy()
    conn.close()
    h["sspi_class"] = h["sspi"].apply(compute_sspi_class)
    h["zone_type"]  = h["district"].map(ZONE_MASTER).fillna("inland")
    if "district" in f.columns:
        f["zone_type"] = f["district"].map(ZONE_MASTER).fillna("inland")
    return h, f, m

history, forecast, metrics = load_data()

latest_all = history.sort_values("year").groupby("district").last().reset_index()
latest_all["zone_type"] = latest_all["district"].map(ZONE_MASTER).fillna("inland")

year_2025   = int(history["year"].max())
current_sspi = (
    history[history["year"]==year_2025]
    [["district","sspi","ndvi","rainfall_annual","temp_rabi"]].copy()
)
current_sspi["sspi_class"] = current_sspi["sspi"].apply(compute_sspi_class)
current_sspi["zone_type"]  = current_sspi["district"].map(ZONE_MASTER).fillna("inland")

all_districts = sorted(history["district"].unique())
zone_map = {d: ZONE_MASTER.get(d,"inland") for d in all_districts}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "app_initialized":True,"sel":None,
    "map_center":DEFAULT_CENTER,"map_zoom":DEFAULT_ZOOM,
    "active_zone":None,"t5":0,"map_layer":"Current 2025",
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

sel_for_data = st.session_state.sel if st.session_state.sel else all_districts[0]

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
_clr_critical = CLR["Critical"]
_clr_high     = CLR["High"]
_clr_moderate = CLR["Moderate"]
_clr_low      = CLR["Low"]
_live_bg      = '#1b5e20' if st.session_state.theme == 'dark' else '#e8f5e9'
_live_color   = '#a5d6a7' if st.session_state.theme == 'dark' else '#2e7d32'

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;background:{T["bg"]};color:{T["text_pri"]};}}
.block-container{{padding:1rem 1.8rem 2rem;max-width:1600px;}}
section[data-testid="stSidebar"]{{background:{T["sidebar"]};border-right:1px solid {T["border"]};}}
div[data-baseweb="tab-list"]{{background:{T["card"]} !important;border-radius:10px;border:1px solid {T["border"]};padding:3px;gap:2px;}}
div[data-baseweb="tab"]{{background:transparent !important;border-radius:8px !important;color:{T["text_sec"]} !important;font-size:.78rem !important;font-weight:600 !important;padding:.4rem .9rem !important;transition:all .2s;}}
div[data-baseweb="tab"][aria-selected="true"]{{background:{T["accent"]} !important;color:#fff !important;}}
div[data-baseweb="select"]>div{{background:{T["input_bg"]} !important;border:1.5px solid {T["border"]} !important;border-radius:10px !important;color:{T["text_pri"]} !important;}}
div[data-baseweb="select"] span{{color:{T["text_pri"]} !important;}}
div[data-baseweb="popover"] ul{{background:{T["card"]} !important;border:1px solid {T["border"]} !important;}}
div[data-baseweb="popover"] li{{color:{T["text_pri"]} !important;}}
div[data-baseweb="popover"] li:hover{{background:{T["border"]} !important;}}
.stButton>button{{background:{T["card"]} !important;border:1px solid {T["border"]} !important;color:{T["text_pri"]} !important;border-radius:8px !important;font-size:.72rem !important;font-family:'DM Sans',sans-serif !important;transition:all .2s !important;}}
.stButton>button:hover{{border-color:{T["accent"]} !important;color:{T["accent"]} !important;}}
.theme-btn>button{{background:{T["accent"]} !important;color:#fff !important;border:none !important;font-weight:700 !important;width:100% !important;}}
.reset-btn>button{{background:transparent !important;border:1.5px dashed {T["border"]} !important;color:{T["text_sec"]} !important;font-size:.68rem !important;}}
.reset-btn>button:hover{{border-color:{T["accent"]} !important;color:{T["accent"]} !important;}}
.zone-active>button{{background:{T["accent"]} !important;color:#fff !important;border-color:{T["accent"]} !important;font-weight:800 !important;}}
.layer-active>button{{background:#43a047 !important;color:#fff !important;border-color:#43a047 !important;font-weight:800 !important;}}
.stDataFrame{{border-radius:10px;overflow:hidden;}}
.kpi{{background:{T["card"]};border:1px solid {T["border"]};border-radius:12px;padding:.85rem 1rem;position:relative;overflow:hidden;height:100%;transition:transform .15s,box-shadow .15s;}}
.kpi:hover{{transform:translateY(-2px);box-shadow:0 6px 24px rgba(0,0,0,.15);}}
.kpi::before{{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:var(--kc,{T["accent"]});border-radius:3px 0 0 3px;}}
.kpi-icon{{font-size:1rem;margin-bottom:.2rem;display:block;}}
.kpi-lbl{{font-size:.58rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.09em;margin-bottom:.12rem;}}
.kpi-val{{font-size:1.5rem;font-weight:700;font-family:'Space Mono',monospace;color:var(--kc,{T["text_pri"]});line-height:1.1;}}
.kpi-sub{{font-size:.62rem;color:{T["text_sec"]};margin-top:.12rem;}}
.sh{{font-size:.68rem;font-weight:700;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.1em;border-left:3px solid {T["accent"]};padding-left:.5rem;margin:1rem 0 .6rem;}}
.ic{{background:{T["card"]};border:1px solid {T["border"]};border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.55rem;transition:border-color .2s;}}
.ic.Critical{{border-color:{_clr_critical};}} .ic.High{{border-color:{_clr_high};}}
.ic.Moderate{{border-color:{_clr_moderate};}} .ic.Low{{border-color:{_clr_low};}}
.zb{{display:inline-block;font-size:.58rem;font-weight:700;padding:.14rem .5rem;border-radius:20px;letter-spacing:.05em;text-transform:uppercase;}}
.zb-coastal{{background:rgba(2,136,209,.15);color:#29b6f6;border:1px solid #0288d1;}}
.zb-canal{{background:rgba(156,39,176,.15);color:#ce93d8;border:1px solid #9c27b0;}}
.zb-inland{{background:rgba(245,124,0,.15);color:#ffb74d;border:1px solid #f57c00;}}
.zb-hilly{{background:rgba(46,125,50,.15);color:#81c784;border:1px solid #2e7d32;}}
.mc{{background:{T["card"]};border:1px solid {T["border"]};border-radius:12px;padding:1rem;text-align:center;position:relative;overflow:hidden;transition:transform .15s,box-shadow .15s;}}
.mc:hover{{transform:translateY(-3px);box-shadow:0 8px 28px rgba(0,0,0,.18);}}
.mc.best{{border-color:{T["accent"]};box-shadow:0 0 18px rgba(79,195,247,.18);}}
.mc-icon{{font-size:1.7rem;margin-bottom:.3rem;}}
.mc-name{{font-size:.78rem;font-weight:700;color:{T["text_pri"]};margin-bottom:.35rem;}}
.mc-r2{{font-size:2.4rem;font-weight:800;font-family:'Space Mono',monospace;line-height:1.1;}}
.mc-meta{{font-size:.64rem;color:{T["text_sec"]};margin-top:.35rem;line-height:1.6;}}
.best-tag{{position:absolute;top:8px;right:8px;background:{T["accent"]};color:#fff;font-size:.54rem;font-weight:800;padding:.15rem .5rem;border-radius:10px;}}
.gauge{{background:{T["border"]};border-radius:5px;height:6px;margin:.3rem 0;overflow:hidden;}}
.gfill{{height:100%;border-radius:5px;transition:width .6s ease;}}
.fp{{display:flex;justify-content:space-between;align-items:center;padding:.42rem .8rem;margin-bottom:.32rem;background:{T["bg"]};border:1px solid {T["border"]};border-radius:8px;transition:border-color .2s;}}
.fp:hover{{border-color:{T["accent"]};}}
.big-score{{font-size:3rem;font-weight:800;font-family:'Space Mono',monospace;line-height:1;}}
.app-hdr{{background:{T["hdr_grad"]};border:1px solid {T["border"]};border-radius:14px;padding:1rem 1.6rem;margin-bottom:.8rem;display:flex;align-items:center;justify-content:space-between;}}
.app-title{{font-size:1.35rem;font-weight:700;color:{T["text_pri"]};margin:0;}}
.app-sub{{font-size:.72rem;color:{T["text_sec"]};margin:.2rem 0 0;line-height:1.5;}}
.live-dot{{background:{_live_bg};border:1px solid #43a047;color:{_live_color};font-size:.63rem;font-weight:700;padding:.2rem .7rem;border-radius:20px;animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.6;}}}}
.acc-banner{{background:{T["hdr_grad"]};border:2px solid {T["accent"]};border-radius:14px;padding:1.1rem 1.4rem;text-align:center;box-shadow:0 0 22px rgba(79,195,247,.12);}}
.acc-ring{{font-size:3.4rem;font-weight:900;font-family:'Space Mono',monospace;background:linear-gradient(135deg,{T["accent"]},#43a047);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;}}
.footer{{text-align:center;font-size:.63rem;color:{T["text_sec"]};border-top:1px solid {T["border"]};padding-top:.8rem;margin-top:1.2rem;}}
.layer-pill{{display:inline-block;font-size:.6rem;font-weight:700;padding:.18rem .7rem;border-radius:12px;margin-right:.3rem;vertical-align:middle;}}
.layer-current{{background:rgba(67,160,71,.2);color:#81c784;border:1px solid #43a047;}}
.layer-forecast{{background:rgba(229,57,53,.2);color:#ef9a9a;border:1px solid #e53935;}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='font-size:.62rem;color:{T['text_sec']};margin-bottom:.3rem;font-weight:600'>{T['mode_label']} MODE ACTIVE</div>",unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='theme-btn'>",unsafe_allow_html=True)
        if st.button(T["toggle_label"],key="theme_toggle",use_container_width=True):
            st.session_state.theme="light" if st.session_state.theme=="dark" else "dark"
            st.session_state.active_zone=None; st.rerun()
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("<div style='margin:.3rem 0'></div>",unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='reset-btn'>",unsafe_allow_html=True)
        if st.button("🔄 Reset to Default View",key="reset_btn",use_container_width=True):
            for _k in list(defaults.keys()): st.session_state.pop(_k,None)
            st.rerun()
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{T['border']};margin:.6rem 0'>",unsafe_allow_html=True)

    # ── MAP LAYER TOGGLE ──────────────────────────────────────────────────────
    st.markdown(f"<div style='font-size:.68rem;color:{T['accent']};text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:.4rem'>🗺️ Map Layer</div>",unsafe_allow_html=True)
    ml_c1,ml_c2 = st.columns(2)
    with ml_c1:
        is_cur = st.session_state.map_layer=="Current 2025"
        if is_cur: st.markdown("<div class='layer-active'>",unsafe_allow_html=True)
        if st.button(f"📍 Current {year_2025}",key="layer_cur",use_container_width=True):
            st.session_state.map_layer="Current 2025"; st.rerun()
        if is_cur: st.markdown("</div>",unsafe_allow_html=True)
    with ml_c2:
        is_fore = st.session_state.map_layer=="Forecast 2030"
        if is_fore: st.markdown("<div class='layer-active'>",unsafe_allow_html=True)
        if st.button("🔮 Forecast 2030",key="layer_fore",use_container_width=True):
            st.session_state.map_layer="Forecast 2030"; st.rerun()
        if is_fore: st.markdown("</div>",unsafe_allow_html=True)
    layer_desc = (f"Showing <b>actual SSPI {year_2025}</b>" if st.session_state.map_layer=="Current 2025"
                  else "Showing <b>predicted SSPI 2030</b>")
    st.markdown(f"<div style='font-size:.62rem;color:{T['text_sec']};margin:.25rem 0 .5rem'>{layer_desc}</div>",unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{T['border']};margin:.4rem 0'>",unsafe_allow_html=True)

    # ── ZONE FILTER ───────────────────────────────────────────────────────────
    st.markdown(f"<div style='font-size:.68rem;color:{T['accent']};text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:.4rem'>🗂️ Filter by Zone</div>",unsafe_allow_html=True)
    qz_cols = st.columns(4)
    for col,(icon,zone) in zip(qz_cols,ZONE_LIST):
        is_active = (st.session_state.active_zone==zone)
        with col:
            if is_active: st.markdown("<div class='zone-active'>",unsafe_allow_html=True)
            clicked = st.button(icon,key=f"qz_{zone}",help=f"{zone.title()} zone",use_container_width=True)
            if is_active: st.markdown("</div>",unsafe_allow_html=True)
            if clicked:
                zone_dists=[d for d in all_districts if zone_map.get(d)==zone]
                if not zone_dists: st.toast(f"No districts for {zone.title()}",icon="⚠️")
                elif st.session_state.active_zone==zone: st.session_state.active_zone=None; st.rerun()
                else:
                    st.session_state.active_zone=zone
                    st.session_state.sel=zone_dists[0]
                    st.session_state.map_center=list(COORDS.get(zone_dists[0],[22.5,71.5]))
                    st.session_state.map_zoom=8; st.rerun()

    if st.session_state.active_zone:
        az=st.session_state.active_zone
        zone_dists=[d for d in all_districts if zone_map.get(d)==az]
        icon_az=dict(ZONE_LIST).get(az,""); n=len(zone_dists)
        st.markdown(f"""<div style='background:{T["card"]};border:1.5px solid {T["accent"]};border-radius:8px;
          padding:.5rem .8rem;margin:.4rem 0;display:flex;justify-content:space-between;align-items:center'>
          <span style='font-size:.75rem;font-weight:700;color:{T["accent"]}'>{icon_az} {az.upper()}</span>
          <span style='font-size:.62rem;color:{T["text_sec"]}'>{n} districts</span></div>""",unsafe_allow_html=True)
        cur_sel=st.session_state.sel if st.session_state.sel in zone_dists else zone_dists[0]
        chosen_zone_dist=st.selectbox("Pick district",options=zone_dists,index=zone_dists.index(cur_sel),
                                      label_visibility="collapsed",key="zone_dist_select")
        if chosen_zone_dist!=st.session_state.sel:
            st.session_state.sel=chosen_zone_dist
            st.session_state.map_center=list(COORDS.get(chosen_zone_dist,[22.5,71.5]))
            st.session_state.map_zoom=10; st.rerun()
        if st.button("✕ Clear zone filter",key="clear_zone",use_container_width=True):
            st.session_state.active_zone=None; st.rerun()
    else:
        st.markdown(f"<div style='font-size:.68rem;color:{T['accent']};text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin:.5rem 0 .4rem'>📍 Select District</div>",unsafe_allow_html=True)
        district_options=["— Select a district —"]+all_districts
        cur_idx=all_districts.index(st.session_state.sel)+1 if (st.session_state.sel and st.session_state.sel in all_districts) else 0
        chosen=st.selectbox("",options=district_options,index=cur_idx,label_visibility="collapsed",key="dist_selector")
        if chosen!="— Select a district —" and chosen!=st.session_state.sel:
            st.session_state.sel=chosen
            st.session_state.map_center=list(COORDS.get(chosen,[22.5,71.5]))
            st.session_state.map_zoom=10; st.rerun()
        elif chosen=="— Select a district —" and st.session_state.sel is not None:
            st.session_state.sel=None; st.session_state.map_center=DEFAULT_CENTER; st.session_state.map_zoom=DEFAULT_ZOOM; st.rerun()

    st.markdown(f"<hr style='border-color:{T['border']};margin:.6rem 0'>",unsafe_allow_html=True)

    # ── SIDEBAR ACCURACY ──────────────────────────────────────────────────────
    best_r2=metrics["r2"].max(); avg_r2=metrics["r2"].mean()
    best_name_sb=metrics.loc[metrics["r2"].idxmax(),"model"]
    overall_acc=round(best_r2*100,1); avg_acc_sb=round(avg_r2*100,1)
    st.markdown(f"""<div style='background:{T["hdr_grad"]};border:1px solid {T["accent"]};border-radius:10px;padding:.8rem;text-align:center;margin-bottom:.4rem'>
      <div style='font-size:.58rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.2rem'>Best Model Accuracy</div>
      <div style='font-size:2.6rem;font-weight:900;font-family:Space Mono,monospace;color:{T["accent"]};line-height:1'>{overall_acc}%</div>
      <div style='font-size:.62rem;color:#43a047;font-weight:700;margin-top:.2rem'>{best_name_sb}</div>
    </div>""",unsafe_allow_html=True)
    st.markdown(f"""<div style='background:{T["card"]};border:1px solid {T["border"]};border-radius:8px;padding:.6rem .8rem;margin-bottom:.4rem'>
      <div style='font-size:.58rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.2rem'>Avg. System Accuracy</div>
      <div style='font-size:1.8rem;font-weight:800;font-family:Space Mono,monospace;color:{T["text_pri"]}'>{avg_acc_sb}%</div>
      <div class='gauge'><div class='gfill' style='width:{avg_acc_sb}%;background:{T["accent"]}'></div></div>
      <div style='font-size:.6rem;color:{T["text_sec"]};margin-top:.2rem'>Across 4 models · 33 districts</div>
    </div>""",unsafe_allow_html=True)
    icons_sb={"Linear Regression":("📐","#5c6bc0"),"Random Forest":("🌲","#43a047"),"XGBoost":("⚡","#fb8c00"),"TFT Transformer":("🤖",T["accent"])}
    for _,row in metrics.iterrows():
        icon,clr_m=icons_sb.get(row["model"],("📊",T["accent"])); acc=round(row["r2"]*100,1)
        st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;padding:.3rem .6rem;background:{T["bg"]};border:1px solid {T["border"]};border-radius:7px;margin-bottom:.25rem'>
          <span style='font-size:.7rem;color:{T["text_pri"]}'>{icon} {row["model"]}</span>
          <span style='font-size:.75rem;font-weight:700;font-family:Space Mono,monospace;color:{clr_m}'>{acc}%</span>
        </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SELECTED DISTRICT DATA
# ══════════════════════════════════════════════════════════════════════════════
sel=sel_for_data
dh=history[history["district"]==sel].sort_values("year").copy()
df=forecast[forecast["district"]==sel].sort_values("year").copy()
zone_type=ZONE_MASTER.get(sel,"inland")
latest_yr=int(dh["year"].max()) if len(dh)>0 else year_2025
latest_sspi=dh[dh["year"]==latest_yr]["sspi"].values[0] if len(dh)>0 else 0
sal_class=compute_sspi_class(latest_sspi)
ndvi_avg=dh["ndvi"].mean() if len(dh)>0 else 0
rain_avg=dh["rainfall_annual"].mean() if len(dh)>0 else 0
temp_avg=dh["temp_rabi"].mean() if len(dh)>0 else 0
s30v=df[df["year"]==2030]["predicted_sspi"].values
sspi_2030=s30v[0] if len(s30v)>0 else None
trend_dir=df["trend"].iloc[0] if len(df)>0 else "Stable"
clr_sel=sc(sal_class)
sspi_delta=(dh["sspi"].iloc[-1]-dh["sspi"].iloc[0]) if len(dh)>1 else 0
best_acc_pct=round(metrics["r2"].max()*100,1)
recommended_action=SMART_POLICY.get((zone_type,sal_class),"⚠️ No specific recommendation available.")

zone_badge=""
if st.session_state.active_zone:
    az=st.session_state.active_zone; n=len([d for d in all_districts if zone_map.get(d)==az])
    zone_badge=f"&nbsp;<span style='background:{T['accent']};color:#fff;font-size:.6rem;font-weight:700;padding:.15rem .6rem;border-radius:10px;vertical-align:middle'>🔍 {az.upper()} ZONE · {n} districts</span>"

display_district=(f"📍 {st.session_state.sel} &nbsp; {zb(zone_type)}" if st.session_state.sel
                  else "📍 No district selected — click map or use search")

# ── HEADER ─────────────────────────────────────────────────────────────────────
active_layer_pill=(
    f"<span class='layer-pill layer-current'>📍 Current {year_2025}</span>"
    if st.session_state.map_layer=="Current 2025"
    else "<span class='layer-pill layer-forecast'>🔮 Forecast 2030</span>"
)
st.markdown(f"""<div class='app-hdr'>
  <div><div class='app-title'>🌾 Gujarat Soil Salinity Intelligence Platform {active_layer_pill}{zone_badge}</div>
  <div class='app-sub'>Spatio-Temporal Prediction 2015–2030 · ML + TFT · 33 Districts · MODIS GEE · NASA POWER · CSSRI Bharuch</div></div>
  <div style='display:flex;gap:.6rem;align-items:center'>
    <div style='text-align:right;font-size:.68rem;color:{T["text_sec"]}'>Overall Accuracy<br>
    <span style='font-size:1.3rem;font-weight:800;font-family:Space Mono,monospace;color:{T["accent"]}'>{round(metrics["r2"].mean()*100,1)}%</span></div>
    <span class='live-dot'>● LIVE</span>
  </div>
</div>""",unsafe_allow_html=True)
st.markdown(f"<div style='font-size:.9rem;font-weight:600;margin:.1rem 0 .5rem'>{display_district}</div>",unsafe_allow_html=True)

# ── KPI ROW ────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6,k7=st.columns(7)
kpis=[
    (k1,"🎯","SSPI Score",f"{latest_sspi:.1f}" if st.session_state.sel else "—",sal_class,clr_sel),
    (k2,"🏷️","Risk Class",sal_class if st.session_state.sel else "—",f"as of {latest_yr}",clr_sel),
    (k3,"🔮","Forecast 2030",f"{sspi_2030:.1f}" if sspi_2030 and st.session_state.sel else "—",
        f"📈 {trend_dir}",CLR["Critical"] if (sspi_2030 or 0)>=75 else CLR["High"] if (sspi_2030 or 0)>=50 else CLR["Low"]),
    (k4,"🌿","Avg NDVI",f"{ndvi_avg:.3f}" if st.session_state.sel else "—","Vegetation index",CLR["Low"]),
    (k5,"🌧️","Avg Rainfall",f"{rain_avg:.0f} mm" if st.session_state.sel else "—","Annual mean",T["accent"]),
    (k6,"📊","10yr Change",
        f"{'▲' if sspi_delta>0 else '▼'} {abs(sspi_delta):.1f}" if st.session_state.sel else "—",
        "SSPI 2015→2025",CLR["Critical"] if sspi_delta>5 else CLR["Low"] if sspi_delta<0 else T["text_sec"]),
    (k7,"🤖","Best Accuracy",f"{best_acc_pct}%","TFT Transformer",T["accent"]),
]
for col,icon,lbl,val,sub,clr in kpis:
    with col:
        st.markdown(f"<div class='kpi' style='--kc:{clr}'><span class='kpi-icon'>{icon}</span>"
                    f"<div class='kpi-lbl'>{lbl}</div><div class='kpi-val' style='color:{clr}'>{val}</div>"
                    f"<div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)
st.markdown("<div style='margin:.5rem 0'></div>",unsafe_allow_html=True)

t5_b1,t5_b2,t5_b3,t5_spacer=st.columns([1,1,1,3])
with t5_b1:
    if st.button("📊 Historical SSPI",key="btn_hist",use_container_width=True): st.session_state["t5"]=0
with t5_b2:
    if st.button("🔮 Forecast 2026–2030",key="btn_fore",use_container_width=True): st.session_state["t5"]=1
with t5_b3:
    if st.button("🗺️ All Districts Latest",key="btn_all",use_container_width=True): st.session_state["t5"]=2

tab1,tab2,tab3,tab4,tab5=st.tabs(["🗺️  Map & Info","📈  Trends","🔮  Forecast","🤖  Model Accuracy","📋  Data"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP & INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    cm,ci=st.columns([3,2])
    with cm:
        # ── Which layer? ──────────────────────────────────────────────────────
        if st.session_state.map_layer=="Current 2025":
            st.markdown(f"<div class='sh'>🗺️ Interactive Map — <span style='color:#43a047'>Current Salinity {year_2025}</span> (actual SSPI)</div>",unsafe_allow_html=True)
            map_source=current_sspi.copy()
        else:
            _fc_clr = CLR["Critical"]
            st.markdown(f"<div class='sh'>🗺️ Interactive Map — <span style='color:{_fc_clr}'>Forecast 2030</span> (predicted SSPI)</div>",unsafe_allow_html=True)
            fore_2030=forecast[forecast["year"]==2030][["district","predicted_sspi"]].copy()
            fore_2030["sspi_class"]=fore_2030["predicted_sspi"].apply(compute_sspi_class)
            fore_2030["zone_type"]=fore_2030["district"].map(ZONE_MASTER).fillna("inland")
            fore_2030["ndvi"]=np.nan; fore_2030["rainfall_annual"]=np.nan
            map_source=fore_2030.rename(columns={"predicted_sspi":"sspi"})

        # Zone filter
        if st.session_state.active_zone:
            map_rows=map_source[map_source["district"].map(ZONE_MASTER).fillna("inland")==st.session_state.active_zone]
        else:
            map_rows=map_source

        # ── BUILD FOLIUM MAP ──────────────────────────────────────────────────
        tiles="CartoDB dark_matter" if st.session_state.theme=="dark" else "CartoDB positron"
        m=folium.Map(location=st.session_state.map_center,zoom_start=st.session_state.map_zoom,tiles=tiles)

        for _,row in map_rows.iterrows():
            dn=row["district"]
            if dn not in COORDS: continue
            lat,lon=COORDS[dn]; isel=(dn==st.session_state.sel)
            sspi_v=float(row["sspi"]); dot_class=compute_sspi_class(sspi_v); clr=sc(dot_class)
            zt=ZONE_MASTER.get(dn,"inland")

            # Outer glow ring for Critical/High districts
            if sspi_v>=50:
                folium.CircleMarker(location=[lat,lon],radius=26 if isel else 18,
                    color=clr,weight=1,fill=True,fill_color=clr,fill_opacity=0.12).add_to(m)

            # NDVI/rain extras for tooltip (only available in current layer)
            ndvi_str=(f"<span style='color:#90afd4;font-size:10px'>NDVI: </span><b>{row['ndvi']:.3f}</b> &nbsp;"
                      if pd.notna(row.get("ndvi")) else "")
            rain_str=(f"<span style='color:#90afd4;font-size:10px'>Rain: </span><b>{row['rainfall_annual']:.0f} mm</b>"
                      if pd.notna(row.get("rainfall_annual")) else "")
            layer_tag=f"Current {year_2025}" if st.session_state.map_layer=="Current 2025" else "Forecast 2030"

            folium.CircleMarker(
                location=[lat,lon],radius=22 if isel else 14,
                color="white" if isel else clr,weight=3 if isel else 1.5,
                fill=True,fill_color=clr,fill_opacity=1.0 if isel else 0.88,
                tooltip=folium.Tooltip(
                    f"<div style='font-family:sans-serif;font-size:12px;background:#0f2040;color:#e8f0fe;"
                    f"padding:8px 12px;border-radius:8px;border:2px solid {clr};min-width:190px;box-shadow:0 4px 12px rgba(0,0,0,.4)'>"
                    f"<b style='font-size:14px;color:{clr}'>{dn}</b><br>"
                    f"<span style='color:#90afd4;font-size:10px'>SSPI ({layer_tag}): </span>"
                    f"<b style='color:{clr};font-size:13px'>{sspi_v:.1f}</b> &nbsp;"
                    f"<b style='color:{clr}'>{dot_class}</b><br>"
                    f"<span style='color:#90afd4;font-size:10px'>Zone: </span><b style='color:#e8f0fe'>{zt}</b><br>"
                    f"{ndvi_str}{rain_str}<br>"
                    f"<i style='font-size:9px;color:#90afd4'>Click to select district</i></div>",sticky=False),
                popup=folium.Popup(dn,max_width=1)
            ).add_to(m)

            if isel:
                folium.Marker(location=[lat,lon],
                    icon=folium.DivIcon(
                        html=(f"<div style='background:{clr};color:#fff;font-size:9px;font-weight:700;"
                              f"white-space:nowrap;padding:2px 7px;border-radius:10px;"
                              f"margin-top:-44px;margin-left:22px;box-shadow:0 2px 6px rgba(0,0,0,.4)'>"
                              f"{dn} · {sspi_v:.0f}</div>"),icon_size=(200,22))
                ).add_to(m)

        # ── MAP LEGEND ────────────────────────────────────────────────────────
        layer_label=f"Current SSPI {year_2025}" if st.session_state.map_layer=="Current 2025" else "Predicted SSPI 2030"
        _lc = CLR["Critical"]; _lh = CLR["High"]; _lm = CLR["Moderate"]; _ll = CLR["Low"]
        legend=(f"<div style='position:fixed;bottom:20px;left:20px;background:#0f2040;padding:10px 14px;"
                f"border-radius:8px;border:1px solid #1e3a6e;color:#e8f0fe;font-family:sans-serif;font-size:11px;z-index:9999'>"
                f"<b>{layer_label}</b><br>"
                f"<span style='color:{_lc}'>●</span> Critical ≥75<br>"
                f"<span style='color:{_lh}'>●</span> High 50–74<br>"
                f"<span style='color:{_lm}'>●</span> Moderate 25–49<br>"
                f"<span style='color:{_ll}'>●</span> Low &lt;25<br>"
                f"<hr style='border-color:#1e3a6e;margin:4px 0'>"
                f"<span style='color:#29b6f6'>●</span> Coastal &nbsp;<span style='color:#ce93d8'>●</span> Canal<br>"
                f"<span style='color:#ffb74d'>●</span> Inland &nbsp;<span style='color:#81c784'>●</span> Hilly</div>")
        m.get_root().html.add_child(folium.Element(legend))

        map_data=st_folium(m,width="100%",height=470,
                           returned_objects=["last_object_clicked_popup","last_object_clicked"])

        # ── CLICK HANDLING ────────────────────────────────────────────────────
        clicked_name=None
        if map_data:
            popup_val=map_data.get("last_object_clicked_popup")
            if popup_val and isinstance(popup_val,str):
                popup_val=popup_val.strip()
                if popup_val in all_districts and popup_val!=st.session_state.sel:
                    clicked_name=popup_val
            if not clicked_name:
                loc=map_data.get("last_object_clicked")
                if loc and isinstance(loc,dict):
                    clat=loc.get("lat"); clng=loc.get("lng")
                    if clat is not None and clng is not None:
                        best_dist,best_name=float("inf"),None
                        for dn,(dlat,dlng) in COORDS.items():
                            d=(dlat-clat)**2+(dlng-clng)**2
                            if d<best_dist: best_dist,best_name=d,dn
                        if best_dist<0.25 and best_name!=st.session_state.sel:
                            clicked_name=best_name
        if clicked_name:
            st.session_state.sel=clicked_name
            st.session_state.map_center=list(COORDS.get(clicked_name,[22.5,71.5]))
            st.session_state.map_zoom=10; st.rerun()

        # ── CURRENT vs FORECAST COMPARISON BAR ───────────────────────────────
        st.markdown(f"<div class='sh'>Current {year_2025} vs Forecast 2030 — Top 10 High-Risk Districts</div>",unsafe_allow_html=True)
        cur_top=(current_sspi[["district","sspi"]].rename(columns={"sspi":f"SSPI {year_2025}"})
                 .sort_values(f"SSPI {year_2025}",ascending=False).head(10))
        fore_top=(forecast[forecast["year"]==2030][["district","predicted_sspi"]]
                  .rename(columns={"predicted_sspi":"SSPI 2030"}))
        comp=cur_top.merge(fore_top,on="district",how="left")
        fig_comp,ax_comp=plt.subplots(figsize=(10,3.2))
        x=np.arange(len(comp)); w=0.38
        ax_comp.bar(x-w/2,comp[f"SSPI {year_2025}"],w,label=f"Current {year_2025}",color=T["accent"],alpha=0.88,edgecolor=T["bg"])
        ax_comp.bar(x+w/2,comp["SSPI 2030"],w,label="Forecast 2030",color=CLR["Critical"],alpha=0.88,edgecolor=T["bg"])
        ax_comp.set_xticks(x); ax_comp.set_xticklabels(comp["district"],rotation=35,ha="right",fontsize=7,color=T["tick"])
        ax_comp.set_ylabel("SSPI Score",color=T["text_sec"])
        ax_comp.set_title(f"Current {year_2025} vs Forecast 2030 — Top 10 High-Risk Districts",color=T["text_pri"],fontsize=9,fontweight="bold")
        ax_comp.axhline(75,color=CLR["Critical"],ls="--",lw=1,alpha=0.6)
        ax_comp.axhline(50,color=CLR["High"],ls="--",lw=1,alpha=0.5)
        ax_comp.legend(facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=8)
        ax_style(ax_comp,fig_comp); fig_comp.tight_layout()
        st.pyplot(fig_comp,use_container_width=True); plt.close()

    # ── DISTRICT INFO PANEL ───────────────────────────────────────────────────
    with ci:
        st.markdown("<div class='sh'>District Profile</div>",unsafe_allow_html=True)
        if not st.session_state.sel:
            st.markdown(f"""<div class='ic' style='text-align:center;padding:2rem 1rem'>
              <div style='font-size:2rem;margin-bottom:.5rem'>🗺️</div>
              <div style='font-size:.85rem;color:{T["text_sec"]};line-height:1.6'>
                Click any district on the map<br>or use the search box<br>to view details</div>
            </div>""",unsafe_allow_html=True)
        else:
            bar_w=min(latest_sspi,100)
            cur_row=current_sspi[current_sspi["district"]==sel]
            cur_sspi_val=cur_row["sspi"].values[0] if len(cur_row)>0 else latest_sspi
            cur_class=compute_sspi_class(cur_sspi_val); cur_clr=sc(cur_class)

            st.markdown(f"""<div class='ic {sal_class}'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div><div style='font-size:1rem;font-weight:700;margin-bottom:.3rem'>📍 {sel}</div>{zb(zone_type)}</div>
                <div style='text-align:right'>
                  <div style='font-size:.58rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.07em'>SSPI {year_2025} (current)</div>
                  <div class='big-score' style='color:{cur_clr}'>{cur_sspi_val:.1f}</div>
                  <div style='font-size:.78rem;color:{cur_clr};font-weight:700'>{cur_class} Risk</div>
                </div>
              </div>
              <div class='gauge' style='margin-top:.8rem'><div class='gfill' style='width:{bar_w}%;background:{cur_clr}'></div></div>
              <div style='display:flex;justify-content:space-between;font-size:.58rem;color:{T["text_sec"]};margin-top:.12rem'>
                <span>0 Low</span><span>25</span><span>50</span><span>75 Critical</span></div>
            </div>""",unsafe_allow_html=True)

            # Current vs Forecast mini comparison
            fore_val_d=df[df["year"]==2030]["predicted_sspi"].values
            fore_val=fore_val_d[0] if len(fore_val_d)>0 else None
            if fore_val is not None:
                delta_cf=fore_val-cur_sspi_val
                delta_clr=CLR["Critical"] if delta_cf>5 else CLR["Low"] if delta_cf<0 else CLR["High"]
                st.markdown(f"""<div class='ic' style='display:flex;justify-content:space-between;align-items:center;padding:.7rem 1rem'>
                  <div style='text-align:center'>
                    <div style='font-size:.55rem;color:{T["text_sec"]};text-transform:uppercase;margin-bottom:.15rem'>Current {year_2025}</div>
                    <div style='font-size:1.4rem;font-weight:800;font-family:Space Mono,monospace;color:{cur_clr}'>{cur_sspi_val:.1f}</div>
                  </div>
                  <div style='text-align:center'>
                    <div style='font-size:1.2rem;color:{delta_clr}'>{"▲" if delta_cf>0 else "▼"}</div>
                    <div style='font-size:.65rem;font-weight:700;color:{delta_clr}'>{abs(delta_cf):.1f}</div>
                  </div>
                  <div style='text-align:center'>
                    <div style='font-size:.55rem;color:{T["text_sec"]};text-transform:uppercase;margin-bottom:.15rem'>Forecast 2030</div>
                    <div style='font-size:1.4rem;font-weight:800;font-family:Space Mono,monospace;color:{sc(compute_sspi_class(fore_val))}'>{fore_val:.1f}</div>
                  </div>
                </div>""",unsafe_allow_html=True)

            _clr_low_env = CLR["Low"]; _clr_high_env = CLR["High"]
            st.markdown(f"""<div class='ic'>
              <div style='font-size:.58rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem'>Environmental Metrics</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:.5rem'>
                <div><div style='font-size:.58rem;color:{T["text_sec"]}'>🌿 NDVI</div><div style='font-size:1.05rem;font-weight:700;color:{_clr_low_env}'>{ndvi_avg:.3f}</div></div>
                <div><div style='font-size:.58rem;color:{T["text_sec"]}'>🌧️ Rainfall</div><div style='font-size:1.05rem;font-weight:700;color:{T["accent"]}'>{rain_avg:.0f} mm</div></div>
                <div><div style='font-size:.58rem;color:{T["text_sec"]}'>🌡️ Rabi Temp</div><div style='font-size:1.05rem;font-weight:700;color:{_clr_high_env}'>{temp_avg:.1f}°C</div></div>
                <div><div style='font-size:.58rem;color:{T["text_sec"]}'>🔮 2030 SSPI</div><div style='font-size:1.05rem;font-weight:700;color:{clr_sel}'>{f"{sspi_2030:.1f}" if sspi_2030 else "—"}</div></div>
              </div></div>""",unsafe_allow_html=True)

            st.markdown(f"<div class='ic'><div style='font-size:.58rem;color:{T['text_sec']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem'>Primary Cause</div>"
                        f"<div style='font-size:.8rem;color:{T['text_pri']};line-height:1.5'>{CAUSES.get(zone_type,'—')}</div></div>",unsafe_allow_html=True)
            st.markdown(f"<div class='ic {sal_class}'><div style='font-size:.58rem;color:{T['text_sec']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem'>Recommended Action</div>"
                        f"<div style='font-size:.78rem;color:{clr_sel};line-height:1.5'>{recommended_action}</div></div>",unsafe_allow_html=True)

            if len(dh)>1:
                fig_m,ax_m=plt.subplots(figsize=(5,1.6))
                ax_m.plot(dh["year"],dh["sspi"],"o-",color=clr_sel,lw=2,ms=4)
                ax_m.fill_between(dh["year"],dh["sspi"],alpha=0.12,color=clr_sel)
                ax_m.set_title(f"SSPI trend 2015–{latest_yr}",color=T["text_pri"],fontsize=8,pad=3)
                ax_style(ax_m,fig_m); fig_m.tight_layout(pad=0.4)
                st.pyplot(fig_m,use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<div class='sh'>Historical Trends — {sel}</div>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(6.5,3.8))
        ax.plot(dh["year"],dh["sspi"],"o-",color=clr_sel,lw=2.5,ms=7,zorder=5)
        ax.fill_between(dh["year"],dh["sspi"],alpha=0.12,color=clr_sel)
        for y,c,l in [(75,CLR["Critical"],"Critical"),(50,CLR["High"],"High"),(25,CLR["Moderate"],"Moderate")]:
            ax.axhline(y,color=c,ls="--",alpha=0.5,lw=1,label=l)
        ax.set_xlabel("Year"); ax.set_ylabel("SSPI Score")
        ax.set_title(f"Soil Salinity Proxy Index — {sel}",fontweight="bold")
        ax.legend(facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=7)
        ax_style(ax,fig); fig.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        fig2,ax2=plt.subplots(figsize=(6.5,3.8))
        ax2.plot(dh["year"],dh["ndvi"],"o-",color=CLR["Low"],lw=2.5,ms=7)
        ax2.fill_between(dh["year"],dh["ndvi"],alpha=0.12,color=CLR["Low"])
        ax2.axhline(0.35,color=CLR["High"],ls="--",lw=1,alpha=0.6,label="Stress threshold 0.35")
        ax2.set_xlabel("Year"); ax2.set_ylabel("NDVI")
        ax2.set_title(f"Vegetation Index (NDVI) — {sel}",fontweight="bold")
        ax2.legend(facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=7)
        ax_style(ax2,fig2); fig2.tight_layout(); st.pyplot(fig2); plt.close()
    c3,c4=st.columns(2)
    with c3:
        fig3,ax3=plt.subplots(figsize=(6.5,3.8))
        ax3.bar(dh["year"],dh["rainfall_annual"],color="#5c6bc0",alpha=0.85,edgecolor=T["bg"])
        ax3.axhline(dh["rainfall_annual"].mean(),color=T["accent"],ls="--",lw=1.2,label=f"Mean {dh['rainfall_annual'].mean():.0f} mm")
        ax3.set_xlabel("Year"); ax3.set_ylabel("Rainfall (mm)")
        ax3.set_title(f"Annual Rainfall — {sel}",fontweight="bold")
        ax3.legend(facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=7)
        ax_style(ax3,fig3); fig3.tight_layout(); st.pyplot(fig3); plt.close()
    with c4:
        fig4,ax4=plt.subplots(figsize=(6.5,3.8))
        ax4.plot(dh["year"],dh["temp_rabi"],"o-",color=CLR["High"],lw=2.5,ms=7)
        ax4.fill_between(dh["year"],dh["temp_rabi"],alpha=0.1,color=CLR["High"])
        ax4.set_xlabel("Year"); ax4.set_ylabel("Temperature (°C)")
        ax4.set_title(f"Rabi Season Temperature — {sel}",fontweight="bold")
        ax_style(ax4,fig4); fig4.tight_layout(); st.pyplot(fig4); plt.close()

    st.markdown("<div class='sh'>SSPI Comparison — all districts</div>",unsafe_allow_html=True)
    chart_src=(latest_all if not st.session_state.active_zone
               else latest_all[latest_all["district"].map(ZONE_MASTER).fillna("inland")==st.session_state.active_zone])
    la_s=chart_src.sort_values("sspi",ascending=False).reset_index(drop=True)
    zc={"coastal":"#0288d1","canal":"#9c27b0","inland":"#f57c00","hilly":"#388e3c"}
    fig5,ax5=plt.subplots(figsize=(14,4.5))
    ax5.set_facecolor(T["chart_bg"]); fig5.patch.set_facecolor(T["chart_fig"])
    bc=[zc.get(ZONE_MASTER.get(d,"inland"),T["accent"]) for d in la_s["district"]]
    b5=ax5.bar(range(len(la_s)),la_s["sspi"],color=bc,alpha=0.82,edgecolor=T["bg"],linewidth=0.4)
    for i,dn in enumerate(la_s["district"]):
        if dn==sel: b5[i].set_edgecolor("white"); b5[i].set_linewidth(2.5); b5[i].set_alpha(1.0)
    for y,c,l in [(75,CLR["Critical"],"Critical 75"),(50,CLR["High"],"High 50"),(25,CLR["Moderate"],"Moderate 25")]:
        ax5.axhline(y,color=c,ls="--",lw=1.2,alpha=0.7)
        ax5.text(len(la_s)-0.5,y+1,l,color=c,fontsize=7,ha="right")
    ax5.set_xticks(range(len(la_s))); ax5.set_xticklabels(la_s["district"],rotation=50,ha="right",fontsize=6.5,color=T["tick"])
    ax5.set_ylabel("SSPI Score",color=T["text_sec"]); ax5.tick_params(colors=T["tick"])
    ax5.set_title("Latest SSPI — All Gujarat Districts",color=T["text_pri"],fontsize=10,fontweight="bold")
    for sp in ax5.spines.values(): sp.set_edgecolor(T["border"])
    ax5.grid(alpha=0.12,axis="y",color=T["grid"],linewidth=0.5)
    ax5.legend(handles=[mpatches.Patch(facecolor=v,label=k.title()) for k,v in zc.items()],
               facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=8,loc="upper right")
    fig5.tight_layout(); st.pyplot(fig5,use_container_width=True); plt.close()
    bar_select=st.selectbox("",la_s["district"].tolist(),
        index=int(la_s[la_s["district"]==sel].index[0]) if sel in la_s["district"].values else 0,
        label_visibility="collapsed",key="bar_sel")
    if bar_select!=sel:
        st.session_state.sel=bar_select
        st.session_state.map_center=list(COORDS.get(bar_select,[22.5,71.5]))
        st.session_state.map_zoom=10; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<div class='sh'>Salinity Forecast 2026–2030 — {sel}</div>",unsafe_allow_html=True)
    cf1,cf2=st.columns([3,1])
    with cf1:
        fig6,ax6=plt.subplots(figsize=(9,4.5))
        ax6.set_facecolor(T["chart_bg"]); fig6.patch.set_facecolor(T["chart_fig"])
        ax6.plot(dh["year"],dh["sspi"],"o-",color=T["accent"],lw=2.5,ms=7,label="Historical SSPI",zorder=5)
        ax6.fill_between(dh["year"],dh["sspi"],alpha=0.1,color=T["accent"])
        if len(df)>0:
            ax6.plot(df["year"],df["predicted_sspi"],"s-",color=CLR["Critical"],lw=2.5,ms=7,label="Forecast 2026–2030",zorder=5)
            ax6.fill_between(df["year"],df["predicted_sspi"],alpha=0.1,color=CLR["Critical"])
            ax6.plot([dh["year"].max(),df["year"].min()],[dh.iloc[-1]["sspi"],df.iloc[0]["predicted_sspi"]],"--",color=T["text_sec"],lw=1.5,alpha=0.5)
        for y,c,l in [(75,CLR["Critical"],"Critical"),(50,CLR["High"],"High"),(25,CLR["Moderate"],"Moderate")]:
            ax6.axhline(y,color=c,ls=":",alpha=0.6,lw=1.3,label=l)
        ax6.axvline(2025.5,color=T["text_sec"],ls=":",alpha=0.3,lw=1.2)
        ylim=ax6.get_ylim()
        ax6.text(2015.3,ylim[1]*0.96,"Historical ◀",color=T["accent"],fontsize=8)
        ax6.text(2026.1,ylim[1]*0.96,"▶ Forecast",color=CLR["Critical"],fontsize=8)
        ax6.set_xlabel("Year",color=T["text_sec"]); ax6.set_ylabel("SSPI Score",color=T["text_sec"])
        ax6.set_title(f"SSPI Historical + Autoregressive Forecast — {sel}",color=T["text_pri"],fontweight="bold")
        ax6.legend(facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=8)
        ax_style(ax6,fig6); fig6.tight_layout(); st.pyplot(fig6,use_container_width=True); plt.close()
    with cf2:
        st.markdown("<div class='sh'>Year Values</div>",unsafe_allow_html=True)
        for _,row in df.iterrows():
            yr=int(row["year"]); v=row["predicted_sspi"]
            fc_cls=compute_sspi_class(v); fc=sc(fc_cls)
            st.markdown(f"<div class='fp'><span style='color:{T['text_sec']};font-size:.78rem;font-weight:600'>{yr}</span>"
                        f"<div style='text-align:right'><div style='color:{fc};font-size:1.05rem;font-weight:800;font-family:Space Mono,monospace'>{v:.1f}</div>"
                        f"<div style='color:{fc};font-size:.58rem'>{fc_cls}</div></div></div>",unsafe_allow_html=True)
    st.markdown("<div class='sh'>Top 10 High-Risk Districts — 2030</div>",unsafe_allow_html=True)
    fore30=forecast[forecast["year"]==2030].sort_values("predicted_sspi",ascending=False)
    rc5=st.columns(5)
    for i,(_,r) in enumerate(fore30.head(10).iterrows()):
        rc=sc(compute_sspi_class(r["predicted_sspi"]))
        _rp = min(r["predicted_sspi"], 100)
        with rc5[i%5]:
            st.markdown(f"<div class='ic' style='text-align:center;padding:.6rem .5rem'>"
                        f"<div style='font-size:.7rem;color:{T['text_pri']};font-weight:600;margin-bottom:.2rem'>{r['district']}</div>"
                        f"<div style='font-size:1.35rem;font-weight:800;font-family:Space Mono,monospace;color:{rc}'>{r['predicted_sspi']:.1f}</div>"
                        f"<div class='gauge' style='margin:.3rem 0'><div class='gfill' style='width:{_rp:.0f}%;background:{rc}'></div></div></div>",
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='sh'>System Accuracy Overview</div>",unsafe_allow_html=True)
    best_r2=metrics["r2"].max(); best_name=metrics.loc[metrics["r2"].idxmax(),"model"]
    best_acc=round(best_r2*100,1); avg_acc=round(metrics["r2"].mean()*100,1); worst_acc=round(metrics["r2"].min()*100,1)
    lr_acc=round(metrics[metrics["model"]=="Linear Regression"]["r2"].values[0]*100,1) if "Linear Regression" in metrics["model"].values else 0
    rf_acc=round(metrics[metrics["model"]=="Random Forest"]["r2"].values[0]*100,1) if "Random Forest" in metrics["model"].values else 0
    xgb_acc=round(metrics[metrics["model"]=="XGBoost"]["r2"].values[0]*100,1) if "XGBoost" in metrics["model"].values else 0
    tft_acc=round(metrics[metrics["model"]=="TFT Transformer"]["r2"].values[0]*100,1) if "TFT Transformer" in metrics["model"].values else 0

    oa1,oa2,oa3,oa4=st.columns(4)
    with oa1:
        st.markdown(f"""<div class='acc-banner'>
          <div style='font-size:.6rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.2rem'>Best Model</div>
          <div class='acc-ring'>{best_acc}%</div>
          <div style='font-size:.68rem;color:{T["accent"]};font-weight:700;margin-top:.2rem'>{best_name}</div>
          <div class='gauge' style='margin:.4rem 0'><div class='gfill' style='width:{best_acc}%;background:linear-gradient(90deg,{T["accent"]},#43a047)'></div></div>
          <div style='font-size:.6rem;color:{T["text_sec"]}'>R² = {best_r2:.3f}</div>
        </div>""",unsafe_allow_html=True)
    with oa2:
        st.markdown(f"""<div style='background:{T["card"]};border:2px solid #43a047;border-radius:14px;padding:1.1rem;text-align:center'>
          <div style='font-size:.6rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.2rem'>System Average</div>
          <div style='font-size:3.2rem;font-weight:900;font-family:Space Mono,monospace;color:#43a047'>{avg_acc}%</div>
          <div class='gauge' style='margin:.4rem 0'><div class='gfill' style='width:{avg_acc}%;background:#43a047'></div></div>
          <div style='font-size:.6rem;color:{T["text_sec"]}'>Across all 4 models</div>
        </div>""",unsafe_allow_html=True)
    with oa3:
        st.markdown(f"""<div style='background:{T["card"]};border:1px solid {T["border"]};border-radius:14px;padding:1.1rem'>
          <div style='font-size:.6rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem'>Classical Models</div>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem'>
            <span style='font-size:.75rem;color:{T["text_pri"]}'>🌲 Random Forest</span>
            <span style='font-size:1.1rem;font-weight:800;font-family:Space Mono,monospace;color:#43a047'>{rf_acc}%</span>
          </div>
          <div class='gauge'><div class='gfill' style='width:{rf_acc}%;background:#43a047'></div></div>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-top:.5rem;margin-bottom:.3rem'>
            <span style='font-size:.75rem;color:{T["text_pri"]}'>⚡ XGBoost</span>
            <span style='font-size:1.1rem;font-weight:800;font-family:Space Mono,monospace;color:#fb8c00'>{xgb_acc}%</span>
          </div>
          <div class='gauge'><div class='gfill' style='width:{xgb_acc}%;background:#fb8c00'></div></div>
        </div>""",unsafe_allow_html=True)
    with oa4:
        st.markdown(f"""<div style='background:{T["card"]};border:1px solid {T["border"]};border-radius:14px;padding:1.1rem'>
          <div style='font-size:.6rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem'>Deep Learning & Baseline</div>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem'>
            <span style='font-size:.75rem;color:{T["text_pri"]}'>🤖 TFT Transformer</span>
            <span style='font-size:1.1rem;font-weight:800;font-family:Space Mono,monospace;color:{T["accent"]}'>{tft_acc}%</span>
          </div>
          <div class='gauge'><div class='gfill' style='width:{tft_acc}%;background:{T["accent"]}'></div></div>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-top:.5rem;margin-bottom:.3rem'>
            <span style='font-size:.75rem;color:{T["text_pri"]}'>📐 Ridge Regression</span>
            <span style='font-size:1.1rem;font-weight:800;font-family:Space Mono,monospace;color:#5c6bc0'>{lr_acc}%</span>
          </div>
          <div class='gauge'><div class='gfill' style='width:{lr_acc}%;background:#5c6bc0'></div></div>
        </div>""",unsafe_allow_html=True)

    st.markdown(f"""<div style='background:{T["bg"]};border:1px solid {T["border"]};border-radius:8px;padding:.6rem 1rem;font-size:.76rem;color:{T["text_sec"]};margin:.6rem 0'>
      📊 System average: <b style='color:{T["accent"]}'>{avg_acc}%</b> &nbsp;|&nbsp;
      Best: <b style='color:#43a047'>{best_name} ({best_acc}%)</b> &nbsp;|&nbsp;
      Range: <b style='color:{T["text_pri"]}'>{worst_acc}% – {best_acc}%</b> &nbsp;|&nbsp;
      Improvement over baseline: <b style='color:{T["accent"]}'>+{best_acc-lr_acc:.1f}%</b>
    </div>""",unsafe_allow_html=True)

    st.markdown("<div class='sh'>Model Cards</div>",unsafe_allow_html=True)
    best_idx=int(metrics["r2"].idxmax())
    icons_m={"Linear Regression":("📐","#5c6bc0"),"Random Forest":("🌲","#43a047"),"XGBoost":("⚡","#fb8c00"),"TFT Transformer":("🤖",T["accent"])}
    mc_cols=st.columns(4)
    for i,(_,row) in enumerate(metrics.iterrows()):
        ib=(i==best_idx); icon,clr_m=icons_m.get(row["model"],("📊",T["accent"]))
        mape_val=f"{row['mape']:.1f}%" if "mape" in row and not pd.isna(row.get("mape",None)) else "—"
        acc_pct=round(row["r2"]*100,1)
        with mc_cols[i]:
            st.markdown(f"""<div class='mc {"best" if ib else ""}'>
              {"<div class='best-tag'>★ BEST</div>" if ib else ""}
              <div class='mc-icon'>{icon}</div>
              <div class='mc-name' style='color:{clr_m}'>{row["model"]}</div>
              <div class='mc-r2' style='color:{clr_m}'>{acc_pct}%</div>
              <div style='font-size:.62rem;color:{T["text_sec"]};margin-bottom:.3rem'>R² = {row["r2"]:.3f}</div>
              <div class='gauge'><div class='gfill' style='width:{acc_pct}%;background:{clr_m}'></div></div>
              <div class='mc-meta'>MAE: <b>{row["mae"]:.2f}</b> &nbsp;|&nbsp; RMSE: <b>{row["rmse"]:.2f}</b><br>MAPE: <b>{mape_val}</b></div>
            </div>""",unsafe_allow_html=True)

    ch1,ch2=st.columns(2)
    with ch1:
        fig7,ax7=plt.subplots(figsize=(6.5,3.6))
        bar_clrs=[icons_m.get(m,("",T["accent"]))[1] for m in metrics["model"]]
        b7=ax7.barh(metrics["model"],metrics["r2"]*100,color=bar_clrs,alpha=0.88,edgecolor=T["bg"],linewidth=0.4)
        for bar,val in zip(b7,metrics["r2"]):
            ax7.text(bar.get_width()+0.8,bar.get_y()+bar.get_height()/2,f"{val*100:.1f}%",va="center",color=T["text_pri"],fontsize=9,fontweight="bold")
        ax7.axvline(75,color=CLR["Low"],ls="--",lw=1.2,alpha=0.7,label="Target 75%")
        ax7.set_xlim(0,110); ax7.set_xlabel("Accuracy % (R²)",color=T["text_sec"])
        ax7.set_title("Model Accuracy Comparison",color=T["text_pri"],fontweight="bold")
        ax7.legend(facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=7)
        ax_style(ax7,fig7); fig7.tight_layout(); st.pyplot(fig7); plt.close()
    with ch2:
        fig8,ax8=plt.subplots(figsize=(6.5,3.6))
        x=np.arange(len(metrics)); w=0.35
        ax8.bar(x-w/2,metrics["mae"],w,label="MAE",color="#5c6bc0",alpha=0.88,edgecolor=T["bg"])
        ax8.bar(x+w/2,metrics["rmse"],w,label="RMSE",color="#ef5350",alpha=0.88,edgecolor=T["bg"])
        ax8.set_xticks(x); ax8.set_xticklabels(metrics["model"],rotation=15,ha="right",fontsize=7)
        ax8.set_ylabel("Error in SSPI units",color=T["text_sec"])
        ax8.set_title("MAE & RMSE Comparison",color=T["text_pri"],fontweight="bold")
        ax8.legend(facecolor=T["card"],edgecolor=T["border"],labelcolor=T["text_pri"],fontsize=8)
        ax_style(ax8,fig8); fig8.tight_layout(); st.pyplot(fig8); plt.close()

    st.markdown("<div class='sh'>Full Metrics Table</div>",unsafe_allow_html=True)
    disp=metrics.copy(); disp["accuracy_%"]=(disp["r2"]*100).round(1); disp=disp.rename(columns=str.upper)
    fmt={"MAE":"{:.2f}","RMSE":"{:.2f}","R2":"{:.3f}","ACCURACY_%":"{:.1f}%"}
    if "MAPE" in disp.columns: fmt["MAPE"]="{:.1f}"
    st.dataframe(disp.style.highlight_max(subset=["R2","ACCURACY_%"],color="#1b3a6e")
                           .highlight_min(subset=["MAE","RMSE"],color="#1b3a6e")
                           .format(fmt,na_rep="—"),use_container_width=True)

    _clr_low_f  = CLR["Low"]
    _clr_high_f = CLR["High"]
    st.markdown(f"""<div class='ic' style='margin-top:.8rem'>
      <div style='font-size:.6rem;color:{T["text_sec"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem'>SSPI Formula</div>
      <code style='color:{T["accent"]};font-size:.83rem'>SSPI = (0.40 × NDVI_deficit + 0.35 × Rainfall_deficit + 0.25 × Temp_anomaly) × 100</code>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;margin-top:.7rem'>
        <div style='font-size:.73rem;color:{T["text_sec"]}'>🌿 <b style='color:{_clr_low_f}'>NDVI deficit 40%</b><br>Vegetation stress proxy</div>
        <div style='font-size:.73rem;color:{T["text_sec"]}'>🌧️ <b style='color:{T["accent"]}'>Rainfall deficit 35%</b><br>Low rain → salt accumulation</div>
        <div style='font-size:.73rem;color:{T["text_sec"]}'>🌡️ <b style='color:{_clr_high_f}'>Temp anomaly 25%</b><br>High temp → capillary rise</div>
      </div></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='sh'>Raw Data Explorer</div>",unsafe_allow_html=True)
    _c=st.session_state["t5"]
    LABELS=["📊 Historical SSPI","🔮 Forecast 2026-2030","🗺️ All Districts Latest"]
    KEYS=["Historical_SSPI","Forecast_2026-2030","All_Districts_Latest"]
    st.markdown(f"<div style='font-size:.75rem;color:{T['accent']};font-weight:700;margin:.3rem 0 .6rem;"
                f"padding:.3rem .8rem;background:{T['card']};border:1px solid {T['accent']};"
                f"border-radius:8px;display:inline-block'>Showing: {LABELS[_c]}</div>",unsafe_allow_html=True)
    zf=st.multiselect("Zone filter",["coastal","canal","inland","hilly"],default=["coastal","canal","inland","hilly"],key="tab5_zone_filter")
    if _c==0:
        show_df=history[history["district"]==sel].sort_values("year").reset_index(drop=True).copy()
    elif _c==1:
        show_df=forecast[forecast["district"]==sel].sort_values("year").reset_index(drop=True).copy()
        show_df["zone_type"]=show_df["district"].map(ZONE_MASTER).fillna("inland")
    else:
        show_df=latest_all.sort_values("sspi",ascending=False).copy()
        show_df["zone_type"]=show_df["district"].map(ZONE_MASTER).fillna("inland")
    if zf:
        show_df=show_df[show_df["district"].map(ZONE_MASTER).fillna("inland").isin(zf)]
    st.dataframe(show_df.reset_index(drop=True),use_container_width=True,height=360)
    d1,d2=st.columns(2)
    with d1:
        st.download_button("⬇️ Download CSV",show_df.to_csv(index=False).encode("utf-8"),
                           f"{sel}_{KEYS[_c]}.csv","text/csv",use_container_width=True)
    with d2:
        st.download_button("⬇️ Download JSON",show_df.to_json(orient="records",indent=2).encode("utf-8"),
                           f"{sel}_{KEYS[_c]}.json","application/json",use_container_width=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(f"<div class='footer'>🌾 Gujarat Soil Salinity Intelligence Platform · ML + TFT · 33 Districts · 2015–2030 · MODIS GEE · NASA POWER · CSSRI Bharuch</div>",unsafe_allow_html=True)
