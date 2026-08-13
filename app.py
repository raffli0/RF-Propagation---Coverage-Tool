"""RF Propagation - Coverage Tool (Streamlit UI entry point).

UI orchestration only; heavy lifting lives in the sibling modules:
  config.py     constants / UI option lists
  colormaps.py  colormap LUTs, RGBA mapping, colorbar rendering
  map_utils.py  folium map construction + overlays + drawing
  heatmap.py    raster grids, cached assets, heatmap rendering
  models.py     propagation models + dispatch
  elevation.py  offline SRTM elevation sampling
"""

import time

import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from config import (COLORMAPS, DEFAULT_CITIES, MODEL_OPTIONS, MAP_STYLES, TERRAIN_TYPES,
                    ENVIRONMENTS, ITM_CONFIDENCE_DEFAULT,
                    ITM_COVERAGE_THRESHOLD_DBM_DEFAULT, ITM_POLARIZATION_DEFAULT,
                    ITM_RELIABILITY_DEFAULT, MAX_ITM_GRID_CELLS,
                    RX_HEIGHT_DEFAULT, TX_HEIGHT_DEFAULT, TX_POWER_DBM_DEFAULT,
                    SOIL_TYPES, CLIMATE_ZONES, SOIL_TYPE_DEFAULT, CLIMATE_ZONE_DEFAULT)
from heatmap import compute_heatmap_assets, render_heatmap, adaptive_grid_resolution
from map_utils import (create_map,
                       add_coverage_raster, add_colorbar_overlay,
                       add_selection_polygon, extract_drawn_polygon,
                       remove_folium_circle, add_tx_marker, add_location_readout)
from models import get_model_fn
from elevation import sample_point_elevation

st.set_page_config(layout="wide")

# ---------------------------------------------------------------------------
# Styling (dark "protelecom" theme) + CloudRF-flavoured chrome
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
        /* Fixed, non-scrollable full-height dashboard */
        html, body, .stApp { height: 100%; overflow: hidden; }
        .main ::-webkit-scrollbar { width: 0; height: 0; }

        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            height: 100vh;
            height: 100dvh;
            position: relative;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .stDeployButton {display: none;}
        div[data-testid="stMainMenu"] {visibility: hidden;}
        footer {visibility: hidden;}

        /* Slim header */
        .rf-header {
            flex: 0 0 auto;
            background: linear-gradient(90deg, #0e1726, #16203a);
            padding: 9px 18px;
            display: flex; align-items: center; justify-content: space-between;
            border-bottom: 1px solid #233152;
        }
        .rf-header h1 { font-size: 18px; margin: 0; color: #e6edf3; }
        .rf-header p { margin: 1px 0 0; font-size: 11px; color: #9fb3c8; }
        .rf-badge {
            background: #00b4d8; color: #06283d; font-weight: 700;
            font-size: 11px; padding: 4px 10px; border-radius: 999px;
            white-space: nowrap;
        }

        /* Map fills the remaining viewport height */
        .element-container:has(.stFolium) { flex: 1 1 auto; min-height: 0; }
        .stFolium { height: 100% !important; }
        .stFolium > div:first-child { height: 100% !important; }

        /* Overlay: metrics (top) */
        .rf-overlay-metrics {
            position: absolute;
            top: clamp(50px, 7vh, 64px); left: clamp(8px, 1.2vw, 12px);
            right: clamp(8px, 1.2vw, 12px); z-index: 1000;
            display: flex; gap: clamp(5px, .6vw, 8px); flex-wrap: wrap; pointer-events: none;
        }
        .rf-mcard {
            flex: 1 1 0; min-width: clamp(96px, 14vw, 150px);
            background: rgba(22,32,58,.82); border: 1px solid #233152;
            border-radius: 10px; padding: clamp(6px, .8vw, 8px) clamp(8px, 1vw, 12px);
            backdrop-filter: blur(6px);
        }
        .rf-mcard .lbl { font-size: clamp(9px, .8vw, 10px); letter-spacing: .06em; text-transform: uppercase; color: #7f93ab; }
        .rf-mcard .val { font-size: clamp(15px, 1.5vw, 20px); font-weight: 700; color: #e6edf3; margin-top: 2px; }
        .rf-mcard .val small { font-size: clamp(10px, .9vw, 11px); font-weight: 600; color: #9fb3c8; }

        /* Overlay: output console (bottom-right) */
        .rf-overlay-console {
            position: absolute; bottom: clamp(8px, 1.2vw, 12px); right: clamp(8px, 1.2vw, 12px);
            width: min(330px, 86vw); max-height: min(44vh, 420px); overflow: auto; z-index: 1000;
            background: rgba(14,23,38,.9); border: 1px solid #233152;
            border-radius: 10px; padding: clamp(9px, 1vw, 12px) clamp(10px, 1.2vw, 14px);
            backdrop-filter: blur(6px);
        }
        .rf-overlay-console h3 { margin: 0 0 6px; font-size: clamp(11px, 1vw, 13px); color: #9fb3c8; }
        .rf-status { font-size: clamp(10px, .9vw, 11px); color: #7ee0c0; margin-bottom: 8px; }
        .rf-status.err { color: #ff8da1; }
        .rf-ptable { width: 100%; border-collapse: collapse; font-size: clamp(11px, .95vw, 12px); }
        .rf-ptable td { padding: 3px 4px; border-bottom: 1px solid #1c2942; color: #cdd9e8; }
        .rf-ptable td.k { color: #7f93ab; }
        .rf-ptable td.v { text-align: right; font-weight: 600; color: #e6edf3; }

        /* Overlay: empty state (center) */
        .rf-overlay-empty {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
            z-index: 900; width: min(480px, 88vw); text-align: center;
            background: rgba(14,23,38,.85); border: 1px solid #233152;
            border-radius: 12px; padding: clamp(14px, 2vw, 18px) clamp(16px, 2.4vw, 22px);
            backdrop-filter: blur(6px);
        }

        /* Responsive: narrow / short viewports */
        @media (max-width: 900px) {
            .rf-overlay-metrics { gap: 4px; }
            .rf-mcard { flex: 1 1 44%; min-width: 44%; }
            .rf-overlay-console {
                left: clamp(8px, 1.2vw, 12px); width: auto; max-height: 40vh;
            }
            .rf-header h1 { font-size: 15px; }
            .rf-header p { display: none; }
        }
        @media (max-width: 560px) {
            .rf-mcard { flex: 1 1 100%; min-width: 100%; }
            .rf-overlay-console { max-height: 36vh; }
        }

        /* Map readout chip */
        .rf-map-readout {
            position: absolute; z-index: 999; bottom: 12px; left: 54px;
            background: rgba(14,23,38,.85); color: #e6edf3;
            padding: 4px 10px; border-radius: 6px;
            font: 12px/1.4 sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,.4);
            pointer-events: none;
        }

        /* Section titles inside sidebar */
        .rf-section {
            font-size: 12px; font-weight: 700; color: #00b4d8;
            margin: 4px 0 2px; text-transform: uppercase; letter-spacing: .04em;
        }

        /* Run button */
        .stButton > button[data-baseweb="button"] {
            background: #00b4d8; color: #06283d; font-weight: 700;
            border: none; border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if 'map_style' not in st.session_state:
    st.session_state.map_style = "OpenStreetMap"
if 'default_city' not in st.session_state:
    st.session_state.default_city = "Bandung"
if 'heatmap_ring' not in st.session_state:
    st.session_state.heatmap_ring = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="rf-header">'
    '<div><h1>RF Propagation — Coverage Tool</h1>'
    '<p>Terrain-aware coverage planning · Longley-Rice (ITM), Free Space &amp; mehr</p></div>'
    '<div class="rf-badge">MODE: ENGINEER</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — CloudRF-style: LIVE zone (map view) + FORM zone (apply on RUN)
# ---------------------------------------------------------------------------
with st.sidebar:
    # ---- LIVE ZONE: Site / TX (map recenters immediately) ----
    with st.expander("1. Site / TX", expanded=True):
        st.markdown('<div class="rf-section">Lokasi &amp; Peta</div>', unsafe_allow_html=True)
        map_style = st.selectbox("Map Style", MAP_STYLES, index=0)
        selected_city = st.selectbox(
            "Default City",
            list(DEFAULT_CITIES.keys()),
            index=list(DEFAULT_CITIES.keys()).index(st.session_state.default_city)
        )
        latitude = st.number_input("Latitude", format="%.6f", value=DEFAULT_CITIES[selected_city][0])
        longitude = st.number_input("Longitude", format="%.6f", value=DEFAULT_CITIES[selected_city][1])
        altitude = st.number_input("Altitude (m)", min_value=0, value=0,
                                   help="Ketinggian elemen pemancar di atas permukaan laut.")

    # ---- LIVE ZONE: Propagation Model (drives which form fields appear) ----
    with st.expander("2. Propagation Model", expanded=True):
        model = st.selectbox("Propagation Model", MODEL_OPTIONS,
                             index=MODEL_OPTIONS.index("LongleyRice"))

    # ---- FORM ZONE: compute parameters (apply on RUN) ----
    with st.form("compute_form"):
        # Signal & Link
        with st.expander("3. Signal & Link", expanded=True):
            freq = st.number_input("Frequency (MHz)", min_value=1, value=900)
            tx_power_dbm = st.number_input("TX Power (dBm)", min_value=-20.0, max_value=100.0,
                                           value=TX_POWER_DBM_DEFAULT, step=1.0)
            coverage_threshold_dbm = st.slider("Coverage threshold (dBm)",
                                               min_value=-160, max_value=-50,
                                               value=int(ITM_COVERAGE_THRESHOLD_DBM_DEFAULT), step=5)
            coverage_distance = st.slider("Distance (km)", min_value=0.1, max_value=50.0,
                                         value=25.0, step=0.1)

        # Model-specific settings
        with st.expander("4. Model Settings", expanded=True):
            rain_rate = 25
            fog_density = 0.5
            terrain_type = "average"
            environment = "urban"
            tx_height = TX_HEIGHT_DEFAULT
            rx_height = RX_HEIGHT_DEFAULT
            itm_reliability = ITM_RELIABILITY_DEFAULT
            itm_confidence = ITM_CONFIDENCE_DEFAULT
            itm_polarization = ITM_POLARIZATION_DEFAULT
            soil_eps = None
            soil_sgm = None
            soil_klim = None
            smoothing = 0
            if model == "Rain":
                rain_rate = st.slider("Rain Rate (mm/h)", min_value=0, max_value=50, value=25)
            elif model == "Fog":
                fog_density = st.slider("Fog Density (g/m³)", min_value=0.0, max_value=1.0,
                                        value=0.5, step=0.1)
            elif model == "LongleyRice":
                tx_height = st.number_input("TX Antenna Height (m)", min_value=1.0, max_value=1000.0,
                                            value=TX_HEIGHT_DEFAULT, step=1.0)
                rx_height = st.number_input("RX Antenna Height (m)", min_value=1.0, max_value=1000.0,
                                            value=RX_HEIGHT_DEFAULT, step=1.0)
                itm_reliability = st.slider("Reliability (%)", min_value=1, max_value=99,
                                            value=int(ITM_RELIABILITY_DEFAULT))
                itm_confidence = st.slider("Confidence (%)", min_value=1, max_value=99,
                                           value=int(ITM_CONFIDENCE_DEFAULT))
                itm_polarization = 0 if st.selectbox("Polarization", ["Horizontal", "Vertical"]) == "Horizontal" else 1
                soil_type = st.selectbox("Soil Type", list(SOIL_TYPES.keys()),
                                         index=list(SOIL_TYPES.keys()).index(SOIL_TYPE_DEFAULT))
                climate_zone = st.selectbox("Climate Zone", list(CLIMATE_ZONES.keys()),
                                            index=list(CLIMATE_ZONES.keys()).index(CLIMATE_ZONE_DEFAULT))
                soil_eps, soil_sgm = SOIL_TYPES[soil_type]
                soil_klim = CLIMATE_ZONES[climate_zone]
            elif model == "TIREM":
                terrain_type = st.selectbox("Terrain Type", TERRAIN_TYPES)
            elif model == "RayTracing":
                environment = st.selectbox("Environment", ENVIRONMENTS)

        # Heatmap & Display compute params
        with st.expander("5. Heatmap & Display", expanded=True):
            cmap_name = st.selectbox("Colormap", list(COLORMAPS.keys()),
                                     index=list(COLORMAPS.keys()).index("RadioMobile"))
            grid_resolution = st.slider("Grid resolution", min_value=30, max_value=700, value=450, step=10)
            smoothing = st.slider("Smoothing contour", min_value=0, max_value=2, value=1,
                                  help="0 = off, 1 = ringan, 2 = kuat. Menghaluskan transisi warna dBm.")
            raster_range = st.slider("Raster Range (km)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
            heatmap_data = st.selectbox("Data heatmap", ["Elevation", "Path Loss"], index=0)

        run = st.form_submit_button("Calculate Coverage", type="primary", use_container_width=True)

    # ---- LIVE ZONE: Display toggles (cheap, no recompute) ----
    with st.expander("6. Display", expanded=True):
        show_heatmap = st.checkbox("Show heatmap", value=False)
        heatmap_opacity = st.slider("Opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05)
        if st.button("Clear area", disabled=st.session_state.heatmap_ring is None):
            st.session_state.heatmap_ring = None
            st.rerun()

    # -----------------------------------------------------------------------
    # Compute on RUN
    # -----------------------------------------------------------------------
    if run:
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            st.error("Latitude / Longitude berada di luar rentang valid "
                     "(lat -90..90, lon -180..180).")
        else:
            try:
                status = ("Menghitung coverage (ITM radial engine)…" if model == "LongleyRice"
                          else f"Menghitung coverage ({model})…")
                with st.spinner(status):
                    t0 = time.time()
                    max_side = int(MAX_ITM_GRID_CELLS ** 0.5)
                    n_cov = int(min(grid_resolution, max_side))
                    rgba, cbar, out_bounds = compute_heatmap_assets(
                        tuple([latitude, longitude]), float(coverage_distance),
                        n_cov, cmap_name, model, float(freq),
                        float(rain_rate), float(fog_density), terrain_type,
                        environment, None, None, float(tx_height), float(rx_height),
                        float(itm_reliability), float(itm_confidence),
                        float(tx_power_dbm), int(itm_polarization),
                        float(coverage_threshold_dbm),
                        int(smoothing), soil_eps, soil_sgm, soil_klim,
                    )
                    runtime = time.time() - t0
                    cov_rgba = np.asarray(rgba)
                    coverage_pct = float((cov_rgba[..., 3] > 0).mean() * 100)

                    if model == "LongleyRice":
                        import math
                        from itm_adapter import terrain_profile
                        lat1 = math.radians(latitude)
                        lon1 = math.radians(longitude)
                        angd = coverage_distance / 6371.0
                        lat2 = math.asin(math.sin(lat1) * math.cos(angd)
                                         + math.cos(lat1) * math.sin(angd))
                        lon2 = lon1 + math.atan2(math.sin(0.0) * math.sin(angd) * math.cos(lat1),
                                                 math.cos(angd) - math.sin(lat1) * math.sin(lat2))
                        lr_profile = terrain_profile(
                            latitude, longitude, math.degrees(lat2), math.degrees(lon2)
                        )
                        path_loss_fn = get_model_fn(
                            model, freq, rain_rate, fog_density, terrain_type, environment,
                            tx_height=tx_height, rx_height=rx_height,
                            longley_rice_profile=lr_profile,
                            longley_rice_reliability=float(itm_reliability),
                            longley_rice_confidence=float(itm_confidence),
                            longley_rice_polarization=int(itm_polarization),
                            longley_rice_eps=soil_eps, longley_rice_sgm=soil_sgm,
                            longley_rice_klim=soil_klim,
                        )
                        path_loss = path_loss_fn(coverage_distance)
                    else:
                        path_loss_fn = get_model_fn(
                            model, freq, rain_rate, fog_density, terrain_type, environment,
                            tx_height=tx_height, rx_height=rx_height,
                        )
                        path_loss = path_loss_fn(coverage_distance)

                st.session_state.lr_heatmap = {
                    "rgba": rgba, "cbar": cbar, "bounds": out_bounds,
                    "opacity": float(heatmap_opacity), "model": model,
                    "range_km": float(coverage_distance),
                }
                st.session_state.calc_result = {
                    "path_loss": path_loss,
                    "distance": coverage_distance,
                    "altitude": altitude,
                    "location": [latitude, longitude],
                    "model": model,
                    "runtime": runtime,
                    "coverage_pct": coverage_pct,
                }
                try:
                    gnd = sample_point_elevation(latitude, longitude)
                    if np.isfinite(gnd) and abs(gnd - altitude) > 100:
                        st.warning(f"Altitude input ({altitude} m) menyimpang >100 m dari "
                                   f"elevasi terrain SRTM ({gnd:.0f} m).")
                except Exception:
                    pass
                st.toast(f"Coverage selesai — {runtime:.1f}s · {n_cov * n_cov:,} sel")
            except RuntimeError as exc:
                st.error(f"{exc}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
location = [latitude, longitude]
if map_style != st.session_state.map_style:
    st.session_state.map_style = map_style
if selected_city != st.session_state.default_city:
    st.session_state.default_city = selected_city
    st.session_state.heatmap_ring = None

lr_heatmap = st.session_state.get("lr_heatmap")
calc = st.session_state.get("calc_result")

# Build the map fresh each rerun (cheap, avoids stale state)
m = create_map(location, map_style, add_draw=True, max_zoom=18)

# Persisted coverage raster (from Calculate Coverage)
show_persisted = (
    lr_heatmap is not None
    and lr_heatmap.get("model") == model
    and (not show_heatmap or heatmap_data == "Path Loss")
)
if show_persisted:
    m = add_coverage_raster(m, lr_heatmap["rgba"], lr_heatmap["bounds"],
                            float(heatmap_opacity))
    m = add_colorbar_overlay(m, lr_heatmap["cbar"], lr_heatmap["bounds"])

# Live drawn-area heatmap
if show_heatmap:
    heatmap_ring = st.session_state.heatmap_ring
    n = adaptive_grid_resolution(heatmap_ring, grid_resolution)
    try:
        use_persisted = (
            lr_heatmap is not None
            and lr_heatmap.get("model") == model
            and heatmap_data == "Path Loss"
        )
        if not use_persisted:
            m = render_heatmap(
                m, location, raster_range, n, cmap_name, heatmap_opacity,
                model, freq, rain_rate, fog_density, terrain_type, environment,
                ring=heatmap_ring, heatmap_data=heatmap_data,
                tx_height=tx_height, rx_height=rx_height,
                reliability=float(itm_reliability), confidence=float(itm_confidence),
                tx_power_dbm=float(tx_power_dbm), ipol=int(itm_polarization),
                coverage_threshold_dbm=float(coverage_threshold_dbm),
                smoothing=int(smoothing), eps=soil_eps, sgm=soil_sgm,
                klim=soil_klim,
            )
    except RuntimeError as exc:
        st.error(f"{exc}")

# Re-add the user's drawn area so it survives map remounts
if st.session_state.heatmap_ring is not None:
    m = add_selection_polygon(m, st.session_state.heatmap_ring)

# TX marker + coordinate readout
m = add_tx_marker(m, location, label=f"TX · {model}")
m = add_location_readout(m, location, mode_text=model)

# Display the map (fills the viewport; overlay panels sit on top)
map_data = st_folium(m, width=True, height=600, key="coverage_map")

# Capture the drawn shape and persist it as the heatmap contour
new_ring = extract_drawn_polygon(map_data)
if new_ring is not None and new_ring != st.session_state.heatmap_ring:
    st.session_state.heatmap_ring = new_ring
    st.rerun()

# Remove button (Leaflet Draw): deleting the drawn shape also removes the
# coverage circle by calling remove_folium_circle(map_obj).
all_drawings = map_data.get("all_drawings") or []
prev_drawings = st.session_state.get("prev_all_drawings", all_drawings)
if (
    st.session_state.heatmap_ring is not None
    and prev_drawings
    and not all_drawings
):
    m = remove_folium_circle(m)
    st.session_state.heatmap_ring = None
    st.session_state.calc_result = None
    st.session_state.lr_heatmap = None
    st.rerun()
st.session_state.prev_all_drawings = all_drawings

# ---- Overlays (absolute, do not add to page height) ----
def _mcard(label, value, unit=""):
    return (f'<div class="rf-mcard"><div class="lbl">{label}</div>'
            f'<div class="val">{value}<small> {unit}</small></div></div>')

m_pl = f"{calc['path_loss']:.1f}" if calc else "—"
m_model = calc["model"] if calc else "—"
m_freq = f"{freq:.0f}" if calc else "—"
m_dist = f"{lr_heatmap['range_km']:.1f}" if lr_heatmap else "—"
cov = calc.get("coverage_pct") if calc else None
m_cov = f"{cov:.1f}" if cov is not None else "—"
st.markdown(
    f'<div class="rf-overlay-metrics">'
    f'{_mcard("Path Loss", m_pl, "dB")}'
    f'{_mcard("Model", m_model)}'
    f'{_mcard("Frequency", m_freq, "MHz")}'
    f'{_mcard("Distance", m_dist, "km")}'
    f'{_mcard("Coverage", m_cov, "%")}'
    f'</div>',
    unsafe_allow_html=True,
)

if lr_heatmap is None:
    st.markdown(
        '<div class="rf-overlay-empty">'
        '<div style="font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:6px;">'
        'Belum ada coverage</div>'
        '<div style="font-size:12px;color:#9fb3c8;line-height:1.5;">'
        'Atur parameter di sidebar, lalu klik <b>Calculate Coverage</b>. '
        'Gambar area (polygon) pada peta untuk menentukan kontur heatmap.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

if calc is not None:
    rows = [
        ("Propagation Model", model),
        ("Frequency", f"{freq:.0f} MHz"),
        ("TX Power", f"{tx_power_dbm:.0f} dBm"),
        ("Coverage threshold", f"{coverage_threshold_dbm:.0f} dBm"),
        ("Distance", f"{coverage_distance:.1f} km"),
        ("Path Loss", f"{calc['path_loss']:.2f} dB"),
        ("Coverage area", f"{calc.get('coverage_pct', 0):.1f} %"),
        ("Grid", f"{grid_resolution} px"),
        ("Colormap", cmap_name),
    ]
    if model == "LongleyRice":
        rows += [
            ("TX Height", f"{tx_height:.1f} m"),
            ("RX Height", f"{rx_height:.1f} m"),
            ("Reliability", f"{itm_reliability:.0f} %"),
            ("Confidence", f"{itm_confidence:.0f} %"),
        ]
    ptable = '<table class="rf-ptable">' + "".join(
        f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>" for k, v in rows
    ) + "</table>"
    st.markdown(
        f'<div class="rf-overlay-console">'
        f'<h3>Output Console</h3>'
        f'<div class="rf-status">✓ Coverage dihitung ({calc.get("runtime", 0):.2f} s) · model {calc["model"]}</div>'
        f'{ptable}</div>',
        unsafe_allow_html=True,
    )
