"""RF Propagation - Coverage Tool (Streamlit UI entry point).

UI orchestration only; heavy lifting lives in the sibling modules:
  config.py     constants / UI option lists
  colormaps.py  colormap LUTs, RGBA mapping, colorbar rendering
  map_utils.py  folium map construction + overlays + drawing
  heatmap.py    raster grids, cached assets, heatmap rendering
  models.py     propagation models + dispatch
  elevation.py  offline SRTM elevation sampling
"""

import streamlit as st
from streamlit_folium import st_folium

from config import (COLORMAPS, DEFAULT_CITIES, MODEL_OPTIONS, MAP_STYLES, TERRAIN_TYPES,
                    ENVIRONMENTS, ITM_CONFIDENCE_DEFAULT,
                    ITM_COVERAGE_THRESHOLD_DBM_DEFAULT, ITM_POLARIZATION_DEFAULT,
                    ITM_RELIABILITY_DEFAULT, MAX_ITM_GRID_CELLS,
                    RX_HEIGHT_DEFAULT, TX_HEIGHT_DEFAULT, TX_POWER_DBM_DEFAULT)
from heatmap import compute_heatmap_assets, render_heatmap, adaptive_grid_resolution
from map_utils import (create_map,
                       terrain_obstruction_indicator, terrain_risk_score,
                       add_coverage_raster, add_colorbar_overlay,
                       add_selection_polygon, extract_drawn_polygon,
                       remove_folium_circle)
from models import get_model_fn

st.set_page_config(layout="wide")

# Hapus atau perkecil padding bawaan Streamlit
st.markdown("""
    <style>
        /* Target class container utama Streamlit */
        .block-container {
            padding-top: 3rem; /* Ubah ke 0rem kalau mau mepet banget ke atas */
            padding-right: 0rem;
            padding-left: 0rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }

        /* Sembunyikan tombol deploy & menu utama, tapi TETAP tampilkan header
           (tempat tombol buka-tutup sidebar) supaya sidebar bisa dibuka lagi. */
        .stDeployButton {display: none;}
        div[data-testid="stMainMenu"] {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state for the map
if 'map_style' not in st.session_state:
    st.session_state.map_style = "OpenStreetMap"
if 'default_city' not in st.session_state:
    st.session_state.default_city = "Bandung"
if 'heatmap_ring' not in st.session_state:
    st.session_state.heatmap_ring = None

# ---- Sidebar controls ----
with st.sidebar:
    st.title("RF Propagation - Coverage Tool")
    with st.expander("Propagation Settings", expanded=True):
        model = st.selectbox("Propagation Model", MODEL_OPTIONS,
                             index=MODEL_OPTIONS.index("LongleyRice"))
        map_style = st.selectbox("Map Style", MAP_STYLES, index=0)
        selected_city = st.selectbox(
            "Default City",
            list(DEFAULT_CITIES.keys()),
            index=list(DEFAULT_CITIES.keys()).index(st.session_state.default_city)
        )

        freq = st.number_input("Frequency (MHz)", min_value=1, value=900)

        # Model-specific parameters
        rain_rate = 25
        fog_density = 0.5
        terrain_type = "average"
        environment = "urban"
        tx_height = TX_HEIGHT_DEFAULT
        rx_height = RX_HEIGHT_DEFAULT
        itm_reliability = ITM_RELIABILITY_DEFAULT
        itm_confidence = ITM_CONFIDENCE_DEFAULT
        tx_power_dbm = TX_POWER_DBM_DEFAULT
        itm_polarization = ITM_POLARIZATION_DEFAULT
        coverage_threshold_dbm = ITM_COVERAGE_THRESHOLD_DBM_DEFAULT
        if model == "Rain":
            rain_rate = st.slider("Rain Rate (mm/h)", min_value=0, max_value=50, value=25)
        elif model == "Fog":
            fog_density = st.slider("Fog Density (g/m³)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        elif model == "LongleyRice":
            tx_height = st.number_input("TX Antenna Height (m)", min_value=1.0, max_value=1000.0,
                                        value=TX_HEIGHT_DEFAULT, step=1.0)
            rx_height = st.number_input("RX Antenna Height (m)", min_value=1.0, max_value=1000.0,
                                        value=RX_HEIGHT_DEFAULT, step=1.0)
            itm_reliability = st.slider("Reliability (%)", min_value=1, max_value=99,
                                        value=int(ITM_RELIABILITY_DEFAULT))
            itm_confidence = st.slider("Confidence (%)", min_value=1, max_value=99,
                                       value=int(ITM_CONFIDENCE_DEFAULT))
            tx_power_dbm = st.number_input("TX Power (dBm)", min_value=-20.0, max_value=100.0,
                                           value=TX_POWER_DBM_DEFAULT, step=1.0)
            itm_polarization = 0 if st.selectbox("Polarization", ["Horizontal", "Vertical"]) == "Horizontal" else 1
            coverage_threshold_dbm = st.slider("Coverage threshold (dBm)",
                                               min_value=-160, max_value=-50,
                                               value=int(ITM_COVERAGE_THRESHOLD_DBM_DEFAULT), step=5)
        elif model == "TIREM":
            terrain_type = st.selectbox("Terrain Type", TERRAIN_TYPES)
        elif model == "RayTracing":
            environment = st.selectbox("Environment", ENVIRONMENTS)

    with st.expander("Site Location", expanded=True):
        if selected_city != st.session_state.default_city:
            st.session_state.default_city = selected_city
            st.session_state.heatmap_ring = None

        latitude = st.number_input("Latitude", format="%.6f", value=DEFAULT_CITIES[selected_city][0])
        longitude = st.number_input("Longitude", format="%.6f", value=DEFAULT_CITIES[selected_city][1])
        altitude = st.number_input("Altitude (m)", min_value=0, value=0)
        coverage_distance = st.slider("Distance (km)", min_value=0.1, max_value=50.0, value=25.0, step=0.1)

    with st.expander("Raster Coverage Heatmap", expanded=False):
        show_heatmap = st.checkbox("Show heatmap", value=False)
        heatmap_data = st.selectbox("Data heatmap", ["Elevation", "Path Loss"], index=0)
        grid_resolution = st.slider("Grid resolution", min_value=30, max_value=500, value=350, step=10)
        cmap_name = st.selectbox("Colormap", list(COLORMAPS.keys()),
                                 index=list(COLORMAPS.keys()).index("RadioMobile") if model == "LongleyRice" else 0)
        heatmap_opacity = st.slider("Opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05)
        raster_range = st.slider("Raster Range (km)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)

        if show_heatmap and st.session_state.heatmap_ring is None:
            st.info("Gambar area (polygon/kotak) pada peta untuk menentukan kontur heatmap. "
                    "Tanpa itu, heatmap mengikuti jari-jari Raster Range.")

        if st.button("Clear area", disabled=st.session_state.heatmap_ring is None):
            st.session_state.heatmap_ring = None
            st.rerun()

    if st.button("Calculate Coverage", type="primary", use_container_width=True):
        try:
            # One coverage raster for every model. LongleyRice runs a per-cell
            # ITM point-to-point call so the footprint follows terrain (hills
            # leave shadow zones); radial models (Free Space, Rain, ...) are
            # computed from the fast distance-only path-loss function.
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
            )
            st.session_state.lr_heatmap = {
                "rgba": rgba, "cbar": cbar, "bounds": out_bounds,
                "opacity": float(heatmap_opacity), "model": model,
                "range_km": float(coverage_distance),
            }
            if model == "LongleyRice":
                # Show the reference path loss at the requested distance as text
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
                )
                path_loss = path_loss_fn(coverage_distance)
            else:
                path_loss_fn = get_model_fn(
                    model, freq, rain_rate, fog_density, terrain_type, environment,
                    tx_height=tx_height, rx_height=rx_height,
                )
                path_loss = path_loss_fn(coverage_distance)
        except RuntimeError as exc:
            st.error(f"{exc}")
            st.session_state.calc_result = None
            st.session_state.lr_heatmap = None
            st.rerun()
        terrain_risk = terrain_risk_score(altitude, coverage_distance)
        st.session_state.calc_result = {
            "path_loss": path_loss,
            "terrain_risk": terrain_risk,
            "distance": coverage_distance,
            "altitude": altitude,
            "location": [latitude, longitude],
            "model": model,
        }
        st.rerun()

# ---- Main area: map only ----
location = [latitude, longitude]

if map_style != st.session_state.map_style:
    st.session_state.map_style = map_style

# Build the map fresh each rerun (cheap, avoids stale state)
m = create_map(location, map_style, add_draw=True, max_zoom=18)

# Calculate Coverage stores the coverage raster. Show heatmap controls whether
# that raster is actually mounted on the map. This prevents two overlays from
# being rendered on top of each other.
lr_heatmap = st.session_state.get("lr_heatmap")
show_persisted = (
    lr_heatmap is not None
    and lr_heatmap.get("model") == model
    and (not show_heatmap or heatmap_data == "Path Loss")
)
if show_persisted:
    m = add_coverage_raster(m, lr_heatmap["rgba"], lr_heatmap["bounds"],
                            lr_heatmap.get("opacity", 0.8))
    m = add_colorbar_overlay(m, lr_heatmap["cbar"], lr_heatmap["bounds"])

if show_heatmap:
    heatmap_ring = st.session_state.heatmap_ring
    n = adaptive_grid_resolution(heatmap_ring, grid_resolution)
    try:
        # For path-loss display, Calculate Coverage already produced the
        # raster. Do not run a second computation.
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
            )
    except RuntimeError as exc:
        st.error(f"{exc}")

# Re-add the user's drawn area so it survives map remounts
if st.session_state.heatmap_ring is not None:
    m = add_selection_polygon(m, st.session_state.heatmap_ring)

# Re-apply the persisted calculation on every rerun
calc = st.session_state.get("calc_result")
if calc is not None:
    st.write(f"Path Loss: {calc['path_loss']:.2f} dB")
    if calc["terrain_risk"] >= 0.35:
        st.warning("Terdapat risiko terhalang terrain. Area merah menandakan kemungkinan blokade elevasi.")
    else:
        st.success("Kondisi terrain memungkinkan propagasi tanpa hambatan signifikan.")

    m = terrain_obstruction_indicator(m, calc["location"], calc["altitude"], calc["distance"])

# Display the map (stable key avoids spurious remounts)
map_data = st_folium(m, width=True, height=550, key="coverage_map")

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