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

from config import COLORMAPS, DEFAULT_CITIES, MODEL_OPTIONS, MAP_STYLES, TERRAIN_TYPES, ENVIRONMENTS
from heatmap import render_heatmap, adaptive_grid_resolution
from map_utils import (create_map, update_map_with_coverage,
                       terrain_obstruction_indicator, terrain_risk_score,
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
        model = st.selectbox("Propagation Model", MODEL_OPTIONS)
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
        if model == "Rain":
            rain_rate = st.slider("Rain Rate (mm/h)", min_value=0, max_value=50, value=25)
        elif model == "Fog":
            fog_density = st.slider("Fog Density (g/m³)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        elif model in ("LongleyRice", "TIREM"):
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
        coverage_distance = st.slider("Distance (km)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)

    with st.expander("Raster Coverage Heatmap", expanded=False):
        show_heatmap = st.checkbox("Show heatmap", value=False)
        heatmap_data = st.selectbox("Data heatmap", ["Elevation", "Path Loss"], index=0)
        grid_resolution = st.slider("Grid resolution", min_value=30, max_value=350, value=300, step=10)
        cmap_name = st.selectbox("Colormap", list(COLORMAPS.keys()), index=0)
        heatmap_opacity = st.slider("Opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05)
        raster_range = st.slider("Raster Range (km)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)

        if show_heatmap and st.session_state.heatmap_ring is None:
            st.info("Gambar area (polygon/kotak) pada peta untuk menentukan kontur heatmap. "
                    "Tanpa itu, heatmap mengikuti jari-jari Raster Range.")

        if st.button("Clear area", disabled=st.session_state.heatmap_ring is None):
            st.session_state.heatmap_ring = None
            st.rerun()

    if st.button("Calculate Coverage", type="primary", use_container_width=True):
        path_loss_fn = get_model_fn(model, freq, rain_rate, fog_density, terrain_type, environment)
        path_loss = path_loss_fn(coverage_distance)
        terrain_risk = terrain_risk_score(altitude, coverage_distance)
        st.session_state.calc_result = {
            "path_loss": path_loss,
            "terrain_risk": terrain_risk,
            "distance": coverage_distance,
            "altitude": altitude,
            "location": [latitude, longitude],
        }
        st.rerun()

# ---- Main area: map only ----
location = [latitude, longitude]

if map_style != st.session_state.map_style:
    st.session_state.map_style = map_style

# Build the map fresh each rerun (cheap, avoids stale state)
m = create_map(location, map_style, add_draw=True, max_zoom=20)

if show_heatmap:
    heatmap_ring = st.session_state.heatmap_ring
    n = adaptive_grid_resolution(heatmap_ring, grid_resolution)
    try:
        m = render_heatmap(
            m, location, raster_range, n, cmap_name, heatmap_opacity,
            model, freq, rain_rate, fog_density, terrain_type, environment,
            ring=heatmap_ring, heatmap_data=heatmap_data,
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

    m = update_map_with_coverage(m, calc["location"], calc["path_loss"], calc["distance"])
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
    st.rerun()
st.session_state.prev_all_drawings = all_drawings