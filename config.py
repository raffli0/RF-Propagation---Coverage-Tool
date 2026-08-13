
"""Global constants and UI option lists shared across modules."""

# RGB anchor points (0-255) for continuous colormaps; interpolated to a 256-entry LUT.
COLORMAPS = {
    "viridis": [
        (68, 1, 84), (71, 44, 122), (59, 82, 139), (44, 113, 142),
        (33, 145, 140), (39, 173, 129), (84, 200, 98), (160, 219, 57),
        (253, 231, 37),
    ],
    "magma": [
        (0, 0, 4), (30, 15, 78), (85, 24, 129), (137, 42, 130),
        (189, 63, 119), (232, 104, 96), (250, 155, 100), (252, 220, 166),
        (252, 253, 191),
    ],
    "jet": [
        (0, 0, 128), (0, 0, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (128, 0, 0),
    ],
    "RdYlGn_r": [
        (0, 104, 55), (26, 152, 80), (102, 189, 99), (166, 217, 106),
        (217, 239, 139), (254, 224, 139), (253, 174, 97), (244, 109, 67),
        (215, 48, 39), (165, 0, 38),
    ],
    "RadioMobile": [
        (46, 204, 113), (126, 217, 111), (193, 228, 107), (241, 196, 15),
        (243, 156, 18), (230, 126, 34), (231, 76, 60), (192, 57, 43),
    ],
}

DEFAULT_CITIES = {
    # "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.914744, 107.609810),
}

MODEL_OPTIONS = ["Free Space", "Rain", "Gas", "Fog", "CloseIn", "LongleyRice", "TIREM", "RayTracing"]
MAP_STYLES = ["OpenStreetMap", "Topografi", "Satelit"]
TERRAIN_TYPES = ["average", "hilly", "mountainous"]
ENVIRONMENTS = ["urban", "rural"]

# Target physical grid cell size (meters) for drawn-area heatmaps, so buildings
# and small hills stay visible (~50-70 m) regardless of the selected area size.
GRID_TARGET_CELL_M = 60

# Longley-Rice (ITM) defaults: antenna heights, per-cell profile sampling, and a
# safety cap on the number of grid cells (each cell runs its own ITM p2p call).
# Benchmark: ~0.19 ms/cell, so the cap allows up to 500x500 (250000 cells,
# ~47 s one-time; 350x350 ~23 s; 300x300 ~18 s). After the first run results
# are served from st.cache_data.
TX_HEIGHT_DEFAULT = 30.0
RX_HEIGHT_DEFAULT = 1.5
ITM_PROFILE_POINTS = 50
MAX_ITM_GRID_CELLS = 250000
# ITM uncertainty knobs (percent): reliability (signal availability) and
# confidence (deviation of the actual environment from the model).
ITM_RELIABILITY_DEFAULT = 50.0
ITM_CONFIDENCE_DEFAULT = 50.0
# Radio-Mobile-style signal display: effective radiated power used to turn
# path loss (dB) into received signal (dBm), the RX sensitivity below which
# coverage cells are hidden, and the default polarization (0=horizontal).
TX_POWER_DBM_DEFAULT = 50.0
ITM_COVERAGE_THRESHOLD_DBM_DEFAULT = -90.0
ITM_POLARIZATION_DEFAULT = 0
# Cap on the raster side (px) embedded in the folium map as a base64 PNG.
# Above ~1200 px the HTML payload grows into many MB and st_folium fails or
# becomes unresponsive, so the coverage "disappears" from the map. The real
# ITM detail comes from the compute grid; this only bounds the display size.
MAX_OVERLAY_SIDE = 1200
