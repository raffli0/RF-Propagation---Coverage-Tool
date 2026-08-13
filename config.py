
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
