"""Raster grid computation and heatmap rendering (path loss & elevation).

Grids are computed once and cached per unique combination of inputs so slider
moves and area re-selections reuse work instead of recomputing.
"""

import numpy as np
import streamlit as st

from colormaps import (cap_overlay_size, colorize, nice_limits, colorbar_image,
                       smooth_upscale)
from config import (GRID_TARGET_CELL_M, ITM_CONFIDENCE_DEFAULT,
                    ITM_COVERAGE_THRESHOLD_DBM_DEFAULT, ITM_POLARIZATION_DEFAULT,
                    ITM_PROFILE_POINTS, ITM_RELIABILITY_DEFAULT, MAX_ITM_GRID_CELLS,
                    MAX_OVERLAY_SIDE, RX_HEIGHT_DEFAULT, TX_HEIGHT_DEFAULT,
                    TX_POWER_DBM_DEFAULT)
from elevation import sample_elevation
from map_utils import add_coverage_raster, add_colorbar_overlay
from models import get_model_fn, longley_rice_path_loss


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def coverage_grid(location, range_km=10, n=80, bounds=None):
    """Build an n x n raster grid. Row 0 = north (origin 'upper').

    If `bounds` ([[lat_min, lon_min], [lat_max, lon_max]]) is given the grid
    fills exactly that area (user-drawn selection); otherwise it is centered on
    the transmitter using `range_km` as the half-width in km.
    """
    lat0, lon0 = location
    if bounds is not None:
        (lat_min, lon_min), (lat_max, lon_max) = bounds
    else:
        range_km = max(range_km, 0.1)
        dlat = (range_km * 1000) / 111320.0
        dlon = (range_km * 1000) / (111320.0 * max(np.cos(np.radians(lat0)), 0.05))
        lat_min, lat_max = lat0 - dlat, lat0 + dlat
        lon_min, lon_max = lon0 - dlon, lon0 + dlon
    lat = np.linspace(lat_max, lat_min, n)
    lon = np.linspace(lon_min, lon_max, n)
    lon_m, lat_m = np.meshgrid(lon, lat)
    dist = np.maximum(haversine_km(lat_m, lon_m, lat0, lon0), 1e-6)
    out_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    return lat_m, lon_m, dist, out_bounds


def polygon_mask(lat_m, lon_m, ring):
    """Boolean mask of grid points inside a polygon ring (ray casting, vectorized).

    `ring` is a list of [lat, lon] vertices. Grid points outside the ring get
    masked out so the raster follows the drawn contour instead of a rectangle.
    """
    ring = np.asarray(ring, dtype=float)
    if ring.ndim != 2 or ring.shape[1] != 2 or ring.shape[0] < 3:
        return None
    lat, lon = ring[:, 0], ring[:, 1]
    x = lon_m.ravel()
    y = lat_m.ravel()
    inside = np.zeros(x.size, dtype=bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        cond = (lon[i] > x) != (lon[j] > x)
        cond &= y < (lat[j] - lat[i]) * (x - lon[i]) / (max(lon[j] - lon[i], 1e-12)) + lat[i]
        inside ^= cond
        j = i
    return inside.reshape(lat_m.shape)


def ring_bounds(ring):
    """Return [[lat_min, lon_min], [lat_max, lon_max]] enclosing a polygon ring."""
    ring = np.asarray(ring, dtype=float)
    return [[float(ring[:, 0].min()), float(ring[:, 1].min())],
            [float(ring[:, 0].max()), float(ring[:, 1].max())]]

# bentuk heatmap 
def apply_contour_mask(rgba, lat_m, lon_m, dist, range_km, ring):
    """Set alpha=0 for grid cells outside the drawn contour.

    Jika ring (polygon) ada, potong sesuai bentuk yang digambar.
    Jika tidak ada, potong jadi lingkaran radial berjari-jari `range_km`
    berpusat di pemancar (mengikuti unit sinyal TX), bukan kotak persegi.
    """
    if ring is not None:
        mask = polygon_mask(lat_m, lon_m, ring)
        if mask is not None:
            rgba[..., 3][~mask] = 0
    else:
        rgba[..., 3][np.asarray(dist) > range_km] = 0

    return rgba


def adaptive_grid_resolution(ring, max_n, target_cell_m=GRID_TARGET_CELL_M):
    """Scale resolution so each grid cell is ~`target_cell_m` wide, capped by `max_n`.

    Small drawn areas get proportionally fewer points than large ones, keeping a
    roughly constant physical cell size (~60 m) so buildings and small hills stay
    visible regardless of the selected area's extent. `max_n` (from the grid
    resolution slider) caps the point count so computation stays cheap.
    """
    if ring is None:
        return max_n
    bounds = ring_bounds(ring)
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    dlat_km = haversine_km(lat_min, lon_min, lat_max, lon_min)
    dlon_km = haversine_km(lat_min, lon_min, lat_min, lon_max)
    span_km = max(dlat_km, dlon_km)
    n = int(np.ceil(span_km * 1000.0 / target_cell_m)) + 1
    return int(np.clip(n, 30, max_n))


def _longley_rice_path_points(lat0, lon0, lat_m, lon_m, n_points):
    """(N, K) lat/lon samples along each great-circle path transmitter -> cell.

    Vectorized over all N grid cells so a single `sample_elevation` call can
    fetch every profile at once (one tile pass instead of N raster opens).
    """
    lat_m = np.asarray(lat_m, dtype=float).ravel()
    lon_m = np.asarray(lon_m, dtype=float).ravel()
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    latr = np.radians(lat_m)
    lonr = np.radians(lon_m)

    p1 = np.array([np.cos(lat0) * np.cos(lon0),
                   np.cos(lat0) * np.sin(lon0),
                   np.sin(lat0)])                                   # (3,)
    p2 = np.stack([np.cos(latr) * np.cos(lonr),
                   np.cos(latr) * np.sin(lonr),
                   np.sin(latr)], axis=0)                           # (3, N)

    a = np.arccos(np.clip(np.einsum('i,in->n', p1, p2), -1.0, 1.0))  # (N,)
    ts = np.linspace(0.0, 1.0, n_points)[:, None]                    # (K, 1)
    sa = np.where(a < 1e-9, 1.0, np.sin(a))                          # (N,)

    c1 = np.sin((1.0 - ts) * a[None, :])                             # (K, N)
    c2 = np.sin(ts * a[None, :])                                     # (K, N)
    p = (c1[:, :, None] * p1[None, None, :]
         + c2[:, :, None] * p2.T[None, :, :]) / sa[None, :, None]    # (K, N, 3)

    lats = np.degrees(np.arcsin(np.clip(p[..., 2], -1.0, 1.0)))      # (K, N)
    lons = np.degrees(np.arctan2(p[..., 1], p[..., 0]))              # (K, N)
    return lats.T, lons.T                                            # (N, K)


def _longley_rice_loss_grid(location, lat_m, lon_m, dist, bounds, freq, tx_height,
                            rx_height, reliability, confidence, ipol=0):
    """Path-loss raster for LongleyRice: one ITM p2p call per grid cell.

    Each cell gets its own SRTM terrain profile (sampled once for the whole
    grid) and a distance, so the result is a true point-to-point estimate
    instead of a directionless radial loss.
    """
    lat0, lon0 = location
    n_cells = int(lat_m.size)
    if n_cells > MAX_ITM_GRID_CELLS:
        raise RuntimeError(
            f"Grid {n_cells} sel terlalu besar untuk LongleyRice "
            f"(maks {MAX_ITM_GRID_CELLS}). Perkecil area atau kurangi resolusi."
        )

    path_lats, path_lons = _longley_rice_path_points(lat0, lon0, lat_m, lon_m, ITM_PROFILE_POINTS)
    elev = sample_elevation(path_lats, path_lons)
    elev = np.nan_to_num(elev, nan=0.0)

    dist_c = np.maximum(np.asarray(dist, dtype=float).ravel(), 0.05)
    loss = np.empty(n_cells, dtype=float)
    for i in range(n_cells):
        loss[i] = longley_rice_path_loss(
            freq, dist_c[i], tx_height, rx_height, elev[i].tolist(),
            ipol=ipol, reliability=reliability, confidence=confidence,
        )
    return loss.reshape(lat_m.shape)


@st.cache_data(show_spinner=False, max_entries=32)
def compute_heatmap_assets(location, range_km, n, cmap_name, model, freq,
                           rain_rate, fog_density, terrain_type, environment,
                           bounds, ring, tx_height=TX_HEIGHT_DEFAULT,
                           rx_height=RX_HEIGHT_DEFAULT,
                           reliability=ITM_RELIABILITY_DEFAULT,
                           confidence=ITM_CONFIDENCE_DEFAULT,
                           tx_power_dbm=TX_POWER_DBM_DEFAULT,
                           ipol=ITM_POLARIZATION_DEFAULT,
                           coverage_threshold_dbm=ITM_COVERAGE_THRESHOLD_DBM_DEFAULT):
    """Compute the path-loss raster (RGBA) and colorbar PNG for a grid.

    Cached by every parameter that affects the output so re-renders and area
    re-selections reuse work instead of recomputing (scalable for big grids).
    When `ring` is given the raster is masked to that contour; otherwise it is
    masked to the radial coverage circle so it never renders as a square box.

    LongleyRice branches into a per-cell ITM computation (each grid cell gets
    its own SRTM terrain profile + point-to-point loss) and is displayed the
    Radio-Mobile way: received signal in dBm (`tx_power_dbm - loss`), cells
    below `coverage_threshold_dbm` hidden, then Lanczos-upscaled so the
    coverage edge blends instead of showing blocky pixels. Other models keep
    the fast radial `path_loss_fn(dist)` path (path loss in dB).
    """
    lat_m, lon_m, dist, out_bounds = coverage_grid(location, range_km, n, bounds=bounds)
    if model == "LongleyRice":
        loss = _longley_rice_loss_grid(
            location, lat_m, lon_m, dist, out_bounds, freq, tx_height, rx_height,
            reliability, confidence, ipol=ipol,
        )
        rx_dbm = float(tx_power_dbm) - loss
        # LR coverage uses a fixed signal scale. The coverage threshold, not
        # the radial distance, defines the visible propagation footprint.
        vmin = float(coverage_threshold_dbm) - 20.0
        vmax = float(tx_power_dbm)
        if vmax <= vmin:
            vmax = vmin + 1.0
        rgba = colorize(rx_dbm, cmap_name, vmin, vmax)
    else:
        path_loss_fn = get_model_fn(model, freq, rain_rate, fog_density, terrain_type, environment)
        loss = path_loss_fn(dist)
        vmin, vmax = nice_limits(float(np.min(loss)), float(np.max(loss)))
        rgba = colorize(loss, cmap_name, vmin, vmax)

    if model == "LongleyRice":
        # Coverage is threshold-driven, NOT clipped to a circle or square:
        # a cell is covered when Rx >= RX sensitivity, so the footprint spreads
        # radially from the TX and hills/valleys leave uncovered shadow zones
        # (exactly like Radio Mobile / SPLAT / CloudRF). A drawn polygon is
        # still respected as an explicit user boundary.
        if ring is not None:
            mask = polygon_mask(lat_m, lon_m, ring)
            if mask is not None:
                rgba[..., 3][~mask] = 0
        rgba[..., 3][rx_dbm < float(coverage_threshold_dbm)] = 0
    else:
        rgba = apply_contour_mask(rgba, lat_m, lon_m, dist, range_km, ring)
    rgba = smooth_upscale(rgba, factor=5)
    rgba = cap_overlay_size(rgba, MAX_OVERLAY_SIDE)
    cbar_label = "Signal [dBm]" if model == "LongleyRice" else "Path Loss [dB]"
    cbar = colorbar_image(cmap_name, vmin, vmax, label=cbar_label)
    return rgba, cbar, out_bounds


@st.cache_data(show_spinner="Mengambil data elevasi...", max_entries=32)
def compute_elevation_assets(location, range_km, n, cmap_name, bounds, ring):
    """Elevation (m) for the grid from local SRTM tiles; build RGBA + colorbar.

    Uses the offline SRTM sampler (tiles cached under ~/.cache/rf_propagation).
    The raster is masked to the drawn contour (or coverage circle) so it follows
    the terrain precisely instead of rendering as a square box.
    """
    lat_m, lon_m, dist, out_bounds = coverage_grid(location, range_km, n, bounds=bounds)
    elev = sample_elevation(lat_m, lon_m, bounds=out_bounds)
    if np.all(np.isnan(elev)):
        raise RuntimeError("Tidak ada data elevasi SRTM untuk area ini (coba pastikan tile sudah diunduh).")
    elev = np.where(np.isnan(elev), 0.0, elev)  # ocean / missing cells = sea level
    vmin, vmax = nice_limits(float(elev.min()), float(elev.max()))
    rgba = colorize(elev, cmap_name, vmin, vmax)
    rgba = apply_contour_mask(rgba, lat_m, lon_m, dist, range_km, ring)
    cbar = colorbar_image(cmap_name, vmin, vmax, label="Elevation [m]")
    return rgba, cbar, out_bounds


def render_heatmap(map_obj, location, range_km, n, cmap_name, opacity, model, freq,
                   rain_rate, fog_density, terrain_type, environment,
                   bounds=None, ring=None, heatmap_data="Elevation",
                   tx_height=TX_HEIGHT_DEFAULT, rx_height=RX_HEIGHT_DEFAULT,
                   reliability=ITM_RELIABILITY_DEFAULT,
                   confidence=ITM_CONFIDENCE_DEFAULT,
                   tx_power_dbm=TX_POWER_DBM_DEFAULT, ipol=ITM_POLARIZATION_DEFAULT,
                   coverage_threshold_dbm=ITM_COVERAGE_THRESHOLD_DBM_DEFAULT):
    """Compute the raster (path loss or elevation) and add heatmap + colorbar."""
    if ring is not None and bounds is None:
        bounds = ring_bounds(ring)
    if model == "LongleyRice" and heatmap_data != "Elevation":
        max_side = int(MAX_ITM_GRID_CELLS ** 0.5)
        if n > max_side:
            st.info(f"Grid di-clamp {n} × {n} → {max_side} × {max_side} "
                    f"untuk LongleyRice (satu ITM p2p per sel).")
            n = max_side
    loc_key = tuple(location)
    bounds_key = tuple(tuple(pt) for pt in bounds) if bounds else None
    ring_key = tuple(tuple(pt) for pt in ring) if ring else None
    if heatmap_data == "Elevation":
        rgba, cbar, out_bounds = compute_elevation_assets(
            loc_key, float(range_km), int(n), cmap_name, bounds_key, ring_key,
        )
    else:
        rgba, cbar, out_bounds = compute_heatmap_assets(
            loc_key, float(range_km), int(n), cmap_name, model, float(freq),
            float(rain_rate), float(fog_density), terrain_type, environment,
            bounds_key, ring_key, float(tx_height), float(rx_height),
            float(reliability), float(confidence), float(tx_power_dbm),
            int(ipol), float(coverage_threshold_dbm),
        )
    map_obj = add_coverage_raster(map_obj, rgba, out_bounds, opacity)
    map_obj = add_colorbar_overlay(map_obj, cbar, out_bounds)
    return map_obj