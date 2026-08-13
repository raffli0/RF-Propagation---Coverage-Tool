"""Terrain elevation (m) for coordinate grids, offline-first.

Primary source: SRTM-derived Mapzen terrain tiles (EPSG:3857 GeoTIFFs) served
from the public AWS elevation-tiles-prod bucket and cached locally under
~/.cache/rf_propagation/srtm. Tiles are downloaded once; afterwards everything
works fully offline with no API key and no rate limits.

Legacy fallback: the Open-Meteo elevation API (kept for reference/emergency).
"""

import os
import shutil
import time
import concurrent.futures
from math import log, tan

import numpy as np
import requests

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

SRTM_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
# Zoom 13 ~= 1 arc-second (SRTM1, ~30 m/pixel) so terrain undulations feeding
# the Longley-Rice engine are not coarse/blocky. Zoom 12 (3 arc-second) is the
# old fallback; tiles are downloaded once and cached, so this only costs the
# first fetch.
SRTM_ZOOM = 13
SRTM_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "rf_propagation", "srtm")
TIMEOUT_SECONDS = 60

# --- Legacy Open-Meteo API settings (fallback only) ---
ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
CHUNK_SIZE = 100
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.5
INTER_CHUNK_DELAY_SECONDS = 0.4


# ---------------------------------------------------------------------------
# Offline SRTM pipeline
# ---------------------------------------------------------------------------

def tile_xy(lat, lon, zoom):
    """Web-mercator tile (x, y) covering a lat/lon point at `zoom`."""
    lat_rad = np.radians(lat)
    x = int(np.floor((lon + 180.0) / 360.0 * (2 ** zoom)))
    y = int(np.floor(
        (1.0 - np.log(tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * (2 ** zoom)
    ))
    return x, y


def tiles_for_bounds(bounds, zoom=SRTM_ZOOM):
    """List of (x, y) tiles covering the [[lat_min, lon_min], [lat_max, lon_max]] box."""
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    x_min, _ = tile_xy(lat_max, lon_min, zoom)
    x_max, _ = tile_xy(lat_max, lon_max, zoom)
    _, y_max = tile_xy(lat_min, lon_min, zoom)
    _, y_min = tile_xy(lat_max, lon_min, zoom)
    xs = range(min(x_min, x_max), max(x_min, x_max) + 1)
    ys = range(min(y_min, y_max), max(y_min, y_max) + 1)
    return [(x, y) for x in xs for y in ys]


def _tile_path(x, y, zoom):
    return os.path.join(SRTM_CACHE_DIR, str(zoom), str(x), f"{y}.tif")


def _ensure_one(tile, zoom, timeout):
    """Download a single SRTM tile (or reuse the cached copy / 404 sentinel)."""
    x, y = tile
    path = _tile_path(x, y, zoom)
    if os.path.exists(path):
        return path if os.path.getsize(path) > 0 else None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = SRTM_BASE_URL.format(z=zoom, x=x, y=y)
    tmp = path + ".part"
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code == 404:
                open(path, "wb").close()
                return None
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                shutil.copyfileobj(resp.raw, fh)
        os.replace(tmp, path)
        return path
    except requests.RequestException as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Gagal mengunduh tile SRTM {zoom}/{x}/{y}: {exc}") from exc


def ensure_tiles(tiles, zoom=SRTM_ZOOM, max_workers=8):
    """Download any missing tiles (in parallel) once; return {tile: path or None}.

    None means "no data" (e.g. an ocean tile that does not exist upstream); an
    empty file is used as the on-disk sentinel so we never re-request it.
    Parallel fetches (ThreadPoolExecutor) cut the first-run download time; each
    tile has a unique ``*.part`` so there is no cross-tile race.
    """
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_ensure_one, t, zoom, TIMEOUT_SECONDS): t for t in tiles}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                result[t] = fut.result()
            except RuntimeError:
                raise
    return result


def _mercator(lons, lats):
    """Convert lon/lat arrays to web-mercator x/y (matches the SRTM tile CRS)."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    k = 20037508.342789244
    x = lons * k / 180.0
    y = np.log(np.tan((90.0 + lats) * np.pi / 360.0)) * k / np.pi
    return x, y


def sample_point_elevation(lat, lon):
    """Ground elevation (m) at a single coordinate from cached SRTM tiles.

    Returns NaN when the tile is missing/unavailable (e.g. first run before
    the offline tile download, or open ocean). Slightly widens the lookup
    window so the bilinear sampler has a 2x2 pixel footprint.
    """
    lat = float(lat)
    lon = float(lon)
    lat_m = np.array([[lat - 1e-4, lat, lat + 1e-4],
                      [lat - 1e-4, lat, lat + 1e-4],
                      [lat - 1e-4, lat, lat + 1e-4]])
    lon_m = np.array([[lon - 1e-4, lon, lon + 1e-4],
                      [lon - 1e-4, lon, lon + 1e-4],
                      [lon - 1e-4, lon, lon + 1e-4]])
    grid = sample_elevation(lat_m, lon_m)
    center = grid[1, 1]
    if np.isnan(center):
        finite = grid[np.isfinite(grid)]
        return float(finite.mean()) if finite.size else float("nan")
    return float(center)


def sample_elevation(lat_m, lon_m, bounds=None):
    """Elevation (m) per grid cell from cached SRTM tiles; NaN where no data.

    `lat_m`/`lon_m` are the 2-D mesh arrays produced by `coverage_grid`.
    Missing cells (ocean tiles, tile edges) come back as NaN.
    """
    if rasterio is None:
        raise RuntimeError("rasterio tidak terinstall; install `pip install rasterio`.")
    lat_m = np.asarray(lat_m, dtype=float)
    lon_m = np.asarray(lon_m, dtype=float)
    if bounds is None:
        bounds = [[float(lat_m.min()), float(lon_m.min())],
                  [float(lat_m.max()), float(lon_m.max())]]
    out = np.full(lat_m.shape, np.nan, dtype=float)
    tiles = tiles_for_bounds(bounds, SRTM_ZOOM)
    paths = ensure_tiles(tiles, SRTM_ZOOM)
    if not any(path is not None for path in paths.values()):
        raise RuntimeError(
            "Tidak ada data elevasi SRTM untuk area ini. "
            "Pastikan koneksi internet tersedia sekali untuk mengunduh tile."
        )
    mx, my = _mercator(lon_m.ravel(), lat_m.ravel())
    for (x, y), path in paths.items():
        if path is None:
            continue
        with rasterio.open(path) as ds:
            a, _, c, _, e, f = (ds.transform[i] for i in range(6))
            left, bottom, right, top = ds.bounds
            sel = (mx >= left) & (mx <= right) & (my >= bottom) & (my <= top)
            if not sel.any():
                continue
            data = ds.read(1)
            xf = (mx[sel] - c) / a
            yf = (my[sel] - f) / e
            out.ravel()[np.where(sel)[0]] = _bilinear_sample(data, xf, yf)
    return out


def _bilinear_sample(data, xf, yf):
    """Bilinear elevation interpolation at fractional (column, row) coords.

    `xf`/`yf` are the fractional pixel coordinates of the sample points.
    Points whose 2x2 window falls outside the band (including a 1-px border)
    return NaN, preserving ``sample_elevation``'s missing-data contract
    (ocean tiles, tile edges). Bilinear resampling keeps neighbouring coverage
    pixels continuous instead of the blocky step of nearest-neighbour lookup.
    """
    xf = np.asarray(xf, dtype=float)
    yf = np.asarray(yf, dtype=float)
    x0 = np.floor(xf).astype(int)
    y0 = np.floor(yf).astype(int)
    wx = xf - x0
    wy = yf - y0
    h, w = data.shape
    ok = (y0 >= 0) & (y0 + 1 < h) & (x0 >= 0) & (x0 + 1 < w)
    out = np.full(xf.shape, np.nan, dtype=float)
    if not ok.any():
        return out
    y0k, x0k = y0[ok], x0[ok]
    wxk, wyk = wx[ok], wy[ok]
    v00 = data[y0k, x0k]
    v01 = data[y0k, x0k + 1]
    v10 = data[y0k + 1, x0k]
    v11 = data[y0k + 1, x0k + 1]
    out[ok] = ((1 - wyk) * ((1 - wxk) * v00 + wxk * v01)
               + wyk * ((1 - wxk) * v10 + wxk * v11))
    return out


# ---------------------------------------------------------------------------
# Legacy Open-Meteo API fallback
# ---------------------------------------------------------------------------

def _request_chunk(chunk):
    """Fetch elevation for one chunk, retrying on 429/5xx with backoff."""
    lats = ",".join(f"{p[0]:.6f}" for p in chunk)
    lons = ",".join(f"{p[1]:.6f}" for p in chunk)
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                ELEVATION_API,
                params={"latitude": lats, "longitude": lons},
                timeout=TIMEOUT_SECONDS,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = RuntimeError(f"HTTP {resp.status_code} (rate limit / server)")
            else:
                resp.raise_for_status()
                return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
        if attempt < MAX_RETRIES - 1:
            time.sleep(BASE_BACKOFF_SECONDS * (2 ** attempt))
    raise RuntimeError(f"Gagal mengambil data elevasi dari Open-Meteo: {last_exc}")


def fetch_elevation(points):
    """Legacy API path: elevation (m) for an (N, 2) array of (lat, lon) points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    n = points.shape[0]
    elevations = np.empty(n, dtype=float)
    for i, start in enumerate(range(0, n, CHUNK_SIZE)):
        chunk = points[start:start + CHUNK_SIZE]
        data = _request_chunk(chunk)
        values = data.get("elevation")
        if not isinstance(values, list) or len(values) != len(chunk):
            raise RuntimeError("Respon Open-Meteo tidak valid (jumlah koordinat tidak cocok).")
        elevations[start:start + len(chunk)] = values
        if start + CHUNK_SIZE < n and i < (n - 1) // CHUNK_SIZE:
            time.sleep(INTER_CHUNK_DELAY_SECONDS)
    return elevations
