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
from math import log, tan

import numpy as np
import requests

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

SRTM_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
SRTM_ZOOM = 12
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


def ensure_tiles(tiles, zoom=SRTM_ZOOM):
    """Download any missing tiles once; return {tile: path or None}.

    None means "no data" (e.g. an ocean tile that does not exist upstream); an
    empty file is used as the on-disk sentinel so we never re-request it.
    """
    result = {}
    for x, y in tiles:
        path = _tile_path(x, y, zoom)
        if os.path.exists(path):
            result[(x, y)] = path if os.path.getsize(path) > 0 else None
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = SRTM_BASE_URL.format(z=zoom, x=x, y=y)
        tmp = path + ".part"
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as resp:
                if resp.status_code == 404:
                    open(path, "wb").close()
                    result[(x, y)] = None
                    continue
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp.raw, fh)
            os.replace(tmp, path)
            result[(x, y)] = path
        except requests.RequestException as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(f"Gagal mengunduh tile SRTM {zoom}/{x}/{y}: {exc}") from exc
    return result


def _mercator(lons, lats):
    """Convert lon/lat arrays to web-mercator x/y (matches the SRTM tile CRS)."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    k = 20037508.342789244
    x = lons * k / 180.0
    y = np.log(np.tan((90.0 + lats) * np.pi / 360.0)) * k / np.pi
    return x, y


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
            cols = np.floor((mx[sel] - c) / a).astype(int)
            rows = np.floor((my[sel] - f) / e).astype(int)
            valid = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
            if not valid.any():
                continue
            data = ds.read(1)
            out.ravel()[np.where(sel)[0][valid]] = data[rows[valid], cols[valid]]
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
