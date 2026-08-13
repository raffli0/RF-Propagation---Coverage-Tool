"""ITM adapter: translates UI inputs into itmlogic's `prop` structure.

This is the only bridge between the coverage tool and the itmlogic library.
The ITM machinery itself (src/itmlogic) is never modified — this module
replicates the exact point-to-point call sequence used by
`itmlogic/scripts/p2p.py::itmlogic_p2p`:

    qlrpfl(prop)  ->  fs = db * ln(2 * wn * dist)  ->  avar(...) -> loss dB

Reference anchor (from itmlogic's own test suite, tests/test_p2p.py):
    fmhz=41.5, hg=[143.9, 8.5], profile (157 pts), d=77.8 km,
    reliability=1%, confidence=50%  =>  128.5969039310673 dB
"""

import math

import numpy as np

from itmlogic.misc.qerfi import qerfi
from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
from itmlogic.statistics.avar import avar

# Conversion factor Neper -> dB
_DB = 8.685890

# Defaults matching scripts/p2p.py (continental temperate overland climate)
_DEFAULT_ENV = {
    "eps": 15.0,      # terrain relative permittivity
    "sgm": 0.005,     # terrain conductivity (S/m)
    "klim": 5,        # climate zone (5 = continental temperate)
    "ens0": 314.0,    # surface refractivity (N-units)
    "gma": 157e-9,    # inverse Earth radius
    "zsys": 0.0,      # average system elevation above sea level (m)
}


def run_longley_rice(
    frequency,
    tx_height,
    rx_height,
    terrain_profile,
    distance_km,
    ipol=0,
    reliability=50,
    confidence=50,
    **env,
):
    """Estimate Longley-Rice point-to-point path loss (dB) for one path.

    Parameters
    ----------
    frequency : float
        Operating frequency in MHz.
    tx_height : float
        Transmitter antenna height above ground (m).
    rx_height : float
        Receiver antenna height above ground (m).
    terrain_profile : list of float
        Ground elevation (m) sampled along the path from transmitter to
        receiver. At least 2 points; spacing is derived from `distance_km`.
    distance_km : float
        Great-circle distance between the terminals (km).
    ipol : int
        Polarization: 0 = horizontal, 1 = vertical.
    reliability : float
        Reliability level requested (percent, e.g. 50).
    confidence : float
        Confidence level requested (percent, e.g. 50).
    raise_on_error : bool
        If True, raise RuntimeError when itmlogic flags the input geometry
        as out of the model's valid range (prop['kwx'] != 0). Default False:
        the loss is still returned, matching scripts/p2p.py which only
        reports kwx as a warning and uses the computed attenuation.
    **env : optional
        Overrides for environmental defaults (eps, sgm, klim, ens0, gma,
        zsys). Leave unset for the reference p2p.py values.

    Returns
    -------
    float
        Basic transmission loss in dB at the requested reliability and
        confidence levels.

    Raises
    ------
    ValueError
        If the profile is too short or the distance is non-positive.
    RuntimeError
        Only when `raise_on_error` is set and itmlogic reports an internal
        error (prop['kwx'] != 0).
    """
    profile = [float(x) for x in terrain_profile]
    if len(profile) < 2:
        raise ValueError("terrain_profile must contain at least 2 points")
    if distance_km <= 0:
        raise ValueError("distance_km must be positive")

    raise_on_error = bool(env.pop("raise_on_error", False))

    settings = dict(_DEFAULT_ENV)
    settings.update(env)

    prop = {
        "fmhz": float(frequency),
        "hg": [float(tx_height), float(rx_height)],
        "d": float(distance_km),
        "ipol": int(ipol),
        "eps": float(settings["eps"]),
        "sgm": float(settings["sgm"]),
        "klim": int(settings["klim"]),
        "ens0": float(settings["ens0"]),
        "lvar": 5,
        "gma": float(settings["gma"]),
        "klimx": 0,
        "mdvarx": 11,
    }

    # Number of points describing the profile minus one
    pfl = [len(profile) - 1, 0]
    pfl.extend(profile)

    # Inverse Earth radius in the prop namespace
    prop["gma"] = float(settings["gma"])

    # Refractivity scaling ens = ens0 * exp(-zsys / 9460)
    zsys = float(settings["zsys"])

    # Distance of the link in km
    dkm = prop["d"]

    # If DKM set <=0, derive it from the profile step times point count
    xkm = 0
    if dkm <= 0:
        dkm = xkm * pfl[0]

    # If XKM <=0, derive the range step from the profile length / point count
    if xkm <= 0:
        xkm = dkm // pfl[0]
        # Range step in meters stored in PFL(2)
        pfl[1] = dkm * 1000 / pfl[0]
        prop["pfl"] = pfl
        # Zero out the error flag
        prop["kwx"] = 0
        # Initialize the omega_n quantity
        prop["wn"] = prop["fmhz"] / 47.7
        # Initialize refractive index properties
        prop["ens"] = prop["ens0"]

    if zsys != 0:
        prop["ens"] = prop["ens"] * math.exp(-zsys / 9460)

    # Include refraction in the effective Earth curvature parameter
    prop["gme"] = prop["gma"] * (1 - 0.04665 * math.exp(prop["ens"] / 179.3))

    # Surface impedance Zq parameter
    zq = complex(prop["eps"], 376.62 * prop["sgm"] / prop["wn"])

    # Z parameter (horizontal polarization)
    prop["zgnd"] = np.sqrt(zq - 1)

    # Z parameter (vertical polarization)
    if prop["ipol"] != 0:
        prop["zgnd"] = prop["zgnd"] / zq

    # Convert requested reliability/confidence levels into standard normal
    # distribution arguments (percent -> fraction -> quantile)
    zr = qerfi([reliability / 100.0])[0]
    zc = qerfi([confidence / 100.0])[0]

    # Initialization routine for point-to-point mode
    prop = qlrpfl(prop)

    if prop["kwx"] != 0 and raise_on_error:
        raise RuntimeError(
            f"itmlogic error: kwx={prop['kwx']} "
            f"(f={frequency} MHz, d={distance_km} km)"
        )

    # Free space loss in dB
    fs = _DB * np.log(2 * prop["wn"] * prop["dist"])

    # Correction for the requested reliability/confidence levels
    avar1, prop = avar(zr, 0, zc, prop)

    return float(fs + avar1)


def _great_circle_points(lat1, lon1, lat2, lon2, n):
    """Equidistant (lat, lon) samples along the great circle A -> B."""
    p1 = np.array([
        np.cos(np.radians(lat1)) * np.cos(np.radians(lon1)),
        np.cos(np.radians(lat1)) * np.sin(np.radians(lon1)),
        np.sin(np.radians(lat1)),
    ])
    p2 = np.array([
        np.cos(np.radians(lat2)) * np.cos(np.radians(lon2)),
        np.cos(np.radians(lat2)) * np.sin(np.radians(lon2)),
        np.sin(np.radians(lat2)),
    ])
    a = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    if a < 1e-9:
        return np.full(n, float(lat1)), np.full(n, float(lon1))
    ts = np.linspace(0.0, 1.0, n)
    sa = np.sin(a)
    p = (
        np.sin((1.0 - ts)[:, None] * a) * p1[None, :]
        + np.sin(ts[:, None] * a) * p2[None, :]
    ) / sa
    return np.degrees(np.arcsin(p[:, 2])), np.degrees(np.arctan2(p[:, 1], p[:, 0]))


def terrain_profile(tx_lat, tx_lon, lat, lon, n_points=50):
    """Ground elevation (m) along the great-circle path tx -> (lat, lon).

    Samples the offline SRTM tiles through `elevation.sample_elevation`;
    missing cells (ocean / tile gaps) are reported as 0 m. Tiles download
    once and are cached under ~/.cache/rf_propagation/srtm.
    """
    from elevation import sample_elevation

    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    lats, lons = _great_circle_points(tx_lat, tx_lon, lat, lon, n_points)
    lat_m = lats.reshape(1, -1)
    lon_m = lons.reshape(1, -1)
    elev = sample_elevation(lat_m, lon_m)
    return [float(v) for v in np.nan_to_num(elev, nan=0.0).ravel()]