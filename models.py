import math

import numpy as np

from itm_adapter import run_longley_rice

# ---------------------------------------------------------------------------
# ITU-R P.838-3 rain attenuation coefficients (k, alpha) vs frequency (GHz).
# Interpolated in log-frequency; gamma_R = k * R^alpha (dB/km).
# ---------------------------------------------------------------------------
RAIN_COEFFS = np.array([
    (1.0, 0.0000356, 0.912),
    (4.0, 0.000594, 1.075),
    (10.0, 0.01217, 1.154),
    (15.0, 0.03588, 1.128),
    (20.0, 0.0751, 1.099),
    (30.0, 0.187, 1.021),
    (40.0, 0.3502, 0.939),
])

### 1. Free Space Propagation Model ###
def free_space_path_loss(frequency, distance):
    """
    Calculate the Free Space Path Loss (FSPL) in dB.
    
    :param frequency: Frequency in MHz
    :param distance: Distance in km
    :return: Path loss in dB
    """
    distance = np.maximum(distance, 1e-9)
    # FSPL (dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.44
    fspl = 20 * np.log10(distance) + 20 * np.log10(frequency) + 32.44
    return fspl

### 2. Rain Propagation Model ###
def rain_attenuation(frequency, distance, rain_rate):
    """
    Calculate the rain attenuation in dB.

    :param frequency: Frequency in GHz
    :param distance: Distance in km
    :param rain_rate: Rainfall rate in mm/h
    :return: Rain attenuation in dB
    """
    frequency = float(np.maximum(frequency, 1e-9))
    f_ghz = RAIN_COEFFS[:, 0]
    k, alpha = np.interp(np.log(frequency), np.log(f_ghz), RAIN_COEFFS[:, 1]), \
               np.interp(np.log(frequency), np.log(f_ghz), RAIN_COEFFS[:, 2])
    specific_attenuation = k * rain_rate ** alpha
    return specific_attenuation * distance

### 3. Gas Propagation Model ###
def gas_attenuation(frequency, distance):
    """
    Calculate the gas attenuation in dB/km.
    
    :param frequency: Frequency in GHz
    :param distance: Distance in km
    :return: Gas attenuation in dB
    """
    # Simplified model assuming dry air at standard temperature and pressure
    specific_attenuation = 0.05 * frequency  # Attenuation factor for dry air
    gas_loss = specific_attenuation * distance
    return gas_loss


def _fog_specific_attenuation(frequency_ghz):
    """K_l (dB/km per g/m³ of liquid water) from the ITU-R P.840 double-Debye model."""
    f = float(np.maximum(frequency_ghz, 1e-9))
    eps0, eps1, eps2 = 77.66, 5.48, 3.51
    f0, f1 = 20.09, 505.84
    e_real = (eps0 - eps1) / (1 + (f / f0) ** 2) + (eps1 - eps2) / (1 + (f / f1) ** 2) + eps2
    e_imag = f * (eps0 - eps1) / (f0 * (1 + (f / f0) ** 2)) + f * (eps1 - eps2) / (f1 * (1 + (f / f1) ** 2))
    eta = (2 + e_real) / e_imag
    return 0.819 * f / (e_imag * (1 + eta ** 2))


### 4. Fog Propagation Model ###
def fog_attenuation(frequency, distance, fog_density):
    """
    Calculate the fog attenuation in dB (ITU-R P.840 specific attenuation).

    :param frequency: Frequency in GHz
    :param distance: Distance in km
    :param fog_density: Fog density in g/m³
    :return: Fog attenuation in dB
    """
    specific_attenuation = _fog_specific_attenuation(frequency) * fog_density
    return specific_attenuation * distance

### 5. Close-In Propagation Model ###
def close_in_path_loss(frequency, distance, reference_distance=1):
    """
    Calculate the Close-In (CI) Path Loss model in dB.
    
    :param frequency: Frequency in MHz
    :param distance: Distance in km
    :param reference_distance: Reference distance in meters (default is 1 meter)
    :return: Path loss in dB
    """
    # CI model: PL(d) = FSPL(d0) + 10*n*log10(d/d0); d0 in meters, d in km
    distance_m = np.maximum(distance, 1e-9) * 1e3
    path_loss_exponent = 2  # Free-space path loss exponent
    fspl_ref = 20 * np.log10(reference_distance) + 20 * np.log10(frequency) - 27.55
    close_in_loss = fspl_ref + 10 * path_loss_exponent * np.log10(distance_m / reference_distance)
    return close_in_loss

### 6. Longley-Rice Propagation Model ###
def longley_rice_path_loss(frequency, distance_km, tx_height, rx_height, profile_m,
                           ipol=0, reliability=50, confidence=50,
                           eps=None, sgm=None, klim=None, diffraction="ITM"):
    """
    Longley-Rice (ITM) point-to-point path loss in dB via itmlogic.

    Delegates to :func:`itm_adapter.run_longley_rice`, which translates these
    inputs into itmlogic's ``prop`` structure. The ITM algorithm itself lives
    in the installed ``itmlogic`` package and is never modified here.

    :param frequency: Frequency in MHz
    :param distance_km: Link distance in km
    :param tx_height: Transmitter antenna height above ground (m)
    :param rx_height: Receiver antenna height above ground (m)
    :param profile_m: Ground elevation (m) sampled along the tx->rx path
    :param ipol: Polarization (0=horizontal, 1=vertical)
    :param reliability: Reliability level in percent
    :param confidence: Confidence level in percent
    :param eps: Terrain relative permittivity (None -> p2p.py default)
    :param sgm: Terrain conductivity in S/m (None -> p2p.py default)
    :param klim: ITM climate zone (None -> p2p.py default)
    :param diffraction: Terrain handling mode. "ITM" (default) runs the full
        Longley-Rice engine; "Knife-edge" uses the single knife-edge diffraction
        model; "Off (LOS)" returns pure free-space loss (no terrain diffraction).
    :return: Path loss in dB
    """
    if diffraction == "Off (LOS)":
        return free_space_path_loss(frequency, distance_km)
    if diffraction == "Knife-edge":
        return knife_edge_path_loss(frequency, distance_km, tx_height, rx_height, profile_m)
    env = {}
    if eps is not None:
        env["eps"] = eps
    if sgm is not None:
        env["sgm"] = sgm
    if klim is not None:
        env["klim"] = klim
    return run_longley_rice(
        frequency, tx_height, rx_height, profile_m, distance_km,
        ipol=ipol, reliability=reliability, confidence=confidence, **env,
    )


### 6b. Knife-Edge Diffraction Propagation Model ###
def knife_edge_path_loss(frequency, distance_km, tx_height, rx_height, profile_m):
    """Single knife-edge diffraction path loss (dB).

    Treats the highest terrain point above the TX->RX line-of-sight as a knife
    edge and adds the classic Fresnel-Kirchhoff diffraction attenuation to the
    free-space loss. When the path is clear (no obstacle above LOS) it evaluates
    to free-space loss only, so it degrades gracefully to LOS.

    :param frequency: Frequency in MHz
    :param distance_km: Link distance in km
    :param tx_height: Transmitter antenna height above ground (m)
    :param rx_height: Receiver antenna height above ground (m)
    :param profile_m: Ground elevation (m) sampled along the tx->rx path
    :return: Path loss in dB
    """
    profile = [float(x) for x in profile_m]
    if len(profile) < 2 or distance_km <= 0 or frequency <= 0:
        return float(free_space_path_loss(frequency, distance_km))
    lam = 299792458.0 / (float(frequency) * 1e6)          # wavelength (m)
    h_tx = profile[0] + float(tx_height)
    h_rx = profile[-1] + float(rx_height)
    n = len(profile)
    d_m = np.linspace(0.0, float(distance_km) * 1000.0, n)  # metres along path
    los = np.full(n, h_tx) if d_m[-1] <= 0 else h_tx + (h_rx - h_tx) * (d_m / d_m[-1])
    abs_h = np.array(profile, dtype=float)
    abs_h[0] = h_tx
    abs_h[-1] = h_rx
    clearance = abs_h - los                                     # +ve = above LOS
    interior = clearance[1:-1]
    if interior.size == 0:
        h_obs, d_obs = 0.0, d_m[-1] / 2.0
    else:
        k = int(np.argmax(interior))
        h_obs = float(interior[k])
        d_obs = float(d_m[1 + k])
    fs = float(free_space_path_loss(frequency, distance_km))
    if h_obs <= 0:
        return fs
    d1 = max(d_obs, 1.0)
    d2 = max(d_m[-1] - d_obs, 1.0)
    v = h_obs * math.sqrt(2.0 * (d1 + d2) / (lam * d1 * d2))
    att = 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return float(fs + att)


### 7. TIREM Propagation Model ###
def tirem_path_loss(frequency, distance, terrain_type="average"):
    """
    Estimate the TIREM path loss.
    
    :param frequency: Frequency in MHz
    :param distance: Distance in km
    :param terrain_type: Terrain type (default is "average")
    :return: Path loss in dB
    """
    # Placeholder model - real TIREM is more complex and requires specific data
    terrain_factor = {"average": 0.8, "hilly": 1.2, "mountainous": 1.5}
    factor = terrain_factor.get(terrain_type, 0.8)
    distance = np.maximum(distance, 1e-9)
    tirem_loss = factor * (20 * np.log10(frequency) + 20 * np.log10(distance) + 0.1)
    return tirem_loss

### 8. Ray Tracing Propagation Model ###
def ray_tracing_path_loss(frequency, distance, environment="urban"):
    """
    Estimate the Ray Tracing path loss.
    
    :param frequency: Frequency in MHz
    :param distance: Distance in km
    :param environment: Environment type ("urban", "rural", etc.)
    :return: Path loss in dB
    """
    # Simplified ray tracing model with basic reflection and diffraction losses
    env_factor = {"urban": 3.0, "rural": 1.5}
    factor = env_factor.get(environment, 3.0)
    distance = np.maximum(distance, 1e-9)
    ray_tracing_loss = factor * (20 * np.log10(frequency) + 20 * np.log10(distance) + 32.44)
    return ray_tracing_loss


### Model dispatch ###
def get_model_fn(model, freq, rain_rate=None, fog_density=None, terrain_type="average",
                 environment="urban", tx_height=30.0, rx_height=1.5,
                 longley_rice_profile=None,
                 longley_rice_reliability=50.0, longley_rice_confidence=50.0,
                 longley_rice_polarization=0, longley_rice_eps=None,
                 longley_rice_sgm=None, longley_rice_klim=None,
                 longley_rice_diffraction="ITM"):
    """Return a callable f(distance_km) -> path_loss_dB for the selected model.

    Model-specific params are passed explicitly so the result is deterministic
    and safe to cache (scalable when grids grow large).

    LongleyRice is point-to-point and needs a terrain profile along the path:
    pass one via `longley_rice_profile` (list of elevation samples in meters),
    or the returned callable raises RuntimeError when invoked.
    """
    if model == "Free Space":
        return lambda d: free_space_path_loss(freq, d)
    if model == "Rain":
        return lambda d: rain_attenuation(freq / 1000, d, rain_rate)
    if model == "Gas":
        return lambda d: gas_attenuation(freq / 1000, d)
    if model == "Fog":
        return lambda d: fog_attenuation(freq / 1000, d, fog_density)
    if model == "CloseIn":
        return lambda d: close_in_path_loss(freq, d)
    if model == "LongleyRice":
        def _lr(d):
            if longley_rice_profile is None:
                raise RuntimeError(
                    "LongleyRice memerlukan terrain profile. "
                    "Gunakan terrain_profile() / render_heatmap per-cell."
                )
            return longley_rice_path_loss(
                freq, d, tx_height, rx_height, longley_rice_profile,
                ipol=longley_rice_polarization,
                reliability=longley_rice_reliability,
                confidence=longley_rice_confidence,
                eps=longley_rice_eps, sgm=longley_rice_sgm,
                klim=longley_rice_klim,
                diffraction=longley_rice_diffraction,
            )
        return _lr
    if model == "TIREM":
        return lambda d: tirem_path_loss(freq, d, terrain_type)
    return lambda d: ray_tracing_path_loss(freq, d, environment)
