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
                           ipol=0, reliability=50, confidence=50):
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
    :return: Path loss in dB
    """
    return run_longley_rice(
        frequency, tx_height, rx_height, profile_m, distance_km,
        ipol=ipol, reliability=reliability, confidence=confidence,
    )

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
                 longley_rice_polarization=0):
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
            )
        return _lr
    if model == "TIREM":
        return lambda d: tirem_path_loss(freq, d, terrain_type)
    return lambda d: ray_tracing_path_loss(freq, d, environment)
