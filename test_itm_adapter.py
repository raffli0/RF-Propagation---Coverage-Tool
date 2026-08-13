"""Regression tests for the itmlogic adapter (itm_adapter.py) and its wiring.

The anchor values are taken verbatim from itmlogic's own test suite
(tests/test_p2p.py) so this suite also guards against accidental changes in
the installed itmlogic engine or in the adapter's prop construction.
"""

import pytest

import itm_adapter
from models import get_model_fn, longley_rice_path_loss

# Point-to-point test profile from itmlogic/tests/conftest.py (157 points)
PROFILE = [
    96, 84, 65, 46, 46, 46, 61, 41, 33, 27, 23, 19, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 17, 19, 21, 23, 25, 27, 29, 35, 46, 41, 35, 30, 33, 35,
    37, 40, 35, 30, 51, 62, 76, 46, 46, 46, 46, 46, 46, 50, 56, 67, 106, 83, 95,
    112, 137, 137, 76, 103, 122, 122, 83, 71, 61, 64, 67, 71, 74, 77, 79, 86, 91,
    83, 76, 68, 63, 76, 107, 107, 107, 119, 127, 133, 135, 137, 142, 148, 152,
    152, 107, 137, 104, 91, 99, 120, 152, 152, 137, 168, 168, 122, 137, 137, 170,
    183, 183, 187, 194, 201, 192, 152, 152, 166, 177, 198, 156, 127, 116, 107,
    104, 101, 98, 95, 103, 91, 97, 102, 107, 107, 107, 103, 98, 94, 91, 105, 122,
    122, 122, 122, 122, 137, 137, 137, 137, 137, 137, 137, 137, 140, 144, 147,
    150, 152, 159,
]

# (reliability, confidence, expected dB) from itmlogic/tests/test_p2p.py
ANCHORS = [
    (1, 50, 128.5969039310673),
    (1, 90, 137.64279211442656),
    (1, 10, 119.55101574770802),
    (99, 50, 139.74127375512774),
    (99, 90, 148.4389165313392),
    (99, 10, 131.04363097891627),
]

# Geometry that reliably trips ITM's validity flag (kwx=3): a single 8 km
# spike in an otherwise flat 77.8 km profile.
SPIKE_PROFILE = [0] * 78 + [8000] + [0] * 78


@pytest.mark.parametrize("reliability,confidence,expected", ANCHORS)
def test_anchor_loss_matches_itmlogic(reliability, confidence, expected):
    got = itm_adapter.run_longley_rice(
        41.5, 143.9, 8.5, PROFILE, 77.8,
        reliability=reliability, confidence=confidence,
    )
    assert got == pytest.approx(expected, abs=1e-6)


def test_models_layer_anchor():
    got = longley_rice_path_loss(41.5, 77.8, 143.9, 8.5, PROFILE,
                                 reliability=1, confidence=50)
    assert got == pytest.approx(128.5969039310673, abs=1e-6)


def test_kwx_is_warning_by_default():
    loss = itm_adapter.run_longley_rice(41.5, 1.5, 1.5, SPIKE_PROFILE, 77.8)
    assert isinstance(loss, float)


def test_kwx_strict_raises():
    with pytest.raises(RuntimeError, match="kwx=3"):
        itm_adapter.run_longley_rice(
            41.5, 1.5, 1.5, SPIKE_PROFILE, 77.8, raise_on_error=True
        )


def test_short_profile_rejected():
    with pytest.raises(ValueError, match="terrain_profile"):
        itm_adapter.run_longley_rice(900, 30, 1.5, [100.0], 5.0)


def test_non_positive_distance_rejected():
    with pytest.raises(ValueError, match="distance_km"):
        itm_adapter.run_longley_rice(900, 30, 1.5, [100.0, 120.0], 0.0)


def test_get_model_fn_lr_requires_profile():
    fn = get_model_fn("LongleyRice", 900)
    with pytest.raises(RuntimeError, match="terrain profile"):
        fn(10.0)


def test_get_model_fn_lr_with_profile():
    fn = get_model_fn("LongleyRice", 41.5, tx_height=143.9, rx_height=8.5,
                      longley_rice_profile=PROFILE,
                      longley_rice_reliability=1, longley_rice_confidence=50)
    assert fn(77.8) == pytest.approx(128.5969039310673, abs=1e-6)


def test_reliability_changes_loss():
    base = itm_adapter.run_longley_rice(41.5, 143.9, 8.5, PROFILE, 77.8,
                                        reliability=50, confidence=50)
    strict = itm_adapter.run_longley_rice(41.5, 143.9, 8.5, PROFILE, 77.8,
                                          reliability=90, confidence=90)
    assert strict > base


def test_non_lr_models_unaffected():
    fn = get_model_fn("Free Space", 900)
    assert fn(1.0) > 0
