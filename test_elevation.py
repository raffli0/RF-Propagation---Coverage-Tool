"""Tests for elevation resampling and contour smoothing."""

import numpy as np

from elevation import SRTM_ZOOM, _bilinear_sample
from heatmap import _smooth_field


def test_srtm_zoom_is_1_arc_second():
    assert SRTM_ZOOM == 13


def test_bilinear_sample_exact_on_linear_plane():
    x, y = np.meshgrid(np.arange(20.0), np.arange(20.0))
    data = (100 + 2.0 * x + 3.0 * y).astype("float32")
    got = _bilinear_sample(data, np.array([5.5]), np.array([7.5]))[0]
    assert got == 133.5


def test_bilinear_sample_returns_nan_outside_band():
    data = np.zeros((4, 4), dtype="float32")
    got = _bilinear_sample(data, np.array([3.5]), np.array([3.5]))[0]
    assert np.isnan(got)


def test_smooth_field_zero_is_passthrough():
    f = np.tile(np.array([0.0, 10.0] * 5), (10, 1)).astype("float32")
    assert np.array_equal(_smooth_field(f, 0), f)


def test_smooth_field_preserves_mean_and_reduces_variance():
    f = np.tile(np.array([0.0] * 5 + [10.0] * 5), (10, 1)).astype("float32")
    for level in (1, 2):
        smoothed = _smooth_field(f, level)
        assert smoothed.shape == f.shape
        assert abs(smoothed.mean() - f.mean()) < 1e-3
        assert smoothed.var() < f.var()