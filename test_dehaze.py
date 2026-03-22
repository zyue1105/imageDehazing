"""Tests for the dark channel prior dehazing algorithm."""

import numpy as np
import pytest

from dehaze import (
    dehaze,
    get_airlight,
    get_dark_channel,
    get_transmission,
    recover,
)


def _make_hazy_image(h=100, w=150):
    """Create a synthetic hazy image (float64, 0-1 range, 3 channels)."""
    rng = np.random.RandomState(42)
    return rng.uniform(0.4, 1.0, (h, w, 3))


class TestGetDarkChannel:
    def test_output_shape(self):
        img = _make_hazy_image()
        dark = get_dark_channel(img, radius=7)
        assert dark.shape == (100, 150)

    def test_values_in_range(self):
        img = _make_hazy_image()
        dark = get_dark_channel(img, radius=7)
        assert dark.min() >= 0.0
        assert dark.max() <= 1.0

    def test_dark_channel_leq_min_channel(self):
        """Dark channel should be <= min across RGB at every pixel."""
        img = _make_hazy_image()
        dark = get_dark_channel(img, radius=1)
        min_channel = np.min(img, axis=2)
        assert np.all(dark <= min_channel + 1e-10)

    def test_uniform_image(self):
        """For a uniform image, dark channel should be constant."""
        img = np.full((50, 50, 3), 0.5)
        dark = get_dark_channel(img, radius=3)
        np.testing.assert_allclose(dark, 0.5, atol=1e-10)


class TestGetAirlight:
    def test_output_shape(self):
        img = _make_hazy_image()
        dark = get_dark_channel(img)
        A = get_airlight(img, dark)
        assert A.shape == (3,)

    def test_airlight_within_image_range(self):
        img = _make_hazy_image()
        dark = get_dark_channel(img)
        A = get_airlight(img, dark)
        assert np.all(A >= 0.0)
        assert np.all(A <= 1.0)


class TestGetTransmission:
    def test_output_shape(self):
        img = _make_hazy_image()
        A = np.array([0.9, 0.9, 0.9])
        t = get_transmission(img, A, omega=0.95)
        assert t.shape == (100, 150)

    def test_omega_zero_gives_all_ones(self):
        """With omega=0, transmission should be 1 everywhere."""
        img = _make_hazy_image()
        A = np.array([0.9, 0.9, 0.9])
        t = get_transmission(img, A, omega=0.0)
        np.testing.assert_allclose(t, 1.0)


class TestRecover:
    def test_output_clipped(self):
        img = _make_hazy_image()
        t = np.full((100, 150), 0.5)
        A = np.array([0.8, 0.8, 0.8])
        result = recover(img, t, A, t0=0.1)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_t0_prevents_division_by_zero(self):
        img = _make_hazy_image()
        t = np.zeros((100, 150))  # all-zero transmission
        A = np.array([0.8, 0.8, 0.8])
        result = recover(img, t, A, t0=0.1)
        assert np.all(np.isfinite(result))


class TestDehaze:
    def test_end_to_end(self):
        """Full pipeline should produce a valid uint8 image."""
        img_uint8 = (_make_hazy_image() * 255).astype(np.uint8)
        result = dehaze(img_uint8)
        assert result.dtype == np.uint8
        assert result.shape == img_uint8.shape

    def test_with_real_image(self):
        """Test with an actual sample image from the repo."""
        import cv2

        img = cv2.imread("results/canon.bmp")
        if img is None:
            pytest.skip("Sample image not available")
        result = dehaze(img)
        assert result.dtype == np.uint8
        assert result.shape == img.shape
        # Dehazed image should generally be darker (less haze glow)
        assert result.mean() < img.mean()
