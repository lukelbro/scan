"""Unit tests for e11scan.fitting — standalone fitters with synthetic data."""
import sys
sys.path.insert(0, '.')

import numpy as np
from pytest import approx
from e11scan.fitting import (
    gaussian_func, double_gaussian_func, rabi_func,
    fit_gaussian, fit_double_gaussian, fit_rabi, FitResult,
)


# ---------- Model function tests -------------------------------------------

def test_gaussian_func():
    x = np.linspace(-3, 3, 100)
    y = gaussian_func(x, A=2.0, mu=0.0, sigma=1.0)
    # Peak should be at x closest to 0
    mid = len(x) // 2
    assert y[mid] == approx(2.0, abs=0.01)  # peak near center
    assert y[0] < 0.1  # tails near zero


def test_double_gaussian_func():
    x = np.linspace(-5, 5, 200)
    y = double_gaussian_func(x, 1.0, -2.0, 0.5, 0.5, 2.0, 0.3)
    # Should have two peaks — check max is significant
    assert np.max(y) > 0.4


def test_rabi_func():
    x = np.linspace(0, 10, 200)
    y = rabi_func(x, omega=2*np.pi*0.5, decay=5.0, a=1.0)
    assert y[0] == approx(0.0, abs=1e-10)  # starts at 0
    assert np.max(y) > 0.5  # should oscillate


# ---------- Gaussian fit tests ---------------------------------------------

def test_fit_gaussian_perfect_data():
    """Fit a clean Gaussian — should recover parameters."""
    x = np.linspace(-5, 5, 200)
    true_A, true_mu, true_sigma = 3.0, 0.5, 1.2
    y = gaussian_func(x, true_A, true_mu, true_sigma)
    # Add tiny noise
    np.random.seed(42)
    y += np.random.normal(0, 0.01, len(y))

    result = fit_gaussian(x, y)
    assert isinstance(result, FitResult)
    assert result.p0[0] == approx(true_A, rel=0.05)
    assert result.p0[1] == approx(true_mu, abs=0.1)
    assert result.p0[2] == approx(true_sigma, rel=0.1)


def test_fit_gaussian_narrow_x_range():
    """The critical regression: narrow x range should NOT blow up sigma guess."""
    # Simulate the actual e11scan scenario: x spans only 0.0013
    x = np.linspace(19.55575, 19.557, 126)
    true_A, true_mu, true_sigma = 0.14, 19.5564, 6e-5
    y = gaussian_func(x, true_A, true_mu, true_sigma)
    np.random.seed(42)
    y += np.random.normal(0, 0.001, len(y))

    result = fit_gaussian(x, y)
    assert result.p0[0] == approx(true_A, rel=0.3)
    assert result.p0[1] == approx(true_mu, abs=0.001)
    # The sigma should be small, not 30x the x range!
    assert result.p0[2] < 0.01


def test_fit_gaussian_returns_cached_fit():
    """Fit property should be cached — calling twice gives same object."""
    x = np.linspace(-3, 3, 50)
    y = gaussian_func(x, 1.0, 0.0, 1.0)
    result = fit_gaussian(x, y)
    fit1 = result.fit
    fit2 = result.fit
    assert fit1 is fit2  # same object, not recomputed


def test_fit_gaussian_fit_interpolated():
    """fit_interpolated should return denser x and y arrays."""
    x = np.linspace(-3, 3, 50)
    y = gaussian_func(x, 1.0, 0.0, 1.0)
    result = fit_gaussian(x, y)
    x_dense, y_dense = result.fit_interpolated(interpolation=10)
    assert len(x_dense) == 500
    assert len(y_dense) == 500
    assert y_dense[250] == approx(1.0, abs=0.01)  # peak near center


# ---------- Double Gaussian fit tests --------------------------------------

def test_fit_double_gaussian():
    """Fit a clean double Gaussian."""
    x = np.linspace(-6, 6, 300)
    y = double_gaussian_func(x, 1.0, -2.0, 0.5, 0.7, 2.0, 0.5)
    np.random.seed(42)
    y += np.random.normal(0, 0.01, len(y))

    result = fit_double_gaussian(x, y)
    assert isinstance(result, FitResult)
    assert len(result.p0) == 6
    # The two peaks should be near -2 and +2
    mus = sorted([result.p0[1], result.p0[4]])
    assert mus[0] == approx(-2.0, abs=0.5)
    assert mus[1] == approx(2.0, abs=0.5)


# ---------- Rabi fit tests -------------------------------------------------

def test_fit_rabi():
    """Fit a Rabi oscillation."""
    x = np.linspace(0, 10, 200)
    true_omega = 2 * np.pi * 0.5  # angular frequency
    true_decay = 5.0
    true_amp = 1.0
    y = rabi_func(x, true_omega, true_decay, true_amp)
    np.random.seed(42)
    y += np.random.normal(0, 0.02, len(y))

    result = fit_rabi(x, y)
    assert isinstance(result, FitResult)
    assert len(result.p0) == 3
    # omega (angular frequency) should be close to true_omega
    assert result.p0[0] == approx(true_omega, rel=0.2)


def test_fit_rabi_shifts_y():
    """Rabi fit subtracts y[0], so even offset data should work."""
    x = np.linspace(0, 10, 200)
    y = rabi_func(x, 2*np.pi*0.5, 5.0, 1.0) + 0.5  # add offset
    result = fit_rabi(x, y)
    assert len(result.p0) == 3
