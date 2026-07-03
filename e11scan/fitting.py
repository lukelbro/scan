"""
Standalone fitting routines for e11scan.

All functions take plain numpy arrays — no HDF5, no Scan objects.
Each returns an immutable FitResult dataclass with lazy-cached fit evaluation.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Model functions (pure math, no side-effects)
# ---------------------------------------------------------------------------

def gaussian_func(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    """Single Gaussian: A * exp(-(x - mu)^2 / (2 * sigma^2))"""
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))


def double_gaussian_func(
    x: np.ndarray,
    A1: float, mu1: float, sigma1: float,
    A2: float, mu2: float, sigma2: float,
) -> np.ndarray:
    """Sum of two Gaussians."""
    return (
        A1 * np.exp(-(x - mu1) ** 2 / (2.0 * sigma1 ** 2))
        + A2 * np.exp(-(x - mu2) ** 2 / (2.0 * sigma2 ** 2))
    )


def rabi_func(x: np.ndarray, omega: float, decay: float, a: float) -> np.ndarray:
    """Rabi oscillation: a * (1 - cos(x * omega) * exp(-x / decay)) / 2"""
    return a * (1 - np.cos(x * omega) * np.exp(-x / decay)) / 2


# ---------------------------------------------------------------------------
# FitResult — immutable result container with lazy-cached evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitResult:
    """Immutable result of a curve fit.

    Attributes
    ----------
    p0 : fitted parameters (numpy array)
    var_matrix : covariance matrix (numpy array)
    func : the model callable  f(x, *params) -> y
    x_source : the x array that was fitted against
    error_source : the error/sigma array used (or None)
    """
    p0: np.ndarray
    var_matrix: np.ndarray
    func: Callable
    x_source: np.ndarray
    error_source: Optional[np.ndarray] = None

    # --- Lazy-cached evaluation (computed once, then stored) ---------------

    _fit_cache: np.ndarray = field(default=None, repr=False, init=False)

    def _compute_fit(self) -> np.ndarray:
        return self.func(self.x_source, *self.p0)

    @property
    def fit(self) -> np.ndarray:
        """Predicted y values at the source x points."""
        # dataclass(frozen=True) prevents normal attribute assignment,
        # so we use object.__setattr__ for caching.
        if self._fit_cache is None:
            object.__setattr__(self, '_fit_cache', self._compute_fit())
        return self._fit_cache

    def fit_interpolated(self, interpolation: int = 100) -> tuple:
        """Return (x_dense, y_dense) with ``interpolation`` times more points."""
        num_points = len(self.x_source) * interpolation
        x = np.linspace(self.x_source[0], self.x_source[-1], num_points)
        y = self.func(x, *self.p0)
        return x, y



# ---------------------------------------------------------------------------
# Initial-guess helpers
# ---------------------------------------------------------------------------

def _guess_gaussian_sigma(x: np.ndarray) -> float:
    """Return a *conservative* initial guess for Gaussian width.

    Uses 1/10 of the x-range rather than std(y), which can be orders of
    magnitude too wide when the x-axis spans a very narrow interval.
    """
    x_range = x[-1] - x[0]
    return max(x_range / 10.0, np.finfo(float).eps)


def _fft_rabi_guess(x: np.ndarray, y: np.ndarray) -> dict:
    """FFT-based initial guess for Rabi oscillation frequency.

    Returns raw FFT frequency (cycles per unit x) — the Rabi model
    uses omega directly as cos(x * omega), matching the original e11scan
    convention where omega is the raw FFT frequency.
    """
    ff = np.fft.fftfreq(len(x), (x[1] - x[0]))
    Fyy = np.abs(np.fft.fft(y))
    # Exclude the DC component (index 0)
    guess_freq = np.abs(ff[np.argmax(Fyy[1:]) + 1])
    guess_amp = np.std(y) * 2.0 ** 0.5
    return {"omega": guess_freq, "amp": guess_amp}


# ---------------------------------------------------------------------------
# Public fitting API
# ---------------------------------------------------------------------------

def fit_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    error: Optional[np.ndarray] = None,
) -> FitResult:
    """Fit a single Gaussian to (x, y) data.

    Returns a FitResult with .p0 = [A, mu, sigma].
    """
    sigma_guess = _guess_gaussian_sigma(x)
    guess = [np.max(y), x[np.argmax(y)], sigma_guess]

    has_sigma = error is not None and len(np.atleast_1d(error)) > 1
    if has_sigma:
        p0, var = curve_fit(
            gaussian_func, x, y, p0=guess,
            absolute_sigma=True, sigma=error,
        )
    else:
        p0, var = curve_fit(
            gaussian_func, x, y, p0=guess,
            absolute_sigma=False,
        )

    return FitResult(
        p0=np.array(p0),
        var_matrix=np.array(var),
        func=gaussian_func,
        x_source=np.asarray(x),
        error_source=np.asarray(error) if error is not None else None,
    )


def fit_double_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    error: Optional[np.ndarray] = None,
) -> FitResult:
    """Fit a sum of two Gaussians to (x, y) data.

    Uses scipy peak finding to seed the two peaks.
    Returns a FitResult with .p0 = [A1, mu1, sigma1, A2, mu2, sigma2].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma_guess = (x[-1] - x[0]) / 8.0

    # Seed the two peaks via scipy peak finding
    peaks, props = find_peaks(y, prominence=0)
    if len(peaks) >= 2:
        order = np.argsort(props['prominences'])[::-1]
        idx1, idx2 = peaks[order[0]], peaks[order[1]]
    elif len(peaks) == 1:
        idx1 = peaks[0]
        idx2 = int(np.argmax(y == np.partition(y, -2)[-2]))
    else:
        idx1 = int(np.argmax(y))
        idx2 = int(np.argmax(y == np.partition(y, -2)[-2]))

    A1, mu1 = y[idx1], x[idx1]
    A2, mu2 = y[idx2], x[idx2]
    guess = [A1, mu1, sigma_guess, A2, mu2, sigma_guess]

    has_sigma = error is not None and len(np.atleast_1d(error)) > 1
    if has_sigma:
        p0, var = curve_fit(
            double_gaussian_func, x, y, p0=guess,
            absolute_sigma=True, sigma=error,
        )
    else:
        p0, var = curve_fit(
            double_gaussian_func, x, y, p0=guess,
            absolute_sigma=False,
        )

    return FitResult(
        p0=np.array(p0),
        var_matrix=np.array(var),
        func=double_gaussian_func,
        x_source=np.asarray(x),
        error_source=np.asarray(error) if error is not None else None,
    )


def fit_rabi(
    x: np.ndarray,
    y: np.ndarray,
    error: Optional[np.ndarray] = None,
) -> FitResult:
    """Fit a Rabi oscillation model to (x, y) data.

    The Rabi model subtracts y[0] before fitting (so oscillation starts at
    the origin), matching the original e11scan behaviour.

    Returns a FitResult with .p0 = [omega, decay, amplitude].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fft_guess = _fft_rabi_guess(x, y)
    # Use x_range/2 as initial decay guess (avoids decay=0 → exp(-x/0) = 0)
    x_range = x[-1] - x[0]
    initial_decay = x_range / 2.0
    guess = [fft_guess["omega"], initial_decay, fft_guess["amp"]]
    bounds = [[0, 0, 0], [np.inf, np.inf, np.inf]]

    # Rabi fit subtracts y[0] to centre the oscillation at the origin
    y_shifted = y - y[0]

    has_sigma = error is not None and len(np.atleast_1d(error)) > 1
    if has_sigma:
        p0, var = curve_fit(
            rabi_func, x, y_shifted, p0=guess,
            absolute_sigma=True, sigma=error,
            bounds=bounds,
        )
    else:
        p0, var = curve_fit(
            rabi_func, x, y_shifted, p0=guess,
            absolute_sigma=False,
            bounds=bounds,
        )

    return FitResult(
        p0=np.array(p0),
        var_matrix=np.array(var),
        func=rabi_func,
        x_source=np.asarray(x),
        error_source=np.asarray(error) if error is not None else None,
    )
