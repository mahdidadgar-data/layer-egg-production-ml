"""Laying curve model of Yang, Wu and McMillan (1989).

    y(t) = a * exp(-b * t) / (1 + exp(-c * (t - d)))

t is age in weeks and y is hen-day production (%).
  a  scale parameter
  b  rate of decrease in laying ability (persistency; smaller is better)
  c  reciprocal indicator of the spread in sexual maturity (larger = more uniform onset)
  d  mean age at sexual maturity (weeks)
"""
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

P0 = (110.0, 0.005, 1.5, 20.5)
BOUNDS = ([50.0, 0.0, 0.2, 17.0], [160.0, 0.03, 6.0, 26.0])
PARAMS = ["a", "b", "c", "d"]


def yang(t, a, b, c, d):
    t = np.asarray(t, dtype=float)
    return a * np.exp(-b * t) / (1.0 + np.exp(-c * (t - d)))


def fit_yang(ages, hen_day, p0=P0):
    """Least-squares fit. Returns (params, rmse) or (None, nan) if it fails."""
    ages = np.asarray(ages, dtype=float)
    y = np.asarray(hen_day, dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 8:
        return None, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            p, _ = curve_fit(yang, ages[ok], y[ok], p0=p0, bounds=BOUNDS, maxfev=20000)
    except (RuntimeError, ValueError):
        return None, np.nan
    rmse = float(np.sqrt(np.mean((yang(ages[ok], *p) - y[ok]) ** 2)))
    return p, rmse


def peak_and_age(params, ages=np.arange(19, 91, 0.25)):
    y = yang(ages, *params)
    i = int(np.argmax(y))
    return float(y[i]), float(ages[i])


def age_at_50pct(params, ages=np.arange(17, 40, 0.05)):
    y = yang(ages, *params)
    above = np.where(y >= 50)[0]
    return float(ages[above[0]]) if len(above) else np.nan
