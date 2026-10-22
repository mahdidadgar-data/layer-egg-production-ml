"""Expected house temperature by climate zone, housing and week of year."""
import numpy as np

TEMP_BASE = {"temperate": 21.0, "warm": 23.5}
TEMP_AMPLITUDE = {"temperate": 3.5, "warm": 5.5}
HOUSING_TEMP_FACTOR = {"enriched cage": 0.7, "barn": 0.9, "free range": 1.1}


def climate_normal(zone, housing, week_of_year):
    """Expected weekly maximum house temperature (C) without heat waves or noise.

    A sine curve over the year peaking in late July. Climate-controlled cage
    houses follow the outside swing less than barns and free-range houses.
    """
    woy = np.asarray(week_of_year, dtype=float)
    return TEMP_BASE[zone] + TEMP_AMPLITUDE[zone] * HOUSING_TEMP_FACTOR[housing] * np.sin(
        2 * np.pi * (woy - 16) / 52)
