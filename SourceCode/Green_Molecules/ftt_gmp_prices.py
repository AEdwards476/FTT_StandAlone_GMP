# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_prices.py
============================================================
Helpers for electricity price inputs used across FTT-GMP.

The NESO average electricity price is the primary source of the
annual-average electricity price in GBP/kWh.  Where the NESO series
has no data for the current year (pre-2024 / post-2050), the
``gm_elec_price_backup`` input is used instead.
"""

NESO_PRICE_OPTION_INDEX = 1
ELECTRICITY_PRICE_SD_FRACTION = 0.2


def get_electricity_price(data):
    """Return the current-year average electricity price in GBP/kWh.

    Uses the NESO annual average (GBP/MWh, converted to GBP/kWh), falling
    back to ``gm_elec_price_backup`` when the NESO value is absent (0).
    """
    neso_price = float(
        data['gm_NESO_av_electricity_price'][0, NESO_PRICE_OPTION_INDEX, 0]
    )
    if neso_price > 0:
        return neso_price / 1000.0
    return float(data['gm_elec_price_backup'][0, 0, 0])


def get_electricity_price_sd(data):
    """Return the standard deviation of the average electricity price.

    Set as a fixed fraction of the current-year mean price.
    """
    return ELECTRICITY_PRICE_SD_FRACTION * get_electricity_price(data)
