"""Resistor power-vs-T and capacitor voltage derating."""

from __future__ import annotations

import pandas as pd

from .params import CircuitParams
from .thermal import power_table


def power_derating_curve(p: CircuitParams, temps: list[float] | None = None) -> pd.DataFrame:
    if temps is None:
        temps = list(range(25, 160, 5))
    rows = []
    for t in temps:
        rows.append(
            {
                "temp_c": t,
                "p_rated_derated_W": p.derating.derated_power_w(t),
                "p_use_W": p.derating.resistor_power_use_fraction * p.derating.derated_power_w(t),
            }
        )
    return pd.DataFrame(rows)


def evaluate_derating(p: CircuitParams, *, vin: float | None = None) -> pd.DataFrame:
    return power_table(p, vin=vin)
