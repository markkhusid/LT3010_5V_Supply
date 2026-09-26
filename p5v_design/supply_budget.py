"""Isolated 9 V rail current and efficiency."""

from __future__ import annotations

import pandas as pd

from .analysis import vout_dc
from .ldo_model import ignd_a
from .params import CircuitParams


def rail_budget(
    p: CircuitParams,
    *,
    vin: float | None = None,
    iload: float | None = None,
    which: str = "typ",
) -> dict[str, float]:
    vin_u = p.op.vin_nom if vin is None else float(vin)
    i = p.op.iload_op if iload is None else float(iload)
    vout = vout_dc(p)
    ig = ignd_a(p, i, which="max" if which != "typ" else "typ")
    iin = i + ig
    return {
        "vin": vin_u,
        "vout": vout,
        "iload": i,
        "ignd": ig,
        "iin": iin,
        "p_in": vin_u * iin,
        "p_out": vout * i,
        "p_diss": vin_u * iin - vout * i,
        "efficiency": (vout * i) / max(vin_u * iin, 1e-30),
    }


def budget_vs_load(
    p: CircuitParams,
    loads: list[float] | None = None,
    *,
    which: str = "typ",
) -> pd.DataFrame:
    if loads is None:
        loads = [1e-3, 10e-3, 50e-3]
    return pd.DataFrame([rail_budget(p, iload=i, which=which) for i in loads])


def budget_vs_vin(
    p: CircuitParams,
    vins: list[float] | None = None,
    *,
    which: str = "typ",
) -> pd.DataFrame:
    if vins is None:
        vins = [6.0, 8.0, 9.0, 10.0, 12.0]
    return pd.DataFrame([rail_budget(p, vin=v, which=which) for v in vins])
