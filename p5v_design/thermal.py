"""Per-part power, voltage utilization, junction temperature."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .analysis import vout_dc
from .ldo_model import COPPER_RTH_JA, ignd_a, tj_c
from .params import CircuitParams


@dataclass(frozen=True)
class ThermalPart:
    name: str
    package: str
    rth_ja_C_per_W: float
    p_rated_W: float
    v_max_V: float
    source: str


THERMAL_LIBRARY = {
    "0805": ThermalPart("0805 0.125 W", "0805", 220.0, 0.125, 150.0, "0805 0.125 W"),
    "LT3010": ThermalPart("LT3010 MS8E", "MS8E", 40.0, 0.0, 80.0, "θJA 40 °C/W (2500 mm²)"),
    "C16V": ThermalPart("Cout 16 V", "1210", 0.0, 0.0, 16.0, "ceramic"),
    "C25V": ThermalPart("Cin 25 V", "1206", 0.0, 0.0, 25.0, "ceramic"),
}


def power_table(
    p: CircuitParams,
    *,
    vin: float | None = None,
    iload: float | None = None,
    ta_C: float = 25.0,
    v_derate_frac: float = 0.80,
    which: str = "typ",
) -> pd.DataFrame:
    vin_u = p.op.vin_nom if vin is None else float(vin)
    i = p.op.iload_op if iload is None else float(iload)
    vout = vout_dc(p)
    th = tj_c(p, vin=vin_u, iload=i, ta_c=ta_C, which="typ" if which == "typ" else "max_ot")
    i_div = p.ldo.vadj_typ / p.r2
    rows = []

    def add(ref, part: ThermalPart, p_w, v_app, note=""):
        v_der = v_derate_frac * part.v_max_V if part.v_max_V else float("nan")
        dt = p_w * part.rth_ja_C_per_W
        p_use = p_w / part.p_rated_W if part.p_rated_W > 0 else float("nan")
        p_th_max = (p.ldo.tj_max_c - ta_C) / part.rth_ja_C_per_W if part.rth_ja_C_per_W > 0 else float("nan")
        p_use_th = p_w / p_th_max if p_th_max and p_th_max > 0 else float("nan")
        v_util = v_app / v_der if v_der and v_der > 0 else float("nan")
        status = "OK"
        if part.p_rated_W > 0 and p_use > p.derating.resistor_power_use_fraction:
            status = "P>50%"
        if part.v_max_V > 0 and v_app > v_der:
            status = "V>80%"
        if ref == "U1" and th["tj_c"] > p.ldo.tj_max_c:
            status = "Tj>max"
        rows.append(
            {
                "ref": ref,
                "component": ref,
                "package": part.package,
                "P_W": p_w,
                "P_mW": 1e3 * p_w,
                "P_rated_W": part.p_rated_W,
                "P_util": p_use,
                "utilization_pct": 100.0 * p_use if p_use == p_use else float("nan"),
                "P_thermal_max_W": p_th_max,
                "P_util_thermal": p_use_th,
                "P_util_thermal_pct": 100.0 * p_use_th if p_use_th == p_use_th else float("nan"),
                "V_applied_V": v_app,
                "V_max_V": part.v_max_V,
                "V_derated_V": v_der,
                "V_util_derated": v_util,
                "V_util_derated_pct": 100.0 * v_util if v_util == v_util else float("nan"),
                "dT_C": dt,
                "Tj_C": ta_C + dt,
                "status": status,
                "note": note,
            }
        )

    add("R1", THERMAL_LIBRARY["0805"], i_div**2 * p.r1, vout - p.vadj, "feedback top")
    add("R2", THERMAL_LIBRARY["0805"], i_div**2 * p.r2, p.vadj, "feedback bot")
    load_part = ThermalPart("operate load", "—", 0.0, 0.0, 0.0, "Iload equivalent, not a BOM resistor")
    add("Rload", load_part, vout * i, vout, "Iload equivalent (not a board part)")
    u1 = ThermalPart("LT3010", "MS8E", p.ldo.rth_ja, 0.0, 80.0, "datasheet")
    add("U1", u1, th["p_tot"], vin_u, f"Ppass={th['p_pass']:.3f} W + Pgnd={th['p_gnd']:.3f} W")
    add("C2", THERMAL_LIBRARY["C16V"], 0.0, vout, "Cout")
    add("C4", THERMAL_LIBRARY["C25V"], 0.0, vin_u, "Cin")
    add("C1", THERMAL_LIBRARY["C16V"], 0.0, vout - p.vadj, "Cff")
    return pd.DataFrame(rows)


def power_tables_vs_vin(
    p: CircuitParams,
    vins: list[float] | None = None,
    **kw,
) -> dict[str, pd.DataFrame]:
    if vins is None:
        vins = [p.op.vin_min, p.op.vin_nom, p.op.vin_max]
    return {f"{v:g} V": power_table(p, vin=v, **kw) for v in vins}


def power_tables_vs_load(
    p: CircuitParams,
    loads: list[float] | None = None,
    **kw,
) -> dict[str, pd.DataFrame]:
    if loads is None:
        loads = [p.op.iload_min, p.op.iload_op, p.ldo.load_i_hi]
    return {f"{1e3 * i:.0f} mA": power_table(p, iload=i, **kw) for i in loads}


def bom_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the dummy Rload row used only as an Iload equivalent."""
    if "ref" in df.columns:
        return df[df["ref"] != "Rload"].copy()
    return df.copy()


def copper_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "topside_mm2": a,
                "backside_mm2": b,
                "board_mm2": c,
                "rth_ja": r,
            }
            for a, b, c, r in COPPER_RTH_JA
        ]
    )
