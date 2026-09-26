"""Closed-form LT3010 setpoint, error budget, regulation, headroom."""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from .ldo_model import (
    cff_z_ohm,
    dropout_v,
    ignd_a,
    line_dv_adj,
    load_dv_adj,
    noise_psd_uv_rtHz,
    psrr_db,
    scale_to_vout,
    soa_iout_max,
    stability_checks,
    tj_c,
    vout_dc,
    vout_ideal,
)
from .params import CircuitParams, format_eng

ErrorMode = Literal["envelope", "stacked"]


def closed_loop_gain(p: CircuitParams) -> float:
    return p.gain


def vout_from_parts(
    p: CircuitParams,
    *,
    vadj: float | None = None,
    iadj: float | None = None,
    r1: float | None = None,
    r2: float | None = None,
    vin: float | None = None,
    iload: float | None = None,
    mode: str | None = None,
    over_temp: bool = False,
) -> float:
    """Vout with optional stacked line/load on top of the given Vadj."""
    r1u = p.r1 if r1 is None else float(r1)
    r2u = p.r2 if r2 is None else float(r2)
    vadj_u = p.vadj if vadj is None else float(vadj)
    iadj_u = p.iadj if iadj is None else float(iadj)
    mode_u = p.error_mode if mode is None else mode
    if mode_u == "stacked":
        which = "max"
        vadj_u = (
            vadj_u
            + line_dv_adj(p, vin, which=which)
            + load_dv_adj(p, iload, which=which, over_temp=over_temp)
        )
    return vout_ideal(r1u, r2u, vadj_u, iadj_u)


def setpoint(p: CircuitParams | None = None) -> dict[str, float]:
    if p is None:
        p = CircuitParams()
    v = vout_dc(p)
    v_no_iadj = vout_ideal(p.r1, p.r2, p.vadj, 0.0)
    return {
        "vadj": p.vadj,
        "gain": p.gain,
        "r1": p.r1,
        "r2": p.r2,
        "iadj": p.iadj,
        "i_div": p.i_div,
        "vout": v,
        "vout_no_iadj": v_no_iadj,
        "dv_iadj": v - v_no_iadj,
        "vout_target": p.op.vout_target,
        "setpoint_bias": v - p.op.vout_target,
        "setpoint_bias_pct": 100.0 * (v - p.op.vout_target) / p.op.vout_target,
        "iload_op": p.op.iload_op,
        "r_load": p.r_load,
        "iload_at_vout": v / max(p.r_load, 1e-12),
    }


# Display units. Scale puts the number in that unit (iadj 50e-9 A → 50 nA)
# so the value column stays ordinary floats except for extremes.
DEFAULT_METRIC_UNITS: dict[str, str] = {
    "vadj": "V",
    "gain": "—",
    "r1": "kΩ",
    "r2": "kΩ",
    "iadj": "nA",
    "i_div": "mA",
    "vout": "V",
    "vout_no_iadj": "V",
    "dv_iadj": "mV",
    "vout_target": "V",
    "setpoint_bias": "mV",
    "setpoint_bias_pct": "%",
    "iload_op": "mA",
    "iload": "mA",
    "r_load": "Ω",
    "iload_at_vout": "mA",
    "vin": "V",
    "vdo": "mV",
    "vdo_typ": "mV",
    "vdo_wc": "mV",
    "vin_min_reg": "V",
    "headroom": "V",
    "headroom_typ": "V",
    "headroom_wc": "V",
    "in_dropout": "—",
    "ignd": "mA",
    "iin": "mA",
    "p_pass": "mW",
    "p_gnd": "mW",
    "p_tot": "mW",
    "p_diss": "mW",
    "p_in": "mW",
    "p_out": "mW",
    "rth_ja": "°C/W",
    "ta_c": "°C",
    "dt_c": "°C",
    "tj_c": "°C",
    "tj_max_c": "°C",
    "margin_c": "°C",
    "efficiency": "%",
    "ilim_margin": "mA",
    "cff_z_10khz": "kΩ",
    "psrr_120_db": "dB",
    "noise_rms": "µV",
}
DEFAULT_METRIC_SCALE: dict[str, float] = {
    "r1": 1e-3,
    "r2": 1e-3,
    "iadj": 1e9,
    "i_div": 1e3,
    "dv_iadj": 1e3,
    "setpoint_bias": 1e3,
    "iload_op": 1e3,
    "iload": 1e3,
    "iload_at_vout": 1e3,
    "vdo": 1e3,
    "vdo_typ": 1e3,
    "vdo_wc": 1e3,
    "ignd": 1e3,
    "iin": 1e3,
    "p_pass": 1e3,
    "p_gnd": 1e3,
    "p_tot": 1e3,
    "p_diss": 1e3,
    "p_in": 1e3,
    "p_out": 1e3,
    "efficiency": 100.0,
    "ilim_margin": 1e3,
    "cff_z_10khz": 1e-3,
    "noise_rms": 1e6,
}
SETPOINT_UNITS = DEFAULT_METRIC_UNITS
SETPOINT_SCALE = DEFAULT_METRIC_SCALE


def fmt_float(x: object, *, small: float = 1e-3, large: float = 1e6) -> str:
    """Ordinary float unless |x| is very small or very large."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return str(v)
    ax = abs(v)
    if ax == 0.0:
        return "0"
    if ax < small or ax >= large:
        return f"{v:.4e}"
    return f"{v:.6g}"


def metrics_table(
    values: Mapping[str, Any],
    units: Mapping[str, str] | None = None,
    *,
    scale: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """One row per metric: numeric ``value`` plus ``unit``.

    ``scale`` converts SI into the unit (mA, mV, kΩ, …). Known names use
    ``DEFAULT_METRIC_UNITS`` / ``DEFAULT_METRIC_SCALE`` unless overridden.
    """
    units_u = {**DEFAULT_METRIC_UNITS, **(units or {})}
    scale_u = {**DEFAULT_METRIC_SCALE, **(scale or {})}
    rows = []
    for name, raw in values.items():
        if isinstance(raw, (bool, np.bool_)):
            rows.append({"parameter": name, "value": bool(raw), "unit": units_u.get(name, "")})
            continue
        try:
            v = float(raw) * float(scale_u.get(name, 1.0))
        except (TypeError, ValueError):
            rows.append({"parameter": name, "value": raw, "unit": units_u.get(name, "")})
            continue
        rows.append({"parameter": name, "value": v, "unit": units_u.get(name, "")})
    return pd.DataFrame(rows).set_index("parameter")


def style_metrics(df: pd.DataFrame):
    """Styler: ordinary floats except very large / very small values."""
    fmt = {c: fmt_float for c in df.columns if c != "unit"}
    return df.style.format(fmt)


def wide_metrics_table(
    df: pd.DataFrame,
    *,
    header: str = "iload",
) -> pd.DataFrame:
    """Transpose an SI operating-point frame: one row per metric, plus ``unit``.

    Column titles come from ``header`` (e.g. load current) after scaling.
    """
    units_u = DEFAULT_METRIC_UNITS
    scale_u = DEFAULT_METRIC_SCALE
    work = df.reset_index(drop=True)
    headers: list[str] = []
    if header in work.columns:
        hs = float(scale_u.get(header, 1.0))
        hu = units_u.get(header, "")
        headers = [f"{float(v) * hs:.6g} {hu}".strip() for v in work[header]]
    else:
        headers = [str(i) for i in range(len(work))]
    rows = []
    for col in work.columns:
        if col == header:
            continue
        sc = float(scale_u.get(col, 1.0))
        un = units_u.get(col, "")
        rec: dict[str, Any] = {"parameter": col, "unit": un}
        for i, v in enumerate(work[col]):
            rec[headers[i]] = float(v) * sc
        rows.append(rec)
    return pd.DataFrame(rows).set_index("parameter")


def setpoint_table(p: CircuitParams | None = None) -> pd.DataFrame:
    return metrics_table(setpoint(p))


def headroom(
    p: CircuitParams | None = None,
    *,
    vin: float | None = None,
    iload: float | None = None,
    which: Literal["typ", "max_25", "max_ot"] = "max_ot",
) -> dict[str, float]:
    if p is None:
        p = CircuitParams()
    vin_u = p.op.vin_nom if vin is None else float(vin)
    i = p.op.iload_op if iload is None else float(iload)
    vout = vout_dc(p)
    vdo = dropout_v(p, i, which=which)
    vin_min = vout + vdo
    return {
        "vin": vin_u,
        "vout": vout,
        "iload": i,
        "vdo": vdo,
        "vin_min_reg": vin_min,
        "headroom": vin_u - vin_min,
        "in_dropout": vin_u < vin_min,
    }


def summary_metrics(
    p: CircuitParams | None = None,
    *,
    vin: float | None = None,
    iload: float | None = None,
) -> dict[str, float]:
    if p is None:
        p = CircuitParams()
    vin_u = p.op.vin_nom if vin is None else float(vin)
    i = p.op.iload_op if iload is None else float(iload)
    sp = setpoint(p)
    hr = headroom(p, vin=vin_u, iload=i, which="typ")
    hr_wc = headroom(p, vin=vin_u, iload=i, which="max_ot")
    th = tj_c(p, vin=vin_u, iload=i, which="typ")
    ig = ignd_a(p, i, which="typ")
    iin = i + ig
    stab = stability_checks(p)
    return {
        "vout": sp["vout"],
        "vadj": sp["vadj"],
        "gain": sp["gain"],
        "setpoint_bias": sp["setpoint_bias"],
        "i_div": sp["i_div"],
        "dv_iadj": sp["dv_iadj"],
        "vin": vin_u,
        "iload": i,
        "vdo_typ": hr["vdo"],
        "vdo_wc": hr_wc["vdo"],
        "headroom_typ": hr["headroom"],
        "headroom_wc": hr_wc["headroom"],
        "ignd": ig,
        "iin": iin,
        "efficiency": (sp["vout"] * i) / max(vin_u * iin, 1e-30),
        "p_diss": th["p_tot"],
        "tj_c": th["tj_c"],
        "ilim_margin": p.ldo.ilim_min_ot_a - i,
        "cff_z_10khz": float(stab["cff_z_10khz"]),
        "psrr_120_db": float(psrr_db(120.0, cout=p.cout)[0]),
        "noise_rms": p.ldo.noise_rms_v * p.gain / (5.0 / 1.275),  # scale vs 5 V-class
    }


def vout_vs_vin(
    p: CircuitParams | None = None,
    vin: np.ndarray | None = None,
    *,
    which_do: Literal["typ", "max_25", "max_ot"] = "typ",
) -> tuple[np.ndarray, np.ndarray]:
    if p is None:
        p = CircuitParams()
    if vin is None:
        vin = np.linspace(3.0, 12.0, 181)
    # Constant setpoint + typ line tilt. Load regulation is an Iload plot, not a Vin plot.
    v0 = vout_dc(p)
    vout_reg = np.array([
        v0 + scale_to_vout(line_dv_adj(p, float(v), which="typ"), p) for v in vin
    ])
    vdo = np.array([dropout_v(p, p.op.iload_op, which=which_do) for _ in vin])
    vout = np.minimum(vout_reg, np.maximum(vin - vdo, 0.0))
    return np.asarray(vin, dtype=float), vout


def vout_vs_iload(
    p: CircuitParams | None = None,
    iload: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if p is None:
        p = CircuitParams()
    if iload is None:
        iload = np.linspace(0.0, 50e-3, 101)
    vout = np.array([vout_from_parts(p, iload=float(i), mode="stacked") for i in iload])
    return np.asarray(iload, dtype=float), vout


def error_budget_table(
    p: CircuitParams | None = None,
    *,
    over_temp: bool = False,
    vin: float | None = None,
    iload: float | None = None,
) -> pd.DataFrame:
    """One-at-a-time Vout contributions. Envelope vs stacked called out.

    Envelope terms (default WC): Vadj box, Iadj, R1, R2.
    Stacked extras (do not add to envelope): line, load, Vadj(T) typical bow.
    """
    if p is None:
        p = CircuitParams()
    vin_u = p.op.vin_nom if vin is None else float(vin)
    i = p.op.iload_op if iload is None else float(iload)
    nom = vout_dc(p)
    ldo = p.ldo
    lo_v, hi_v = ldo.vadj_bounds(over_temp=over_temp)
    rows: list[dict[str, Any]] = []

    def add(name, v_lo, v_hi, group, note=""):
        rows.append(
            {
                "term": name,
                "group": group,
                "vout_lo": v_lo,
                "vout_hi": v_hi,
                "delta_hi_mV": 1e3 * (v_hi - nom),
                "delta_lo_mV": 1e3 * (v_lo - nom),
                "half_span_mV": 1e3 * 0.5 * abs(v_hi - v_lo),
                "note": note,
            }
        )

    add(
        "Vadj box",
        vout_ideal(p.r1, p.r2, lo_v, p.iadj),
        vout_ideal(p.r1, p.r2, hi_v, p.iadj),
        "envelope",
        "page 3 ADJ pin; over-temp includes line+load" if over_temp else "page 3 25 °C ADJ",
    )
    add(
        "Iadj",
        vout_ideal(p.r1, p.r2, p.vadj, 0.0),
        vout_ideal(p.r1, p.r2, p.vadj, ldo.iadj_max_at_temp(p.op.temp_c)),
        "envelope",
        "current into ADJ through R1",
    )
    r1_lo, r1_hi = p.r_tol.bounds(p.r1)
    r2_lo, r2_hi = p.r_tol.bounds(p.r2)
    add(
        "R1 (top)",
        vout_ideal(r1_lo, p.r2, p.vadj, p.iadj),
        vout_ideal(r1_hi, p.r2, p.vadj, p.iadj),
        "envelope",
        f"±{p.r_tol.percent:.2g} %",
    )
    add(
        "R2 (bot)",
        vout_ideal(p.r1, r2_hi, p.vadj, p.iadj),  # larger R2 → smaller Vout
        vout_ideal(p.r1, r2_lo, p.vadj, p.iadj),
        "envelope",
        f"±{p.r_tol.percent:.2g} %",
    )
    dv_line = scale_to_vout(line_dv_adj(p, vin_u, which="max"), p)
    add(
        "Line (stacked extra)",
        nom - abs(dv_line),
        nom + abs(dv_line),
        "stacked",
        "Do not add on top of over-temp Vadj box",
    )
    dv_load = scale_to_vout(load_dv_adj(p, i, which="max", over_temp=over_temp), p)
    add(
        "Load (stacked extra)",
        nom + dv_load,  # dv_load is negative at operate I
        nom,
        "stacked",
        "G23: Vout falls as Iload rises",
    )
    df = pd.DataFrame(rows)
    env = df[df["group"] == "envelope"]
    rss = float(np.sqrt(np.sum(np.square(env["half_span_mV"].to_numpy(dtype=float)))))
    df.attrs["nominal_vout"] = nom
    df.attrs["rss_half_span_mV"] = rss
    df.attrs["rss_lo"] = nom - rss * 1e-3
    df.attrs["rss_hi"] = nom + rss * 1e-3
    return df


def envelope_bounds(
    p: CircuitParams | None = None,
    *,
    over_temp: bool = False,
) -> dict[str, float]:
    """Endpoint box on envelope sources: Vadj, Iadj, R1, R2."""
    if p is None:
        p = CircuitParams()
    ldo = p.ldo
    va_lo, va_hi = ldo.vadj_bounds(over_temp=over_temp)
    ia_lo, ia_hi = 0.0, ldo.iadj_max_at_temp(p.op.temp_c)
    r1_lo, r1_hi = p.r_tol.bounds(p.r1)
    r2_lo, r2_hi = p.r_tol.bounds(p.r2)
    nom = vout_dc(p)
    # Vout rises with Vadj, Iadj, R1; falls with R2
    vmin = vout_ideal(r1_lo, r2_hi, va_lo, ia_lo)
    vmax = vout_ideal(r1_hi, r2_lo, va_hi, ia_hi)
    return {
        "nominal": nom,
        "min": vmin,
        "max": vmax,
        "span": vmax - vmin,
        "min_pct": 100.0 * (vmin / p.op.vout_target - 1.0),
        "max_pct": 100.0 * (vmax / p.op.vout_target - 1.0),
    }


def stacked_bounds(
    p: CircuitParams | None = None,
    *,
    over_temp: bool = False,
    vin_lo: float | None = None,
    vin_hi: float | None = None,
    iload_lo: float | None = None,
    iload_hi: float | None = None,
) -> dict[str, float]:
    if p is None:
        p = CircuitParams()
    vin_lo = p.op.vin_min if vin_lo is None else float(vin_lo)
    vin_hi = p.op.vin_max if vin_hi is None else float(vin_hi)
    iload_lo = p.op.iload_min if iload_lo is None else float(iload_lo)
    iload_hi = p.op.iload_op if iload_hi is None else float(iload_hi)
    env = envelope_bounds(p, over_temp=False)  # 25 °C Vadj box
    # line and load referred to Vout, using max columns
    d_line = scale_to_vout(abs(line_dv_adj(p, vin_hi, which="max")), p)
    d_line += scale_to_vout(abs(line_dv_adj(p, vin_lo, which="max")), p)
    d_line *= 0.5
    d_load = abs(scale_to_vout(load_dv_adj(p, iload_hi, which="max", over_temp=over_temp), p))
    return {
        "nominal": env["nominal"],
        "min": env["min"] - d_line - d_load,
        "max": env["max"] + d_line,
        "span": (env["max"] + d_line) - (env["min"] - d_line - d_load),
        "line_half_v": d_line,
        "load_drop_v": d_load,
    }


def dc_error_budget(p: CircuitParams | None = None) -> pd.DataFrame:
    return error_budget_table(p)


def latex_vout() -> str:
    return (
        r"V_{\mathrm{OUT}}=V_{\mathrm{ADJ}}\left(1+\frac{R_1}{R_2}\right)+I_{\mathrm{ADJ}}R_1"
    )


def regulation_family_vin(
    p: CircuitParams | None = None,
) -> pd.DataFrame:
    if p is None:
        p = CircuitParams()
    vin, v_typ = vout_vs_vin(p, which_do="typ")
    _, v_wc = vout_vs_vin(p, which_do="max_ot")
    return pd.DataFrame({"vin": vin, "vout_typ": v_typ, "vout_do_wc": v_wc})


def noise_rms_scaled(p: CircuitParams) -> float:
    """Datasheet 100 µVrms is at a 5 V-class / 1.275 V ref ratio ~3.92.

    Treat the 100 µVrms number as output-referred at the specified condition
    and scale only with sqrt(Cout_ref/Cout) as a weak Cout dependence.
    """
    c_ref = 10e-6
    scale_c = math_sqrt(c_ref / max(p.cout, 1e-12))
    # Output noise of an adjustable LDO scales ~ G = Vout/Vadj
    scale_g = p.gain / (5.0 / 1.275)
    return p.ldo.noise_rms_v * scale_g * scale_c


def math_sqrt(x: float) -> float:
    return float(np.sqrt(max(x, 0.0)))


def psrr_ripple_out(
    p: CircuitParams,
    vin_ripple_pp: float,
    f_hz: float,
) -> float:
    db = float(psrr_db(f_hz, cout=p.cout)[0])
    return vin_ripple_pp * 10 ** (-db / 20.0)


# Re-exports used by notebooks
__all__ = [
    "closed_loop_gain",
    "vout_from_parts",
    "vout_dc",
    "setpoint",
    "headroom",
    "summary_metrics",
    "vout_vs_vin",
    "vout_vs_iload",
    "error_budget_table",
    "envelope_bounds",
    "stacked_bounds",
    "dc_error_budget",
    "latex_vout",
    "regulation_family_vin",
    "noise_rms_scaled",
    "psrr_ripple_out",
    "dropout_v",
    "ignd_a",
    "tj_c",
    "soa_iout_max",
    "stability_checks",
    "cff_z_ohm",
    "psrr_db",
    "noise_psd_uv_rtHz",
    "format_eng",
]
