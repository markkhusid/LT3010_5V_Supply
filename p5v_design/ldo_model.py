"""Datasheet interpolators for dropout, Ignd, PSRR, noise, Vadj(T)."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .params import CircuitParams, LdoParams

Which = Literal["typ", "max_25", "max_ot"]


def _interp_i(i_a: float, i_pts: tuple[float, ...], y_pts: tuple[float, ...]) -> float:
    i = max(float(i_a), 0.0)
    return float(np.interp(i, i_pts, y_pts))


def dropout_v(
    p: CircuitParams | LdoParams,
    iload: float | None = None,
    *,
    which: Which = "typ",
    temp_c: float | None = None,
) -> float:
    ldo = p if isinstance(p, LdoParams) else p.ldo
    if iload is None:
        iload = p.op.iload_op if not isinstance(p, LdoParams) else 50e-3
    if which == "typ":
        y = ldo.vdo_typ_v
    elif which == "max_25":
        y = ldo.vdo_max_25_v
    else:
        y = ldo.vdo_max_ot_v
    v = _interp_i(iload, ldo.vdo_i_a, y)
    if temp_c is not None and which == "typ":
        # G03: ~+1.4 mV/°C at 50 mA, less at light load. Scale with I.
        dtdc = 1.4e-3 * (float(iload) / 50e-3)
        v = max(0.0, v + dtdc * (float(temp_c) - ldo.t0_c))
    return v


def ignd_a(
    p: CircuitParams | LdoParams,
    iload: float | None = None,
    *,
    which: Which = "typ",
) -> float:
    ldo = p if isinstance(p, LdoParams) else p.ldo
    if iload is None:
        iload = p.op.iload_op if not isinstance(p, LdoParams) else 50e-3
    y = ldo.ignd_typ_a if which == "typ" else ldo.ignd_max_a
    return _interp_i(iload, ldo.ignd_i_a, y)


def vadj_typical_vs_temp(temp_c: float, ldo: LdoParams | None = None) -> float:
    """G05 typical ADJ pin voltage vs T (digitized, ~5 mV bow)."""
    ldo = ldo or LdoParams()
    t = np.array([-50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0])
    v = np.array([1.268, 1.272, 1.275, 1.2755, 1.276, 1.274, 1.272, 1.270, 1.267])
    return float(np.interp(temp_c, t, v) * (ldo.vadj_typ / 1.2755))


def psrr_db(
    f_hz,
    *,
    cout: float = 10e-6,
    typ: bool = True,
    ldo: LdoParams | None = None,
) -> np.ndarray:
    """Piecewise log-f fit to G21 (Cout = 1 µF vs 10 µF)."""
    ldo = ldo or LdoParams()
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    # 10 µF curve (operate Cout)
    f10 = np.array([10.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6])
    db10 = np.array([72.0, 70.0, 62.0, 50.0, 42.0, 38.0, 42.0, 48.0, 42.0, 30.0])
    f1 = np.array([10.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6])
    db1 = np.array([72.0, 70.0, 62.0, 50.0, 38.0, 32.0, 30.0, 35.0, 38.0, 28.0])
    xf = np.log10(np.clip(f, 10.0, 1e6))
    if cout >= 5e-6:
        y = np.interp(xf, np.log10(f10), db10)
    else:
        y = np.interp(xf, np.log10(f1), db1)
    if not typ:
        y = y - (ldo.psrr_typ_db - ldo.psrr_min_db)
    return np.clip(y, 0.0, 90.0)


def noise_psd_uv_rtHz(f_hz, ldo: LdoParams | None = None) -> np.ndarray:
    """G24: ~1.5 µV/√Hz to 10 kHz, then 1/f-ish roll-off."""
    del ldo
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    dens = np.where(f < 1e4, 1.5, 1.5 * np.sqrt(1e4 / np.clip(f, 1.0, None)))
    return dens


def line_dv_adj(
    p: CircuitParams,
    vin: float | None = None,
    *,
    which: Literal["typ", "max"] = "max",
) -> float:
    """ADJ-referred line term vs Vin, referenced to vin_nom."""
    ldo = p.ldo
    vin_u = p.op.vin_nom if vin is None else float(vin)
    mag = ldo.line_typ_v if which == "typ" else ldo.line_max_v
    return mag * (vin_u - p.op.vin_nom) / ldo.line_dv_in


def load_dv_adj(
    p: CircuitParams,
    iload: float | None = None,
    *,
    which: Literal["typ", "max"] = "max",
    over_temp: bool = False,
) -> float:
    """ADJ-referred load term. Negative as Iload increases (G23)."""
    ldo = p.ldo
    i = p.op.iload_op if iload is None else float(iload)
    if which == "typ":
        mag = ldo.load_typ_v
    else:
        mag = ldo.load_max_v(over_temp=over_temp)
    return -mag * (i - ldo.load_i_lo) / ldo.load_di_a


def scale_to_vout(dv_adj: float, p: CircuitParams) -> float:
    return float(dv_adj) * p.gain


def tj_c(
    p: CircuitParams,
    *,
    vin: float | None = None,
    iload: float | None = None,
    ta_c: float | None = None,
    rth_ja: float | None = None,
    which: Which = "typ",
) -> dict[str, float]:
    vin_u = p.op.vin_nom if vin is None else float(vin)
    i = p.op.iload_op if iload is None else float(iload)
    ta = p.op.temp_c if ta_c is None else float(ta_c)
    rth = p.ldo.rth_ja if rth_ja is None else float(rth_ja)
    vout = vout_dc(p)
    ig = ignd_a(p, i, which="max" if which != "typ" else "typ")
    p_pass = max(0.0, i * (vin_u - vout))
    p_gnd = ig * vin_u
    p_tot = p_pass + p_gnd
    dt = p_tot * rth
    return {
        "vin": vin_u,
        "iload": i,
        "vout": vout,
        "ignd": ig,
        "p_pass": p_pass,
        "p_gnd": p_gnd,
        "p_tot": p_tot,
        "rth_ja": rth,
        "ta_c": ta,
        "dt_c": dt,
        "tj_c": ta + dt,
        "tj_max_c": p.ldo.tj_max_c,
        "margin_c": p.ldo.tj_max_c - (ta + dt),
    }


def soa_iout_max(
    p: CircuitParams,
    vin: float,
    *,
    ta_c: float = 25.0,
    rth_ja: float | None = None,
    which: Which = "max_ot",
) -> float:
    """Iout such that Tj = Tjmax, ignoring Ignd first then one correction."""
    rth = p.ldo.rth_ja if rth_ja is None else float(rth_ja)
    p_max = max(0.0, (p.ldo.tj_max_c - ta_c) / max(rth, 1e-9))
    vout = vout_dc(p)
    head = max(vin - vout, 0.05)
    i0 = p_max / head
    ig = ignd_a(p, min(i0, p.ldo.ilim_typ_a), which="max" if which != "typ" else "typ")
    i = max(0.0, (p_max - ig * vin) / head)
    return min(i, p.ldo.ilim_typ_a)


def vout_ideal(
    r1: float,
    r2: float,
    vadj: float,
    iadj: float,
) -> float:
    """Datasheet Fig. 2: Vout = Vadj*(1+Rtop/Rbot) + Iadj*Rtop."""
    return float(vadj) * (1.0 + float(r1) / max(float(r2), 1e-30)) + float(iadj) * float(r1)


def vout_dc(
    p: CircuitParams,
    *,
    vadj: float | None = None,
    iadj: float | None = None,
    r1: float | None = None,
    r2: float | None = None,
) -> float:
    return vout_ideal(
        p.r1 if r1 is None else r1,
        p.r2 if r2 is None else r2,
        p.vadj if vadj is None else vadj,
        p.iadj if iadj is None else iadj,
    )


def cff_z_ohm(p: CircuitParams, f_hz: float = 10e3) -> float:
    if p.cff <= 0.0:
        return float("inf")
    return 1.0 / (2.0 * math.pi * f_hz * p.cff)


def stability_checks(p: CircuitParams) -> dict[str, float | bool | str]:
    z10k = cff_z_ohm(p, 10e3)
    return {
        "i_div_a": p.i_div,
        "iadj_typ_a": p.ldo.iadj_typ,
        "i_div_over_iadj": p.i_div / max(p.ldo.iadj_typ, 1e-30),
        "r2_lt_250k": p.r2 < p.ldo.rbot_max_ohm,
        "cout_ok": p.cout >= p.ldo.cout_min_f,
        "esr_ok": p.esr_out <= p.ldo.esr_max_ohm,
        "cff_z_10khz": z10k,
        "cff_rule_ok": z10k < p.r2,  # datasheet: |Xc(10 kHz)| < Rbot
        "note": "Datasheet Fig. 2: Z(Cff) at 10 kHz should be < bottom resistor (our R2).",
    }


# Copper-area table, datasheet Table 1
COPPER_RTH_JA = (
    (2500.0, 2500.0, 2500.0, 40.0),
    (1000.0, 2500.0, 2500.0, 45.0),
    (225.0, 2500.0, 2500.0, 50.0),
    (100.0, 2500.0, 2500.0, 62.0),
)
