"""Analytical estimates for startup and load/line steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import vout_dc
from .params import CircuitParams


@dataclass
class TranResult:
    t: np.ndarray
    vout: np.ndarray
    vin: np.ndarray | None = None
    iout: np.ndarray | None = None


def load_step(
    p: CircuitParams | None = None,
    *,
    i0: float = 1e-3,
    i1: float | None = None,
    t_step: float = 50e-6,
    t_stop: float = 1e-3,
    n: int = 2000,
    tau_loop: float = 80e-6,
) -> TranResult:
    """ESR spike + Cout ramp, then 1st-order loop recovery.

    ``tau_loop`` is an editable stand-in for the error-amp recovery time
    (G26 is ~200 µs with 1 µF; 10 µF is slower/smaller).
    """
    if p is None:
        p = CircuitParams()
    i1 = p.op.iload_op if i1 is None else float(i1)
    t = np.linspace(0.0, t_stop, n)
    vss = vout_dc(p)
    di = i1 - i0
    esr_spike = di * p.esr_out
    # charge deficit until the loop supplies di: ~ di * tau / C
    cap_dip = di * tau_loop / max(p.cout, 1e-12)
    dip = esr_spike + cap_dip
    y = np.ones_like(t) * vss
    mask = t >= t_step
    dt = t[mask] - t_step
    y[mask] = vss - dip * np.exp(-dt / tau_loop)
    # instantaneous ESR: add a short rectangular spike
    spike = mask & (t < t_step + 2e-6)
    y[spike] -= esr_spike
    iout = np.where(t < t_step, i0, i1)
    return TranResult(t=t, vout=y, iout=iout)


def startup(
    p: CircuitParams | None = None,
    *,
    t_stop: float = 2e-3,
    n: int = 2000,
    i_lim: float | None = None,
) -> TranResult:
    """Cout charged at Ilim until Vout, then settles to regulation."""
    if p is None:
        p = CircuitParams()
    i_lim = p.ldo.ilim_typ_a if i_lim is None else float(i_lim)
    vss = vout_dc(p)
    t_ch = p.cout * vss / max(i_lim - p.op.iload_op, 1e-6)
    t = np.linspace(0.0, t_stop, n)
    v = np.minimum(vss, (i_lim - p.op.iload_op) * t / max(p.cout, 1e-12))
    v = np.minimum(v, vss)
    vin = np.full_like(t, p.op.vin_nom)
    return TranResult(t=t, vout=v, vin=vin)


def line_step(
    p: CircuitParams | None = None,
    *,
    vin0: float = 9.0,
    vin1: float = 12.0,
    t_step: float = 100e-6,
    t_stop: float = 1e-3,
    n: int = 2000,
    tau: float = 50e-6,
) -> TranResult:
    from .analysis import vout_from_parts

    if p is None:
        p = CircuitParams()
    t = np.linspace(0.0, t_stop, n)
    v0 = vout_from_parts(p, vin=vin0, mode="stacked")
    v1 = vout_from_parts(p, vin=vin1, mode="stacked")
    vin = np.where(t < t_step, vin0, vin1)
    y = np.where(t < t_step, v0, v0 + (v1 - v0) * (1.0 - np.exp(-(t - t_step) / tau)))
    return TranResult(t=t, vout=y, vin=vin)
