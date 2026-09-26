"""Endpoint WC, OAT, RSS for Vout / headroom / dissipation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from .analysis import envelope_bounds, headroom, stacked_bounds, vout_dc, vout_from_parts
from .ldo_model import dropout_v, ignd_a, tj_c, vout_ideal
from .params import CircuitParams


def _metrics(
    p: CircuitParams,
    *,
    vadj: float | None = None,
    iadj: float | None = None,
    vin: float | None = None,
    iload: float | None = None,
    mode: str | None = None,
) -> dict[str, float]:
    vin_u = p.op.vin_nom if vin is None else float(vin)
    i = p.op.iload_op if iload is None else float(iload)
    vout = vout_from_parts(
        p, vadj=vadj, iadj=iadj, vin=vin_u, iload=i, mode=mode or "envelope",
    )
    vdo = dropout_v(p, i, which="typ")
    ig = ignd_a(p, i, which="typ")
    th = tj_c(p, vin=vin_u, iload=i, which="typ")
    return {
        "vout": vout,
        "gain": p.gain,
        "headroom": vin_u - vout - vdo,
        "vdo": vdo,
        "ignd": ig,
        "iin": i + ig,
        "p_diss": th["p_tot"],
        "efficiency": (vout * i) / max(vin_u * (i + ig), 1e-30),
        "r1": p.r1,
        "r2": p.r2,
        "vadj": p.vadj if vadj is None else float(vadj),
        "iadj": p.iadj if iadj is None else float(iadj),
    }


@dataclass
class WorstCaseResult:
    nominal: dict[str, float]
    minimum: dict[str, float]
    maximum: dict[str, float]
    rows: list[dict]

    def span(self, key: str) -> float:
        return self.maximum[key] - self.minimum[key]


def analytical_worst_case(
    p: CircuitParams | None = None,
    *,
    over_temp: bool = False,
    mode: str = "envelope",
) -> WorstCaseResult:
    if p is None:
        p = CircuitParams()
    nom = _metrics(p, mode="envelope")
    rows: list[dict] = [dict(nom, corner="nominal")]
    ldo = p.ldo
    va_lo, va_hi = ldo.vadj_bounds(over_temp=over_temp)
    ia_lo, ia_hi = 0.0, ldo.iadj_max_at_temp(p.op.temp_c)
    r1_lo, r1_hi = p.r_tol.bounds(p.r1)
    r2_lo, r2_hi = p.r_tol.bounds(p.r2)

    for r1s, r2s, vas, ias in product(("lo", "hi"), repeat=4):
        r1 = r1_lo if r1s == "lo" else r1_hi
        r2 = r2_lo if r2s == "lo" else r2_hi
        va = va_lo if vas == "lo" else va_hi
        ia = ia_lo if ias == "lo" else ia_hi
        pc = p.with_values(r1=r1, r2=r2, vadj=va, iadj=ia)
        m = _metrics(pc, vadj=va, iadj=ia, mode="envelope")
        m["corner"] = f"R1{r1s}_R2{r2s}_Va{vas}_Ia{ias}"
        rows.append(m)

    if mode == "stacked":
        for vin in (p.op.vin_min, p.op.vin_max):
            m = _metrics(p, vin=vin, mode="stacked")
            m["corner"] = f"stacked_vin={vin:g}"
            rows.append(m)
        for i in (p.op.iload_min, p.op.iload_op):
            m = _metrics(p, iload=i, mode="stacked")
            m["corner"] = f"stacked_i={i:g}"
            rows.append(m)

    skip = {"corner", "vadj", "iadj", "r1", "r2"}
    keys = [k for k in nom if k not in skip]
    minimum = {k: min(r[k] for r in rows) for k in keys}
    maximum = {k: max(r[k] for r in rows) for k in keys}
    return WorstCaseResult(nominal=nom, minimum=minimum, maximum=maximum, rows=rows)


def oat_contributions(
    p: CircuitParams | None = None,
    *,
    metrics: tuple[str, ...] = ("vout", "headroom", "p_diss"),
    over_temp: bool = False,
) -> dict[str, dict[str, float]]:
    if p is None:
        p = CircuitParams()
    contrib: dict[str, dict[str, float]] = {m: {} for m in metrics}

    def span(a: dict, b: dict, met: str) -> float:
        return 0.5 * abs(b[met] - a[met])

    ldo = p.ldo
    va_lo, va_hi = ldo.vadj_bounds(over_temp=over_temp)
    m_lo = _metrics(p, vadj=va_lo)
    m_hi = _metrics(p, vadj=va_hi)
    for met in metrics:
        contrib[met]["Vadj"] = span(m_lo, m_hi, met)

    ia_hi = ldo.iadj_max_at_temp(p.op.temp_c)
    m_lo = _metrics(p, iadj=0.0)
    m_hi = _metrics(p, iadj=ia_hi)
    for met in metrics:
        contrib[met]["Iadj"] = span(m_lo, m_hi, met)

    r1_lo, r1_hi = p.r_tol.bounds(p.r1)
    m_lo = _metrics(p.with_values(r1=r1_lo))
    m_hi = _metrics(p.with_values(r1=r1_hi))
    for met in metrics:
        contrib[met]["R1"] = span(m_lo, m_hi, met)

    r2_lo, r2_hi = p.r_tol.bounds(p.r2)
    m_lo = _metrics(p.with_values(r2=r2_lo))
    m_hi = _metrics(p.with_values(r2=r2_hi))
    for met in metrics:
        contrib[met]["R2"] = span(m_lo, m_hi, met)

    return contrib


def rss_bounds(
    p: CircuitParams | None = None,
    *,
    over_temp: bool = False,
    metric: str = "vout",
) -> dict[str, float]:
    if p is None:
        p = CircuitParams()
    oat = oat_contributions(p, metrics=(metric,), over_temp=over_temp)
    deltas = np.array(list(oat[metric].values()), dtype=float)
    rss = float(np.sqrt(np.sum(deltas**2)))
    nom = _metrics(p)[metric]
    return {
        "nominal": nom,
        "delta_rss": rss,
        "low": nom - rss,
        "high": nom + rss,
        "half_spans": oat[metric],
    }


def envelope_worst_case(
    p: CircuitParams | None = None,
    *,
    temps_c: list[float] | None = None,
) -> tuple[dict[float, WorstCaseResult], WorstCaseResult]:
    if p is None:
        p = CircuitParams()
    if temps_c is None:
        temps_c = [p.ldo.tmin_c, p.ldo.t0_c, p.ldo.tmax_c]
    by_t: dict[float, WorstCaseResult] = {}
    for T in temps_c:
        over = abs(T - p.ldo.t0_c) > 1.0
        pc = p.scaled_at_temp(T)
        by_t[T] = analytical_worst_case(pc, over_temp=over)
    ref = min(by_t, key=lambda t: abs(t - p.ldo.t0_c))
    keys = [k for k in by_t[ref].nominal if k not in {"corner", "vadj", "iadj", "r1", "r2"}]
    minimum = {k: min(r.minimum[k] for r in by_t.values()) for k in keys}
    maximum = {k: max(r.maximum[k] for r in by_t.values()) for k in keys}
    rows = []
    for T, wc in by_t.items():
        for row in wc.rows:
            rows.append(dict(row, temp_c=T))
    env = WorstCaseResult(
        nominal=dict(by_t[ref].nominal), minimum=minimum, maximum=maximum, rows=rows,
    )
    return by_t, env


METRIC_KEYS = ("vout", "headroom", "p_diss", "efficiency", "iin", "vdo")


def corners_frame(wc: WorstCaseResult):
    import pandas as pd

    return pd.DataFrame([r for r in wc.rows if r.get("corner") != "nominal"])


def rss_bounds_many(
    p: CircuitParams | None = None,
    *,
    over_temp: bool = False,
    metrics: tuple[str, ...] = ("vout", "headroom", "p_diss", "efficiency"),
) -> dict[str, dict[str, float]]:
    if p is None:
        p = CircuitParams()
    return {m: rss_bounds(p, over_temp=over_temp, metric=m) for m in metrics}


def spice_wc_corners(
    p: CircuitParams | None = None,
    *,
    workdir: Path | str,
    root: Path | str | None = None,
    max_workers: int | None = None,
    over_temp: bool = False,
    full: bool = True,
) -> list[dict]:
    """ngspice OP at hypercube corners (16 if ``full``, else 4 grouped)."""
    from .parallel import run_parallel
    from .simulate import simulate_op

    if p is None:
        p = CircuitParams()
    ldo = p.ldo
    va_lo, va_hi = ldo.vadj_bounds(over_temp=over_temp)
    ia_lo, ia_hi = 0.0, ldo.iadj_max_at_temp(p.op.temp_c)
    r1_lo, r1_hi = p.r_tol.bounds(p.r1)
    r2_lo, r2_hi = p.r_tol.bounds(p.r2)
    if full:
        specs = []
        for r1s, r2s, vas, ias in product(("lo", "hi"), repeat=4):
            r1 = r1_lo if r1s == "lo" else r1_hi
            r2 = r2_lo if r2s == "lo" else r2_hi
            va = va_lo if vas == "lo" else va_hi
            ia = ia_lo if ias == "lo" else ia_hi
            specs.append((f"R1{r1s}_R2{r2s}_Va{vas}_Ia{ias}", r1, r2, va, ia))
    else:
        specs = [
            ("R_lo_Va_lo", r1_lo, r2_hi, va_lo, ia_lo),
            ("R_hi_Va_hi", r1_hi, r2_lo, va_hi, ia_hi),
            ("R_lo_Va_hi", r1_lo, r2_hi, va_hi, ia_hi),
            ("R_hi_Va_lo", r1_hi, r2_lo, va_lo, ia_lo),
        ]
    workdir = Path(workdir)
    jobs = []
    for name, r1, r2, va, ia in specs:
        pc = p.with_values(r1=r1, r2=r2, vadj=va, iadj=ia)
        jobs.append({"name": name, "p": pc, "r1": r1, "r2": r2, "vadj": va, "iadj": ia})

    def _one(job: dict) -> dict:
        op = simulate_op(job["p"], workdir=workdir / job["name"], root=root)
        vout = op.nodes.get("out", float("nan"))
        an = _metrics(job["p"], vadj=job["vadj"], iadj=job["iadj"])
        return {
            "name": job["name"],
            "ok": True,
            "detail": f"Vout={vout:.4f}",
            "value": {
                "corner": job["name"],
                "vout_spice": vout,
                "vout_an": an["vout"],
                "headroom": an["headroom"] + (an["vout"] - vout),
                "p_diss": an["p_diss"],
                "efficiency": an["efficiency"],
                "r1": job["r1"],
                "r2": job["r2"],
                "vadj": job["vadj"],
                "iadj": job["iadj"],
            },
        }

    values, _rep = run_parallel(
        _one, jobs, max_workers=max_workers, desc="ngspice WC hypercube",
        engine="ngspice", name_of=lambda j: j["name"],
    )
    return [v for v in values if v is not None]
