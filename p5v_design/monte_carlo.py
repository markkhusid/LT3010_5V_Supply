"""Uniform Monte Carlo for analytical metrics and optional ngspice subset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .params import CircuitParams
from .worst_case import _metrics


@dataclass
class MCResult:
    frame: pd.DataFrame
    summary: pd.DataFrame

    def save(self, dir_path: Path | str) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(d / "mc_samples.csv", index=False)
        self.summary.to_csv(d / "mc_summary.csv")
        meta = {"n": len(self.frame), "columns": list(self.frame.columns)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _draw_passives(p: CircuitParams, rng: np.random.Generator) -> dict[str, float]:
    out = {}
    for n, nom in p.passive_dict().items():
        if n == "Rload":
            continue
        out[n] = p.tol_for(n).scale(nom, rng.uniform())
    return out


def _draw_ldo(
    p: CircuitParams,
    rng: np.random.Generator,
    *,
    over_temp: bool = False,
) -> dict[str, float]:
    va_lo, va_hi = p.ldo.vadj_bounds(over_temp=over_temp)
    ia_lo, ia_hi = p.ldo.iadj_bounds(p.op.temp_c)
    return {
        "vadj": float(rng.uniform(va_lo, va_hi)),
        "iadj": float(rng.uniform(ia_lo, ia_hi)),
    }


def sample_metrics(
    p: CircuitParams,
    *,
    n: int = 2000,
    seed: int = 42,
    over_temp: bool = False,
    draw_temp: bool = False,
    include_ldo: bool = True,
) -> MCResult:
    """Uniform MC. ``draw_temp=True`` draws T in [Tmin, Tmax] then scales R."""
    rng = np.random.default_rng(seed)
    t_lo, t_hi = p.tempco.temp_min_c, p.tempco.temp_max_c
    rows = []
    for i in range(int(n)):
        if draw_temp:
            T = float(rng.uniform(t_lo, t_hi))
            base = p.scaled_at_temp(T)
            ot = abs(T - p.ldo.t0_c) > 1.0
        else:
            T = p.op.temp_c
            base = p
            ot = over_temp
        pas = _draw_passives(base, rng)
        pc = base.apply_passives(pas)
        ldo = _draw_ldo(base, rng, over_temp=ot) if include_ldo else {
            "vadj": p.vadj, "iadj": p.iadj,
        }
        pc = pc.with_values(vadj=ldo["vadj"], iadj=ldo["iadj"])
        m = _metrics(pc, vadj=ldo["vadj"], iadj=ldo["iadj"], mode="envelope")
        rows.append({"sample": i, "temp_c": T, **pas, **ldo, **m})
    frame = pd.DataFrame(rows)
    metric_cols = [c for c in ("vout", "headroom", "p_diss", "efficiency", "iin") if c in frame]
    summary = frame[metric_cols].agg(["mean", "std", "min", "max"]).T
    for q, name in ((0.01, "p01"), (0.05, "p05"), (0.50, "p50"), (0.95, "p95"), (0.99, "p99")):
        summary[name] = frame[metric_cols].quantile(q).values
    return MCResult(frame=frame, summary=summary)


def spice_monte_carlo(
    p: CircuitParams | None = None,
    *,
    n: int = 40,
    seed: int = 42,
    workdir: Path | str,
    root: Path | str | None = None,
    max_workers: int | None = None,
    over_temp: bool = False,
) -> MCResult:
    """Uniform MC of passives + Vadj/Iadj; each sample an ngspice OP."""
    from .parallel import run_parallel
    from .simulate import simulate_op

    if p is None:
        p = CircuitParams()
    rng = np.random.default_rng(seed)
    workdir = Path(workdir)
    jobs = []
    draws = []
    for i in range(int(n)):
        pas = _draw_passives(p, rng)
        ldo = _draw_ldo(p, rng, over_temp=over_temp)
        pc = p.apply_passives(pas).with_values(vadj=ldo["vadj"], iadj=ldo["iadj"])
        draws.append((i, pas, ldo, pc))
        jobs.append({"name": f"mc_{i:04d}", "idx": i, "p": pc})

    def _one(job: dict) -> dict:
        op = simulate_op(
            job["p"],
            workdir=workdir / job["name"],
            root=root,
        )
        vout = op.nodes.get("out", op.nodes.get("v(out)", float("nan")))
        return {
            "name": job["name"],
            "ok": True,
            "detail": f"Vout={vout:.4f}",
            "value": {"idx": job["idx"], "vout_spice": vout, **op.nodes},
        }

    values, _rep = run_parallel(
        _one, jobs, max_workers=max_workers, desc=f"ngspice MC n={n}",
        engine="ngspice", name_of=lambda j: j["name"],
    )
    rows = []
    for (i, pas, ldo, pc), val in zip(draws, values):
        m = _metrics(pc, vadj=ldo["vadj"], iadj=ldo["iadj"])
        row = {"sample": i, **pas, **ldo, **m}
        if val is not None:
            row["vout_spice"] = val.get("vout_spice", float("nan"))
        rows.append(row)
    frame = pd.DataFrame(rows)
    metric_cols = [c for c in ("vout", "vout_spice", "headroom") if c in frame]
    summary = frame[metric_cols].agg(["mean", "std", "min", "max"]).T
    return MCResult(frame=frame, summary=summary)
