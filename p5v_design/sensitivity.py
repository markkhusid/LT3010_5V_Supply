"""OAT and normalized sensitivity of Vout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .params import CircuitParams
from .worst_case import _metrics, oat_contributions


@dataclass
class SensitivityResult:
    oat: pd.DataFrame
    normalized: pd.DataFrame


def full_sensitivity(
    p: CircuitParams | None = None,
    *,
    over_temp: bool = False,
    rel_step: float = 0.001,
) -> SensitivityResult:
    if p is None:
        p = CircuitParams()
    oat = oat_contributions(p, metrics=("vout", "headroom", "p_diss"), over_temp=over_temp)
    rows = []
    for name, half in oat["vout"].items():
        rows.append({"name": name, "half_span_vout_mV": 1e3 * half, "half_span_vout": half})
    oat_df = pd.DataFrame(rows).sort_values("half_span_vout", ascending=False)
    rss = float(np.sqrt(np.sum(np.square(oat_df["half_span_vout"]))))
    oat_df["pct_of_rss"] = 100.0 * oat_df["half_span_vout"] / max(rss, 1e-30)

    nom = _metrics(p)["vout"]
    n_rows = []
    for field, nom_x in (("r1", p.r1), ("r2", p.r2), ("vadj", p.vadj), ("iadj", max(p.iadj, 1e-12))):
        dx = rel_step * nom_x
        if field == "r1":
            vp = _metrics(p.with_values(r1=nom_x + dx))["vout"]
            vm = _metrics(p.with_values(r1=nom_x - dx))["vout"]
        elif field == "r2":
            vp = _metrics(p.with_values(r2=nom_x + dx))["vout"]
            vm = _metrics(p.with_values(r2=nom_x - dx))["vout"]
        elif field == "vadj":
            vp = _metrics(p, vadj=nom_x + dx)["vout"]
            vm = _metrics(p, vadj=nom_x - dx)["vout"]
        else:
            vp = _metrics(p, iadj=nom_x + dx)["vout"]
            vm = _metrics(p, iadj=max(nom_x - dx, 0.0))["vout"]
        dVdx = 0.0 if dx == 0.0 else (vp - vm) / (2.0 * dx)
        n_rows.append(
            {
                "name": field,
                "dVout_dx": dVdx,
                "S": (nom_x / max(nom, 1e-30)) * dVdx,
            }
        )
    return SensitivityResult(oat=oat_df, normalized=pd.DataFrame(n_rows))
