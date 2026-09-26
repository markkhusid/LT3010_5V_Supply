"""Cout/ESR window and Cff 10 kHz rule."""

from __future__ import annotations

import pandas as pd

from .ldo_model import cff_z_ohm, stability_checks
from .params import CircuitParams


def stability_table(p: CircuitParams | None = None) -> pd.DataFrame:
    if p is None:
        p = CircuitParams()
    s = stability_checks(p)
    rows = [
        ("Divider current", s["i_div_a"], "A", ">> Iadj"),
        ("Idiv / Iadj typ", s["i_div_over_iadj"], "", "> 100 recommended"),
        ("R2 < 250 kΩ", int(s["r2_lt_250k"]), "", "datasheet Iadj guideline"),
        ("Cout ≥ 1 µF", int(s["cout_ok"]), "", "min for stability"),
        ("ESR ≤ 3 Ω", int(s["esr_ok"]), "", "datasheet"),
        ("|Xcff| @ 10 kHz", s["cff_z_10khz"], "Ω", f"must be < R2 = {p.r2:.4g} Ω"),
        ("Cff rule", int(s["cff_rule_ok"]), "", s["note"]),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "unit", "note"])


def cff_impedance_sweep(p: CircuitParams | None = None):
    import numpy as np

    if p is None:
        p = CircuitParams()
    f = np.logspace(2, 6, 81)
    if p.cff <= 0.0:
        z = np.full_like(f, np.inf)
    else:
        z = 1.0 / (2.0 * np.pi * f * p.cff)
    return f, z, p.r2
