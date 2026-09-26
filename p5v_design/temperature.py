"""Temperature sweep of setpoint, dropout, Ignd, and SPICE WC overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import envelope_bounds, vout_dc
from .ldo_model import dropout_v, ignd_a, tj_c, vadj_typical_vs_temp
from .params import CircuitParams


def temp_tag(temp_c: float) -> str:
    t = int(round(float(temp_c)))
    return f"Tm{abs(t)}" if t < 0 else f"Tp{t}"


def over_temp_at(temp_c: float, t0_c: float = 25.0) -> bool:
    return abs(float(temp_c) - float(t0_c)) > 1.0


@dataclass
class TempSweep:
    frame: pd.DataFrame

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)


def sweep_analytical(p: CircuitParams | None = None) -> TempSweep:
    if p is None:
        p = CircuitParams()
    rows = []
    for T in p.tempco.grid():
        pc = p.scaled_at_temp(T)
        va = vadj_typical_vs_temp(T, p.ldo)
        pc = pc.with_values(vadj=va)
        over = over_temp_at(T, p.ldo.t0_c)
        env = envelope_bounds(pc, over_temp=over)
        vout = vout_dc(pc)
        th = tj_c(pc, ta_c=T, which="typ")
        ig = ignd_a(pc, which="typ")
        iin = pc.op.iload_op + ig
        rows.append(
            {
                "temp_c": T,
                "vadj_typ": va,
                "r1": pc.r1,
                "r2": pc.r2,
                "vout": vout,
                "vout_wc_lo": env["min"],
                "vout_wc_hi": env["max"],
                "vdo_typ": dropout_v(pc, which="typ", temp_c=T),
                "vdo_wc": dropout_v(pc, which="max_ot"),
                "ignd_typ": ig,
                "ignd_max": ignd_a(pc, which="max"),
                "p_diss": th["p_tot"],
                "tj_c": th["tj_c"],
                "efficiency": (vout * pc.op.iload_op) / max(pc.op.vin_nom * iin, 1e-30),
            }
        )
    return TempSweep(frame=pd.DataFrame(rows))


def _temps(p: CircuitParams, temps: list[float] | None) -> list[float]:
    return list(p.tempco.grid() if temps is None else temps)


def sweep_ngspice_nominal(
    p: CircuitParams | None = None,
    *,
    workdir: Path | str,
    root: Path | str | None = None,
    temps: list[float] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Typical Vout vs T: TCR-scaled R, G05 Vadj(T), behavioral OP."""
    from .simulate import parse_op_log, simulate_op

    if p is None:
        p = CircuitParams()
    workdir = Path(workdir)
    rows = []
    for T in _temps(p, temps):
        tag = temp_tag(T)
        wd = workdir / tag / "nom"
        pc = p.scaled_at_temp(T)
        va = vadj_typical_vs_temp(T, p.ldo)
        pc = pc.with_values(vadj=va)
        log = wd / "ngspice.log"
        if cache and log.is_file():
            op = parse_op_log(log)
        else:
            op = simulate_op(pc, workdir=wd, root=root)
        vout = float(op.nodes.get("out", op.nodes.get("v(out)", float("nan"))))
        rows.append({"temp_c": float(T), "vout": vout, "vadj": va, "engine": "ngspice"})
    return pd.DataFrame(rows)


def sweep_ngspice_wc(
    p: CircuitParams | None = None,
    *,
    workdir: Path | str,
    root: Path | str | None = None,
    temps: list[float] | None = None,
    max_workers: int | None = None,
    full: bool = True,
) -> pd.DataFrame:
    """16-corner ngspice WC min/max vs T (same ADJ-box switch as analytical)."""
    from .worst_case import spice_wc_corners

    if p is None:
        p = CircuitParams()
    workdir = Path(workdir)
    rows = []
    for T in _temps(p, temps):
        pc = p.scaled_at_temp(T)
        over = over_temp_at(T, p.ldo.t0_c)
        corners = spice_wc_corners(
            pc, workdir=workdir / temp_tag(T) / "wc", root=root,
            max_workers=max_workers, over_temp=over, full=full,
        )
        vs = np.array(
            [c.get("vout_spice", float("nan")) for c in corners], dtype=float,
        )
        vs = vs[np.isfinite(vs)]
        an = vout_dc(pc.with_values(vadj=vadj_typical_vs_temp(T, p.ldo)))
        rows.append(
            {
                "temp_c": float(T),
                "vout_nom": float(an),
                "vout_lo": float(np.min(vs)) if vs.size else float("nan"),
                "vout_hi": float(np.max(vs)) if vs.size else float("nan"),
                "n_corners": int(vs.size),
                "engine": "ngspice",
            }
        )
    return pd.DataFrame(rows)


def reduce_ltspice_wc_log(log_path: Path | str) -> dict[str, float]:
    from .ltspice_io import meas_frame

    df = meas_frame(log_path, "vout_op", "vout")
    if df.empty:
        return {"vout_lo": float("nan"), "vout_hi": float("nan"), "n_corners": 0}
    v = df["vout"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return {
        "vout_lo": float(np.min(v)) if v.size else float("nan"),
        "vout_hi": float(np.max(v)) if v.size else float("nan"),
        "n_corners": int(v.size),
    }


def sweep_ltspice_from_logs(
    p: CircuitParams | None = None,
    *,
    temp_root: Path | str,
    temps: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse LTSpice/Temp/{tag}/ logs into typical and WC frames."""
    from .ltspice_io import parse_meas_log

    if p is None:
        p = CircuitParams()
    if temps is None:
        temps = [-55.0, 25.0, 125.0]
    temp_root = Path(temp_root)
    nom_rows = []
    wc_rows = []
    for T in temps:
        tag = temp_tag(T)
        d = temp_root / tag
        op_log = d / "P5V_Regulator_op.log"
        if not op_log.is_file():
            op_log = d / "P5V_Regulator_tran.log"
        vnom = float("nan")
        if op_log.is_file():
            try:
                m = parse_meas_log(op_log)
                vnom = float(m.get("vout_op", m.get("vout_end", float("nan"))))
            except Exception:
                vnom = float("nan")
        nom_rows.append({"temp_c": float(T), "vout": vnom, "engine": "LTspice"})
        wc_log = d / "P5V_Regulator_WC.log"
        red = reduce_ltspice_wc_log(wc_log) if wc_log.is_file() else {
            "vout_lo": float("nan"), "vout_hi": float("nan"), "n_corners": 0,
        }
        wc_rows.append({"temp_c": float(T), "engine": "LTspice", **red})
    return pd.DataFrame(nom_rows), pd.DataFrame(wc_rows)
