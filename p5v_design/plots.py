"""Regulator plots. Every artist that is a callout takes ``annotations=``.

Annotation dict keys (all optional except ``text`` and ``xy``):

    text        str
    xy          (x, y) data coords of the arrow tip / anchor
    offset      (dx, dy) label in **points** from xy  — easiest to nudge
    xytext      (x, y) label in data coords (used if offset is omitted)
    arrow       bool, default True when a label position is given
    fontsize    default 8
    color       default '0.15'
    ha, va      default left / center
    bbox        bool, default True (white rounded box)
    fc, ec, alpha   box colors

Pass ``annotations=[]`` to draw the traces with no callouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .ldo_model import (
    COPPER_RTH_JA,
    dropout_v,
    ignd_a,
    noise_psd_uv_rtHz,
    psrr_db,
    soa_iout_max,
    tj_c,
)
from .params import CircuitParams


VOLT_TICK = FormatStrFormatter("%.3f")


def _is_volt_label(text: object) -> bool:
    """True for axes in volts, not millivolts / percent of Vmax."""
    if not text:
        return False
    s = str(text)
    if "mV" in s or "µV" in s or "uV" in s or "nV" in s:
        return False
    if "Vmax" in s or "V /" in s or "V/" in s:
        return False
    return "(V)" in s or s.strip().lower() in {"voltage", "voltage (v)", "v", "volts"}


def apply_voltage_format(ax) -> None:
    """Three digits after the decimal on voltage axes."""
    if _is_volt_label(ax.get_xlabel()):
        ax.xaxis.set_major_formatter(VOLT_TICK)
    if _is_volt_label(ax.get_ylabel()):
        ax.yaxis.set_major_formatter(VOLT_TICK)


def vfmt(v: float) -> str:
    return f"{float(v):.3f}"


def metric_1d(df: pd.DataFrame, name: str) -> np.ndarray:
    """1-D finite samples for a metric.

    Prefers ``vout_spice`` / ``vout_op`` when present so an ngspice frame
    that still has analytical ``vout`` is not plotted twice. Duplicate
    columns from ``rename(vout_spice→vout)`` use the last copy.
    """
    aliases = (name, f"{name}_spice", f"{name}_op")
    if name == "vout":
        aliases = ("vout_spice", "vout_op", "vout")
    s = None
    for cand in aliases:
        if cand in df.columns:
            s = df[cand]
            break
    if s is None:
        return np.asarray([], dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, -1]
    a = np.asarray(s, dtype=float).ravel()
    return a[np.isfinite(a)]


def annotate_ends(
    ax,
    x,
    y,
    *,
    fmt: str | None = None,
    left_offset: tuple[float, float] = (-6, 8),
    right_offset: tuple[float, float] = (6, 8),
    fontsize: int = 7,
) -> None:
    """Label the first and last samples of a trace."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = min(xa.size, ya.size)
    if n == 0:
        return
    f = fmt or "{:.3f}"
    apply_annotations(
        ax,
        [
            dict(text=f.format(ya[0]), xy=(float(xa[0]), float(ya[0])),
                 offset=left_offset, ha="right", fontsize=fontsize, arrow=False),
            dict(text=f.format(ya[n - 1]), xy=(float(xa[n - 1]), float(ya[n - 1])),
                 offset=right_offset, ha="left", fontsize=fontsize, arrow=False),
        ],
    )


def annotate_minmax(
    ax,
    x,
    y,
    *,
    fmt: str | None = None,
    fontsize: int = 7,
) -> None:
    """Label the min and max of a trace (undershoot / peak)."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = min(xa.size, ya.size)
    if n == 0:
        return
    xa, ya = xa[:n], ya[:n]
    i_lo = int(np.nanargmin(ya))
    i_hi = int(np.nanargmax(ya))
    f = fmt or "{:.3f}"
    items = [
        dict(text=f.format(ya[i_lo]), xy=(float(xa[i_lo]), float(ya[i_lo])),
             offset=(8, -14), ha="left", fontsize=fontsize, arrow=False),
    ]
    if i_hi != i_lo:
        items.append(
            dict(text=f.format(ya[i_hi]), xy=(float(xa[i_hi]), float(ya[i_hi])),
                 offset=(8, 10), ha="left", fontsize=fontsize, arrow=False)
        )
    apply_annotations(ax, items)


def _expand_degenerate_limits(ax) -> None:
    """A log axis with one finite value warns that the limits are identical."""
    for getter, setter in ((ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)):
        lo, hi = getter()
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            center = lo if np.isfinite(lo) else (hi if np.isfinite(hi) else 1.0)
            if center == 0:
                setter(-1.0, 1.0)
            else:
                setter(center / 10.0, center * 10.0)


def savefig(fig: plt.Figure, path: Path | str | None) -> plt.Figure:
    for ax in fig.axes:
        apply_voltage_format(ax)
        _expand_degenerate_limits(ax)
    try:
        fig.tight_layout()
    except Exception:
        pass
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def apply_annotations(ax, items: list[dict] | None) -> None:
    """Draw editable callouts. Safe no-op if ``items`` is None or empty."""
    if not items:
        return
    for it in items:
        text = it.get("text", "")
        xy = tuple(it["xy"])
        fontsize = it.get("fontsize", 8)
        color = it.get("color", "0.15")
        ha = it.get("ha", "left")
        va = it.get("va", "center")
        zorder = it.get("zorder", 8)
        bbox = None
        if it.get("bbox", True):
            bbox = dict(
                boxstyle=it.get("boxstyle", "round,pad=0.22"),
                fc=it.get("fc", "white"),
                ec=it.get("ec", "0.65"),
                alpha=it.get("alpha", 0.88),
                lw=0.6,
            )
        offset = it.get("offset")
        xytext = it.get("xytext")
        use_arrow = it.get("arrow", offset is not None or xytext is not None)
        arrowprops = None
        if use_arrow:
            arrowprops = dict(
                arrowstyle=it.get("arrowstyle", "->"),
                color=it.get("arrowcolor", "0.35"),
                lw=it.get("arrowlw", 0.85),
                shrinkA=0,
                shrinkB=2,
            )
        kw: dict[str, Any] = dict(
            fontsize=fontsize, color=color, ha=ha, va=va, zorder=zorder, bbox=bbox,
        )
        if offset is not None:
            ax.annotate(
                text, xy=xy, xytext=tuple(offset), textcoords="offset points",
                arrowprops=arrowprops, **kw,
            )
        elif xytext is not None:
            ax.annotate(text, xy=xy, xytext=tuple(xytext), arrowprops=arrowprops, **kw)
        else:
            ax.annotate(text, xy=xy, **kw)


def _grid(ax) -> None:
    ax.grid(True, which="both", alpha=0.3)


def plot_vout_vs_vin(
    vin: np.ndarray,
    vout: np.ndarray,
    *,
    vout_wc: np.ndarray | None = None,
    vin_nom: float = 9.0,
    vout_target: float = 5.0,
    vdo: float | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = r"$V_{\mathrm{OUT}}$ vs $V_{\mathrm{IN}}$",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(vin, vout, "k-", lw=1.8, label="analytical (typ dropout)")
    if vout_wc is not None:
        ax.plot(vin, vout_wc, "C3--", lw=1.3, label="max dropout (over temp)")
    ax.axhline(vout_target, color="C2", ls=":", lw=1.1, label=f"target {vout_target:.3f} V")
    ax.axvline(vin_nom, color="0.45", ls="--", lw=0.9, label=f"operate {vin_nom:.3f} V")
    if vdo is not None:
        ax.axvline(vout_target + vdo, color="C1", ls="-.", lw=1.0, label=f"Vin,min ≈ {vout_target + vdo:.3f} V")
    ax.set_xlabel(r"$V_{\mathrm{IN}}$ (V)")
    ax.set_ylabel(r"$V_{\mathrm{OUT}}$ (V)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    _grid(ax)
    annotate_ends(ax, vin, vout, fmt="{:.3f}")
    if vout_wc is not None:
        apply_annotations(ax, [
            dict(text=vfmt(vout_wc[-1]), xy=(float(vin[-1]), float(vout_wc[-1])),
                 offset=(6, -12), ha="left", fontsize=7, arrow=False),
        ])
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_vout_vs_iload(
    iload: np.ndarray,
    vout: np.ndarray,
    *,
    iload_op: float = 50e-3,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = r"$V_{\mathrm{OUT}}$ vs $I_{\mathrm{LOAD}}$ (load regulation)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(1e3 * iload, vout, "k-", lw=1.8)
    ax.axvline(1e3 * iload_op, color="C3", ls="--", lw=1.0, label=f"operate {1e3 * iload_op:.0f} mA")
    ax.set_xlabel(r"$I_{\mathrm{LOAD}}$ (mA)")
    ax.set_ylabel(r"$V_{\mathrm{OUT}}$ (V)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _grid(ax)
    annotate_ends(ax, 1e3 * iload, vout, fmt="{:.3f}")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_dropout_vs_i(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    i = np.linspace(0.5e-3, 50e-3, 200)
    typ = np.array([dropout_v(p, x, which="typ") for x in i])
    m25 = np.array([dropout_v(p, x, which="max_25") for x in i])
    mot = np.array([dropout_v(p, x, which="max_ot") for x in i])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(1e3 * i, 1e3 * typ, "k-", lw=1.8, label="typ 25 °C")
    ax.plot(1e3 * i, 1e3 * m25, "C0--", lw=1.4, label="max 25 °C")
    ax.plot(1e3 * i, 1e3 * mot, "C3-.", lw=1.5, label="max over temp")
    ax.axvline(1e3 * p.op.iload_op, color="0.45", ls="--", lw=0.9)
    ax.set_xlabel(r"$I_{\mathrm{LOAD}}$ (mA)")
    ax.set_ylabel("dropout (mV)")
    ax.set_title("LT3010 dropout (page 3 + interpolation)")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_ignd_vs_i(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    i = np.linspace(0.0, 50e-3, 200)
    typ = np.array([ignd_a(p, x, which="typ") for x in i])
    mx = np.array([ignd_a(p, x, which="max") for x in i])
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(1e3 * i, 1e3 * typ, "k-", lw=1.8, label="typ")
    ax.plot(1e3 * i, 1e3 * mx, "C3--", lw=1.4, label="max")
    ax.axvline(1e3 * p.op.iload_op, color="0.45", ls="--", lw=0.9)
    ax.set_xlabel(r"$I_{\mathrm{LOAD}}$ (mA)")
    ax.set_ylabel(r"$I_{\mathrm{GND}}$ (mA)")
    ax.set_title("GND pin current vs load (page 3)")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_psrr(
    p: CircuitParams,
    *,
    spice_f: np.ndarray | None = None,
    spice_db: np.ndarray | None = None,
    lt_f: np.ndarray | None = None,
    lt_db: np.ndarray | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    f = np.logspace(1, 6, 241)
    db = psrr_db(f, cout=p.cout)
    db_min = psrr_db(f, cout=p.cout, typ=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.semilogx(f, db, "k-", lw=1.8, label="datasheet G21 (Cout=10 µF class)")
    ax.semilogx(f, db_min, color="0.55", ls=":", lw=1.2, label="min (typ−10 dB)")
    if spice_f is not None and spice_db is not None:
        ax.semilogx(spice_f, spice_db, "C0--", lw=1.4, label="ngspice")
    if lt_f is not None and lt_db is not None:
        ax.semilogx(lt_f, lt_db, "C1:", lw=1.6, label="LTspice")
    ax.axvline(120.0, color="C3", ls="--", lw=0.9, label="120 Hz spec")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("PSRR (dB)")
    ax.set_title("Input ripple rejection")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_noise_psd(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    f = np.logspace(1, 5, 241)
    dens = noise_psd_uv_rtHz(f)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.loglog(f, dens, "k-", lw=1.8)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(r"e$_n$ (µV/$\sqrt{\mathrm{Hz}}$)")
    ax.set_title("Output noise spectral density (G24, scaled typical)")
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_error_budget(
    df: pd.DataFrame,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Vout error budget (OAT half-span)",
) -> plt.Figure:
    env = df[df["group"] == "envelope"].copy()
    fig, ax = plt.subplots(figsize=(8.4, max(3.2, 0.42 * len(env) + 1.4)))
    y = np.arange(len(env))
    ax.barh(y, env["half_span_mV"], color="#1f77b4", alpha=0.85, edgecolor="k", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(env["term"])
    ax.invert_yaxis()
    ax.set_xlabel("half-span (mV)")
    ax.set_title(title)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_error_budget_signed(
    df: pd.DataFrame,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, max(3.4, 0.45 * len(df) + 1.5)))
    y = np.arange(len(df))
    ax.barh(y, df["delta_lo_mV"], color="C0", alpha=0.85, edgecolor="k", lw=0.4, label="low")
    ax.barh(y, df["delta_hi_mV"], color="C3", alpha=0.75, edgecolor="k", lw=0.4, label="high")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["term"])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta V_{\mathrm{OUT}}$ from nominal (mV)")
    ax.set_title("Signed OAT contributions")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_tornado(
    contrib: dict[str, float],
    *,
    xlabel: str = "half-span (mV)",
    title: str = "Sensitivity tornado",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
    names = [k for k, _ in items]
    vals = [abs(v) for _, v in items]
    fig, ax = plt.subplots(figsize=(7.8, max(3.0, 0.34 * len(names) + 1.2)))
    y = np.arange(len(names))
    ax.barh(y, vals, color="#1f77b4", edgecolor="k", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_mc_histograms(
    frame: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("vout",),
    labels: dict[str, str] | None = None,
    nominal: dict[str, float] | None = None,
    wc: dict[str, tuple[float, float]] | None = None,
    rss: dict[str, tuple[float, float]] | None = None,
    spice: pd.DataFrame | None = None,
    vout_target: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    labels = labels or {"vout": r"$V_{\mathrm{OUT}}$ (V)", "headroom": "headroom (V)", "p_diss": "P (W)"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 3.9))
    if len(metrics) == 1:
        axes = [axes]
    for ax, met in zip(axes, metrics):
        data = frame[met].to_numpy(dtype=float)
        data = data[np.isfinite(data)]
        lo, hi = float(np.min(data)), float(np.max(data))
        if wc and met in wc:
            lo = min(lo, wc[met][0])
            hi = max(hi, wc[met][1])
        pad = 0.08 * (hi - lo) if hi > lo else 1e-3
        bins = np.linspace(lo - pad, hi + pad, 41)
        ax.hist(data, bins=bins, density=True, color="#1f77b4", alpha=0.7, edgecolor="k", lw=0.3, label="analytical MC")
        if spice is not None and met in spice.columns:
            ax.hist(
                spice[met].to_numpy(dtype=float), bins=bins, density=True,
                histtype="step", color="k", lw=1.4, label="ngspice MC",
            )
        if nominal and met in nominal:
            ax.axvline(nominal[met], color="k", ls="-", lw=1.2, label="nominal")
        if wc and met in wc:
            ax.axvspan(wc[met][0], wc[met][1], color="C3", alpha=0.08)
            ax.axvline(wc[met][0], color="C3", ls="--", lw=1.3, label="WC")
            ax.axvline(wc[met][1], color="C3", ls="--", lw=1.3)
        if rss and met in rss:
            ax.axvline(rss[met][0], color="C2", ls="-.", lw=1.2, label="RSS")
            ax.axvline(rss[met][1], color="C2", ls="-.", lw=1.2)
        if met == "vout" and vout_target is not None:
            ax.axvline(vout_target, color="C4", ls=":", lw=1.4, label=f"{vout_target:.3f} V")
        if wc and met in wc and met in ("vout", "headroom"):
            ax.annotate(vfmt(wc[met][0]), xy=(wc[met][0], 0), xytext=(-4, 18),
                        textcoords="offset points", ha="right", fontsize=7)
            ax.annotate(vfmt(wc[met][1]), xy=(wc[met][1], 0), xytext=(4, 18),
                        textcoords="offset points", ha="left", fontsize=7)
        ax.set_xlabel(labels.get(met, met))
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        _grid(ax)
        apply_annotations(ax, annotations if len(metrics) == 1 else None)
    if title:
        fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    return savefig(fig, path)


def plot_soa(
    p: CircuitParams,
    *,
    ta_c: float = 25.0,
    rth_ja: float | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    vin = np.linspace(5.5, 30.0, 80)
    i_max = np.array([soa_iout_max(p, float(v), ta_c=ta_c, rth_ja=rth_ja) for v in vin])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.fill_between(vin, 0.0, 1e3 * i_max, color="C2", alpha=0.18, label="Tj ≤ 125 °C")
    ax.plot(vin, 1e3 * i_max, "k-", lw=1.8)
    ax.axhline(1e3 * p.op.iload_op, color="C3", ls="--", lw=1.1, label=f"operate {1e3 * p.op.iload_op:.0f} mA")
    ax.axvline(p.op.vin_nom, color="0.45", ls="--", lw=0.9)
    ax.set_xlabel(r"$V_{\mathrm{IN}}$ (V)")
    ax.set_ylabel(r"$I_{\mathrm{OUT,max}}$ (mA)")
    ax.set_title(f"Thermal SOA  Ta={ta_c:.0f} °C  θJA={(rth_ja or p.ldo.rth_ja):.0f} °C/W")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_tj_map(
    p: CircuitParams,
    *,
    ta_c: float = 25.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    vin = np.linspace(6.0, 20.0, 60)
    i = np.linspace(1e-3, 50e-3, 50)
    V, I = np.meshgrid(vin, i)
    tj = np.empty_like(V)
    for r in range(I.shape[0]):
        for c in range(I.shape[1]):
            tj[r, c] = tj_c(p, vin=float(V[r, c]), iload=float(I[r, c]), ta_c=ta_c)["tj_c"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cf = ax.contourf(V, 1e3 * I, tj, levels=16, cmap="magma")
    cs = ax.contour(V, 1e3 * I, tj, levels=[p.ldo.tj_max_c], colors="w", linewidths=1.6)
    ax.clabel(cs, fmt="Tjmax", fontsize=8)
    fig.colorbar(cf, ax=ax, label="Tj (°C)")
    ax.plot(p.op.vin_nom, 1e3 * p.op.iload_op, "wo", ms=8, mew=1.2, mec="k")
    ax.set_xlabel(r"$V_{\mathrm{IN}}$ (V)")
    ax.set_ylabel(r"$I_{\mathrm{OUT}}$ (mA)")
    ax.set_title(f"Junction temperature map  Ta={ta_c:.0f} °C")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_efficiency(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    vin = np.linspace(5.5, 16.0, 80)
    i = p.op.iload_op
    rows = []
    from .ldo_model import vout_dc
    vout = vout_dc(p)
    for v in vin:
        ig = ignd_a(p, i, which="typ")
        eta = (vout * i) / max(v * (i + ig), 1e-30)
        rows.append((v, eta))
    vin_a, eta = np.array(rows).T
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(vin_a, 100.0 * eta, "k-", lw=1.8)
    ax.axvline(p.op.vin_nom, color="C3", ls="--", lw=1.0)
    ax.set_xlabel(r"$V_{\mathrm{IN}}$ (V)")
    ax.set_ylabel("efficiency (%)")
    ax.set_title(f"Linear-regulator efficiency at {1e3 * i:.0f} mA")
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_copper_rth(
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    area = [r[0] for r in COPPER_RTH_JA]
    rth = [r[3] for r in COPPER_RTH_JA]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(area, rth, "o-", color="k", lw=1.6, ms=7)
    ax.set_xlabel("topside copper (mm²)")
    ax.set_ylabel("θJA (°C/W)")
    ax.set_title("Datasheet Table 1 — MS8E on 3/32 in FR-4, 1 oz, still air")
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_temp_sweep(
    temps: np.ndarray,
    vout: np.ndarray,
    *,
    vout_lo: np.ndarray | None = None,
    vout_hi: np.ndarray | None = None,
    ng_temps: np.ndarray | None = None,
    ng_vout: np.ndarray | None = None,
    ng_lo: np.ndarray | None = None,
    ng_hi: np.ndarray | None = None,
    lt_temps: np.ndarray | None = None,
    lt_vout: np.ndarray | None = None,
    lt_lo: np.ndarray | None = None,
    lt_hi: np.ndarray | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = r"$V_{\mathrm{OUT}}$ vs temperature",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(temps, vout, "k-", lw=1.8, label="analytical typical")
    if vout_lo is not None and vout_hi is not None:
        ax.fill_between(temps, vout_lo, vout_hi, color="C3", alpha=0.14, label="analytical WC")
        ax.plot(temps, vout_lo, "C3--", lw=1.1)
        ax.plot(temps, vout_hi, "C3--", lw=1.1)
    if ng_temps is not None and ng_vout is not None:
        ax.plot(ng_temps, ng_vout, "C0--", lw=1.5, label="ngspice typical")
    if ng_temps is not None and ng_lo is not None and ng_hi is not None:
        ax.plot(ng_temps, ng_lo, "C0:", lw=1.1)
        ax.plot(ng_temps, ng_hi, "C0:", lw=1.1, label="ngspice WC")
    if lt_temps is not None and lt_vout is not None:
        yerr = None
        if lt_lo is not None and lt_hi is not None:
            lo = np.asarray(lt_lo, dtype=float)
            hi = np.asarray(lt_hi, dtype=float)
            mid = np.asarray(lt_vout, dtype=float)
            yerr = np.vstack([mid - lo, hi - mid])
        ax.errorbar(
            lt_temps, lt_vout, yerr=yerr, fmt="C1o", ms=6, capsize=4, lw=1.3,
            label="LTspice typical + WC", zorder=6,
        )
    ax.axhline(5.0, color="C2", ls=":", lw=1.1, label="5.000 V target")
    ax.set_xlabel("T (°C)")
    ax.set_ylabel(r"$V_{\mathrm{OUT}}$ (V)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    _grid(ax)
    annotate_ends(ax, temps, vout, fmt="{:.3f}")
    if vout_lo is not None and vout_hi is not None:
        apply_annotations(ax, [
            dict(text=vfmt(vout_lo[-1]), xy=(float(temps[-1]), float(vout_lo[-1])),
                 offset=(6, -12), ha="left", fontsize=7, arrow=False),
            dict(text=vfmt(vout_hi[-1]), xy=(float(temps[-1]), float(vout_hi[-1])),
                 offset=(6, 8), ha="left", fontsize=7, arrow=False),
        ])
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_tran(
    t: np.ndarray,
    vout: np.ndarray,
    *,
    vin: np.ndarray | None = None,
    iout: np.ndarray | None = None,
    spice_t: np.ndarray | None = None,
    spice_v: np.ndarray | None = None,
    lt_t: np.ndarray | None = None,
    lt_v: np.ndarray | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Transient",
    t_scale: float = 1e3,
    t_label: str = "t (ms)",
) -> plt.Figure:
    n = 2 if (vin is not None or iout is not None) else 1
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    axes[0].plot(t_scale * t, vout, "k-", lw=1.7, label="analytical / expected")
    if spice_t is not None and spice_v is not None:
        axes[0].plot(t_scale * spice_t, spice_v, "C0--", lw=1.3, label="ngspice")
    if lt_t is not None and lt_v is not None:
        axes[0].plot(t_scale * lt_t, lt_v, "C1:", lw=1.5, label="LTspice")
    axes[0].set_ylabel(r"$V_{\mathrm{OUT}}$ (V)")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    _grid(axes[0])
    annotate_minmax(axes[0], t_scale * t, vout, fmt="{:.3f}")
    apply_annotations(axes[0], annotations)
    if n == 2:
        ax2 = axes[1]
        if iout is not None:
            ax2.plot(t_scale * t, 1e3 * iout, "C4-", lw=1.5, label="Iout")
            ax2.set_ylabel("I (mA)")
        if vin is not None:
            ax2.plot(t_scale * t, vin, "C5-", lw=1.3, label="Vin")
            ax2.set_ylabel("V / mA")
        ax2.legend(fontsize=8)
        _grid(ax2)
    axes[-1].set_xlabel(t_label)
    fig.tight_layout()
    return savefig(fig, path)


def plot_overlay_dc(
    x: np.ndarray,
    y_an: np.ndarray,
    y_sp: np.ndarray | None = None,
    y_lt: np.ndarray | None = None,
    *,
    xlabel: str = "x",
    ylabel: str = "Vout (V)",
    title: str = "DC overlay",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)
    axes[0].plot(x, y_an, "k-", lw=1.8, label="analytical")
    if y_sp is not None:
        axes[0].plot(x, y_sp, "C0--", lw=1.3, label="ngspice")
        if len(y_sp) == len(y_an):
            axes[1].plot(x, 1e3 * (y_sp - y_an), "C0-")
    if y_lt is not None:
        axes[0].plot(x, y_lt, "C1:", lw=1.5, label="LTspice")
        if len(y_lt) == len(y_an):
            axes[1].plot(x, 1e3 * (y_lt - y_an), "C1-")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("error (mV)")
    axes[1].set_xlabel(xlabel)
    _grid(axes[0])
    _grid(axes[1])
    annotate_ends(axes[0], x, y_an, fmt="{:.3f}")
    apply_annotations(axes[0], annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_wc_span_compare(
    rows: list[dict],
    *,
    metric: str = "vout",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "WC span comparison",
    target: float | None = 5.0,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    names = [r["name"] for r in rows]
    lo = [float(r["min"]) for r in rows]
    hi = [float(r["max"]) for r in rows]
    nom = [float(r.get("nominal", 0.5 * (r["min"] + r["max"]))) for r in rows]
    y = np.arange(len(names))
    ax.hlines(y, lo, hi, color="C3", lw=4, alpha=0.7)
    ax.plot(nom, y, "ko", ms=7, zorder=5, label="nominal")
    for yi, l, h, n in zip(y, lo, hi, nom):
        ax.plot([l, h], [yi, yi], "|", color="0.15", ms=12, mew=1.4, zorder=6)
        ax.annotate(vfmt(l), xy=(l, yi), xytext=(-7, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color="0.15")
        ax.annotate(vfmt(h), xy=(h, yi), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color="0.15")
        ax.annotate(vfmt(n), xy=(n, yi), xytext=(0, -13), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="0.2")
    if target is not None:
        ax.axvline(target, color="#9467bd", ls=":", lw=1.4, label=f"{target:.3f} V")
    span = max(hi) - min(lo) if hi else 1.0
    pad = max(0.12 * span, 0.02)
    ax.set_xlim(min(lo) - pad, max(hi) + pad)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    xlab = metric if "(" in str(metric) or str(metric).endswith("V") else str(metric)
    if xlab in ("vout", "Vout"):
        xlab = r"$V_{\mathrm{OUT}}$ (V)"
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_headroom_bar(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    from .analysis import headroom, vout_dc

    vout = vout_dc(p)
    h = headroom(p, which="max_ot")
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    segs = [
        (0.0, vout, "#1f77b4", r"$V_{\mathrm{OUT}}$"),
        (vout, vout + h["vdo"], "#ff7f0e", r"$V_{\mathrm{DO,wc}}$"),
        (vout + h["vdo"], p.op.vin_nom, "#2ca02c", "headroom"),
    ]
    for x0, x1, c, lab in segs:
        ax.barh(0, x1 - x0, left=x0, color=c, edgecolor="k", height=0.45, label=lab)
        ax.annotate(vfmt(x1), xy=(x1, 0), xytext=(0, 14), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.annotate(vfmt(0.0), xy=(0.0, 0), xytext=(0, 14), textcoords="offset points",
                ha="left", va="bottom", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("voltage (V)")
    ax.set_title(f"Headroom stack at Vin={p.op.vin_nom:.3f} V, {1e3 * p.op.iload_op:.0f} mA")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def _prep_stress(
    df: pd.DataFrame,
    value_col: str,
    name_col: str,
    *,
    dropna: bool = True,
    exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    d = df.copy()
    if name_col not in d.columns and "component" in d.columns:
        d[name_col] = d["component"]
    if exclude and name_col in d.columns:
        d = d[~d[name_col].isin(list(exclude))]
    if dropna and value_col in d.columns:
        d = d[d[value_col].notna()]
    return d


def plot_stress_bars(
    df: pd.DataFrame,
    *,
    value_col: str,
    xlabel: str,
    title: str,
    vline: float | None = None,
    vline_label: str | None = None,
    vline2: float | None = None,
    vline2_label: str | None = None,
    color: str = "C4",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    name_col: str = "ref",
    dropna: bool = True,
    exclude: Sequence[str] | None = None,
    show_values: bool = False,
) -> plt.Figure:
    """Comparator-style horizontal stress bars (one operating condition)."""
    d = _prep_stress(df, value_col, name_col, dropna=dropna, exclude=exclude)
    if name_col not in d.columns:
        name_col = "component" if "component" in d.columns else d.columns[0]
    d = d.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, max(3.6, 0.38 * max(len(d), 1) + 1.2)))
    ax.barh(d[name_col], d[value_col], color=color, alpha=0.85, edgecolor="k", lw=0.4)
    if vline is not None:
        ax.axvline(vline, color="C3", ls="--", lw=1.5, label=vline_label or f"{vline:g}")
    if vline2 is not None:
        ax.axvline(vline2, color="0.4", ls=":", lw=1.3, label=vline2_label or f"{vline2:g}")
    if vline is not None or vline2 is not None:
        ax.legend(fontsize=8)
    if show_values:
        for y, v in zip(d[name_col], d[value_col]):
            if v == v:
                ax.text(v, y, f"  {v:.3g}", va="center", fontsize=7, color="0.2")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_stress_grouped(
    tables: dict[str, pd.DataFrame],
    *,
    value_col: str,
    xlabel: str,
    title: str,
    vline: float | None = None,
    vline_label: str | None = None,
    vline2: float | None = None,
    vline2_label: str | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    name_col: str = "ref",
    exclude: Sequence[str] | None = None,
    dropna: bool = True,
) -> plt.Figure:
    """Grouped horizontal bars — one color per operating condition (comparator 06)."""
    cleaned: dict[str, pd.DataFrame] = {}
    for key, df in tables.items():
        cleaned[key] = _prep_stress(df, value_col, name_col, dropna=dropna, exclude=exclude)
    first = next(iter(cleaned.values()))
    if name_col not in first.columns:
        name_col = "component" if "component" in first.columns else first.columns[0]
    labels = first.sort_values(value_col, ascending=True)[name_col].tolist()
    for df in cleaned.values():
        for n in df[name_col].tolist():
            if n not in labels:
                labels.append(n)
    fig, ax = plt.subplots(figsize=(11.0, max(4.8, 0.36 * max(len(labels), 1) + 1.4)))
    y = np.arange(len(labels))
    n = len(cleaned)
    h = min(0.28, 0.8 / max(n, 1))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (key, df) in enumerate(cleaned.items()):
        m = dict(zip(df[name_col], df[value_col]))
        vals = [m.get(c, np.nan) for c in labels]
        ax.barh(y + (i - 0.5 * (n - 1)) * h, vals, height=h, color=colors[i % len(colors)],
                alpha=0.88, edgecolor="k", lw=0.3, label=key)
    if vline is not None:
        ax.axvline(vline, color="0.15", ls="--", lw=1.4, label=vline_label or f"{vline:g}")
    if vline2 is not None:
        ax.axvline(vline2, color="0.45", ls=":", lw=1.3, label=vline2_label or f"{vline2:g}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_applied_voltage(
    df: pd.DataFrame,
    *,
    name_col: str = "ref",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Applied voltage vs 80% derated / catalog Vmax",
    exclude: Sequence[str] | None = None,
) -> plt.Figure:
    """Comparator 06 applied-voltage bars with derated and catalog ticks."""
    d = _prep_stress(df, "V_applied_V", name_col, dropna=True, exclude=exclude)
    if "V_max_V" in d.columns:
        d = d[d["V_max_V"] > 0]
    d = d.sort_values("V_applied_V", ascending=True)
    fig, ax = plt.subplots(figsize=(9.4, max(3.6, 0.4 * max(len(d), 1) + 1.2)))
    ax.barh(d[name_col], d["V_applied_V"], color="C1", alpha=0.85, edgecolor="k", lw=0.4,
            label="applied")
    if "V_derated_V" in d.columns:
        ax.plot(d["V_derated_V"], d[name_col], "k|", ms=14, mew=1.6, label="80% derated")
    if "V_max_V" in d.columns:
        ax.plot(d["V_max_V"], d[name_col], color="0.45", marker="|", ls="none",
                ms=14, mew=1.4, label="catalog Vmax")
    ax.set_xlabel("voltage (V)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_applied_voltage_grouped(
    tables: dict[str, pd.DataFrame],
    *,
    name_col: str = "ref",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Applied voltage vs operating condition",
    exclude: Sequence[str] | None = None,
) -> plt.Figure:
    """Grouped V_applied with 80% derated ticks from the last table (comparator 06)."""
    cleaned = {
        k: _prep_stress(df, "V_applied_V", name_col, dropna=True, exclude=exclude)
        for k, df in tables.items()
    }
    first = next(iter(cleaned.values()))
    if name_col not in first.columns:
        name_col = "component" if "component" in first.columns else first.columns[0]
    labels = first.sort_values("V_applied_V", ascending=True)[name_col].tolist()
    fig, ax = plt.subplots(figsize=(11.0, max(4.8, 0.38 * max(len(labels), 1) + 1.4)))
    y = np.arange(len(labels))
    n = len(cleaned)
    h = min(0.28, 0.8 / max(n, 1))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    last = None
    for i, (key, df) in enumerate(cleaned.items()):
        m = dict(zip(df[name_col], df["V_applied_V"]))
        vals = [m.get(c, np.nan) for c in labels]
        ax.barh(y + (i - 0.5 * (n - 1)) * h, vals, height=h, color=colors[i % len(colors)],
                alpha=0.88, edgecolor="k", lw=0.3, label=key)
        last = df
    if last is not None and "V_derated_V" in last.columns:
        vder = dict(zip(last[name_col], last["V_derated_V"]))
        ax.plot([vder.get(c, np.nan) for c in labels], y, "k|", ms=12, mew=1.5,
                label="80% derated")
    if last is not None and "V_max_V" in last.columns:
        vmax = dict(zip(last[name_col], last["V_max_V"]))
        ax.plot([vmax.get(c, np.nan) for c in labels], y, color="0.45", marker="|",
                ls="none", ms=12, mew=1.3, label="catalog Vmax")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("voltage (V)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_regulation_family(
    p: CircuitParams,
    *,
    which: str = "vin",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Vout vs Vin at several loads, or Vout vs Iload at several Vin."""
    from dataclasses import replace

    from .analysis import vout_vs_vin, vout_vs_iload

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if which == "vin":
        for i, lab in ((1e-3, "1 mA"), (10e-3, "10 mA"), (50e-3, "50 mA")):
            pc = replace(p, op=replace(p.op, iload_op=i))
            vin, vo = vout_vs_vin(pc)
            ax.plot(vin, vo, lw=1.6, label=lab)
        ax.set_xlabel(r"$V_{\mathrm{IN}}$ (V)")
        ax.set_title(r"$V_{\mathrm{OUT}}$ vs $V_{\mathrm{IN}}$ family")
    else:
        for v, lab in ((8.0, "8 V"), (9.0, "9 V"), (12.0, "12 V")):
            pc = replace(p, op=replace(p.op, vin_nom=v))
            i, vo = vout_vs_iload(pc)
            ax.plot(1e3 * i, vo, lw=1.6, label=lab)
        ax.set_xlabel(r"$I_{\mathrm{LOAD}}$ (mA)")
        ax.set_title(r"$V_{\mathrm{OUT}}$ vs $I_{\mathrm{LOAD}}$ family")
    ax.set_ylabel(r"$V_{\mathrm{OUT}}$ (V)")
    ax.axhline(5.0, color="0.4", ls=":", lw=1.0, label="5.000 V")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_temp_grid(
    frame,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    have_extra = all(c in frame.columns for c in ("tj_c", "p_diss", "efficiency"))
    nrows, ncols = (2, 3) if have_extra else (2, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.4 if have_extra else 10.6, 6.4), sharex=True)
    t = frame["temp_c"].to_numpy(dtype=float)
    specs = [
        (0, 0, "vout", r"$V_{\mathrm{OUT}}$ (V)"),
        (0, 1, "vadj_typ", r"$V_{\mathrm{ADJ}}$ typ (V)"),
        (1, 0, "vdo_typ", r"$V_{\mathrm{DO}}$ typ (V)"),
        (1, 1, "ignd_typ", r"$I_{\mathrm{GND}}$ typ (A)"),
    ]
    if have_extra:
        specs.extend([
            (0, 2, "efficiency", "efficiency"),
            (1, 2, "tj_c", r"$T_j$ (°C)"),
        ])
    for r, c, col, ylab in specs:
        ax = axes[r, c]
        y = frame[col].to_numpy(dtype=float)
        if col == "efficiency":
            y = 100.0 * y
            ylab = "efficiency (%)"
        elif col == "ignd_typ":
            y = 1e3 * y
            ylab = r"$I_{\mathrm{GND}}$ (mA)"
        ax.plot(t, y, "k-o", ms=4, lw=1.4)
        if col == "vout" and "vout_wc_lo" in frame.columns:
            ax.fill_between(t, frame["vout_wc_lo"], frame["vout_wc_hi"], color="C3", alpha=0.15)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.axvline(25.0, color="0.6", ls=":", lw=0.9)
    for ax in axes[nrows - 1, :]:
        ax.set_xlabel("T (°C)")
    fig.suptitle("Temperature family (MP −55…125 °C)", y=1.01, fontsize=12)
    apply_annotations(axes[0, 0], annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_oat_share(
    oat_df,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    d = oat_df.sort_values("pct_of_rss", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.4 * len(d) + 1.0)))
    ax.barh(d["name"], d["pct_of_rss"], color="C0", edgecolor="k", lw=0.4)
    ax.set_xlabel("share of RSS (%)")
    ax.set_title(r"OAT share of $V_{\mathrm{OUT}}$ RSS")
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_tornado_compare(
    a: dict[str, float],
    b: dict[str, float],
    *,
    label_a: str = "±1 %",
    label_b: str = "±0.1 %",
    xlabel: str = "half-span (mV)",
    title: str = "Sensitivity: 1 % vs 0.1 % resistors",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    names = list(dict.fromkeys(list(a) + list(b)))
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.4 * len(names) + 1.0)))
    ax.barh(y + 0.18, [1e3 * a.get(n, 0.0) for n in names], height=0.34, color="C0", edgecolor="k", lw=0.3, label=label_a)
    ax.barh(y - 0.18, [1e3 * b.get(n, 0.0) for n in names], height=0.34, color="C3", edgecolor="k", lw=0.3, label=label_b)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_rail_stack(
    budget,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Stacked Iin = Iout + Ignd and Pin = Pout + Pdiss."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    axes[0].barh(0, 1e3 * budget["iload"], color="C0", edgecolor="k", height=0.45, label="Iout")
    axes[0].barh(0, 1e3 * budget["ignd"], left=1e3 * budget["iload"], color="C3", edgecolor="k", height=0.45, label="Ignd")
    axes[0].set_xlabel("mA")
    axes[0].set_yticks([])
    axes[0].set_title(r"$I_{\mathrm{IN}}$ stack")
    axes[0].legend(fontsize=8)
    axes[1].barh(0, budget["p_out"], color="C2", edgecolor="k", height=0.45, label="Pout")
    axes[1].barh(0, budget["p_diss"], left=budget["p_out"], color="C1", edgecolor="k", height=0.45, label="Pdiss")
    axes[1].set_xlabel("W")
    axes[1].set_yticks([])
    axes[1].set_title(r"$P_{\mathrm{IN}}$ stack")
    axes[1].legend(fontsize=8)
    apply_annotations(axes[0], annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_ceramic_dc_bias(
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Datasheet Figs. 3–4 style: X7R vs Y5V remaining C vs DC bias (16 V 10 µF class)."""
    v = np.linspace(0.0, 16.0, 81)
    # Digitized typical remaining fraction
    x7r = np.clip(1.0 - 0.025 * v - 0.0012 * v**2, 0.35, 1.05)
    y5v = np.clip(np.exp(-0.22 * v), 0.05, 1.05)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(v, 100 * x7r, "k-", lw=1.8, label="X7R / X5R class")
    ax.plot(v, 100 * y5v, "C3--", lw=1.6, label="Y5V / Z5U class")
    ax.axvline(5.0, color="C0", ls=":", lw=1.2, label="5 V Cout bias")
    ax.set_xlabel("DC bias (V)")
    ax.set_ylabel("remaining C (%)")
    ax.set_title("Ceramic DC-bias (datasheet Figs. 3–4, qualitative)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_error_db(
    f_hz: np.ndarray,
    err: np.ndarray,
    *,
    ylabel: str = "Δ (dB)",
    title: str = "SPICE − datasheet",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """HV_monitor / Opamp residual plot (frequency-domain error)."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.semilogx(f_hz, err, "C0-")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_op_compare_bars(
    rows: dict[str, dict[str, float]],
    *,
    metrics: tuple[str, ...] = ("vout",),
    units: dict[str, str] | None = None,
    scale: dict[str, float] | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Operating-point comparison",
) -> plt.Figure:
    """Grouped bars: one group per metric, one color per engine (HV_monitor 05)."""
    units = units or {"vout": "V"}
    scale = scale or {}
    engines = list(rows)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(metrics) + 2), 4.0))
    x = np.arange(len(metrics), dtype=float)
    w = 0.8 / max(len(engines), 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, eng in enumerate(engines):
        vals = [scale.get(m, 1.0) * rows[eng].get(m, float("nan")) for m in metrics]
        ax.bar(x + (i - 0.5 * (len(engines) - 1)) * w, vals, width=w, color=colors[i % 3],
               edgecolor="k", lw=0.4, label=eng)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}" + (f"\n({units[m]})" if m in units else "") for m in metrics])
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_ilim_margin(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Operate load vs current-limit min/typ (datasheet page 3)."""
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    labels = ["operate Iload", "Ilim min (over temp)", "Ilim typ"]
    vals = [1e3 * p.op.iload_op, 1e3 * p.ldo.ilim_min_ot_a, 1e3 * p.ldo.ilim_typ_a]
    colors = ["C0", "C3", "C2"]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor="k", lw=0.4)
    ax.set_xlabel("mA")
    ax.set_title("Current-limit margin")
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_psrr_family(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Datasheet G21 family: 1 µF vs 10 µF Cout."""
    f = np.logspace(1, 6, 241)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.semilogx(f, psrr_db(f, cout=1e-6), "C0--", lw=1.5, label="Cout = 1 µF")
    ax.semilogx(f, psrr_db(f, cout=10e-6), "k-", lw=1.8, label="Cout = 10 µF")
    ax.semilogx(f, psrr_db(f, cout=p.cout, typ=False), color="0.55", ls=":", lw=1.2,
                label="min (typ−10 dB), 10 µF class")
    ax.axvline(120.0, color="C3", ls="--", lw=0.9, label="120 Hz spec")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("PSRR (dB)")
    ax.set_title("PSRR family (datasheet G21)")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_noise_rms_vs_bw(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Integrated output noise vs bandwidth from the G24 PSD."""
    f = np.logspace(1, 5, 400)
    dens = noise_psd_uv_rtHz(f) * 1e-6  # V/√Hz
    # cumulative RMS from 10 Hz
    df = np.diff(f, prepend=f[0])
    rms = np.sqrt(np.cumsum((dens ** 2) * df))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.semilogx(f, 1e6 * rms, "k-", lw=1.8)
    ax.set_xlabel("bandwidth (Hz)")
    ax.set_ylabel("e_n,rms (µV)")
    ax.set_title("Integrated output noise vs bandwidth (G24 PSD)")
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_wc_vout_vs_vin(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Opamp 04 analog: WC envelope as a band on the Vin sweep."""
    from .analysis import envelope_bounds, vout_vs_vin

    vin, v_typ = vout_vs_vin(p, which_do="typ")
    _, v_do = vout_vs_vin(p, which_do="max_ot")
    env = envelope_bounds(p, over_temp=False)
    env_ot = envelope_bounds(p, over_temp=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.fill_between(vin, env_ot["min"], env_ot["max"], color="C3", alpha=0.12,
                    label="over-temp envelope")
    ax.fill_between(vin, env["min"], env["max"], color="C0", alpha=0.18,
                    label="25 °C envelope")
    ax.plot(vin, v_typ, "k-", lw=1.8, label="typ dropout")
    ax.plot(vin, v_do, "C3--", lw=1.3, label="max dropout")
    ax.axhline(5.0, color="C2", ls=":", lw=1.1, label="5.000 V goal")
    ax.axvline(p.op.vin_nom, color="0.45", ls="--", lw=0.9)
    ax.set_xlabel(r"$V_{\mathrm{IN}}$ (V)")
    ax.set_ylabel(r"$V_{\mathrm{OUT}}$ (V)")
    ax.set_title("WC envelope on the Vin sweep")
    ax.legend(fontsize=8, loc="lower right")
    _grid(ax)
    apply_annotations(ax, [
        dict(text=f"{env['min']:.3f}", xy=(float(vin[-1]), env["min"]),
             offset=(6, -10), ha="left", fontsize=7, arrow=False),
        dict(text=f"{env['max']:.3f}", xy=(float(vin[-1]), env["max"]),
             offset=(6, 8), ha="left", fontsize=7, arrow=False),
        dict(text=f"{env_ot['min']:.3f}", xy=(float(vin[0]), env_ot["min"]),
             offset=(-6, -10), ha="right", fontsize=7, arrow=False),
        dict(text=f"{env_ot['max']:.3f}", xy=(float(vin[0]), env_ot["max"]),
             offset=(-6, 8), ha="right", fontsize=7, arrow=False),
    ])
    annotate_ends(ax, vin, v_typ, fmt="{:.3f}")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_efficiency_vs_load(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    from .analysis import vout_dc

    i = np.linspace(1e-3, 50e-3, 80)
    vout = vout_dc(p)
    eta = []
    for ii in i:
        ig = ignd_a(p, float(ii), which="typ")
        eta.append((vout * ii) / max(p.op.vin_nom * (ii + ig), 1e-30))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(1e3 * i, 100.0 * np.array(eta), "k-", lw=1.8)
    ax.axvline(1e3 * p.op.iload_op, color="C3", ls="--", lw=1.0)
    ax.set_xlabel(r"$I_{\mathrm{LOAD}}$ (mA)")
    ax.set_ylabel("efficiency (%)")
    ax.set_title(f"Efficiency vs load at Vin={p.op.vin_nom:.3g} V")
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_budget_grouped(
    typ: dict[str, float],
    mx: dict[str, float],
    *,
    keys: tuple[str, ...] = ("iin", "ignd", "p_in", "p_diss"),
    labels: dict[str, str] | None = None,
    scale: dict[str, float] | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Rail budget  typ vs max Ignd",
) -> plt.Figure:
    """Comparator 07 grouped bars: typical vs datasheet-max Ignd."""
    labels = labels or {
        "iin": r"$I_{\mathrm{IN}}$ (mA)",
        "ignd": r"$I_{\mathrm{GND}}$ (mA)",
        "p_in": r"$P_{\mathrm{IN}}$ (mW)",
        "p_diss": r"$P_{\mathrm{diss}}$ (mW)",
        "p_out": r"$P_{\mathrm{OUT}}$ (mW)",
        "efficiency": "η (%)",
    }
    scale = scale or {
        "iin": 1e3, "ignd": 1e3, "iload": 1e3,
        "p_in": 1e3, "p_diss": 1e3, "p_out": 1e3,
        "efficiency": 100.0,
    }
    fig, ax = plt.subplots(figsize=(max(7.2, 1.5 * len(keys) + 2), 4.0))
    x = np.arange(len(keys), dtype=float)
    w = 0.35
    ax.bar(x - w / 2, [scale.get(k, 1.0) * typ[k] for k in keys], width=w,
           color="C0", edgecolor="k", lw=0.4, label="typ Ignd")
    ax.bar(x + w / 2, [scale.get(k, 1.0) * mx[k] for k in keys], width=w,
           color="C3", edgecolor="k", lw=0.4, label="max Ignd")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(k, k) for k in keys])
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_mc_box(
    frames: dict[str, pd.DataFrame],
    *,
    metric: str = "vout",
    target: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Box plot of MC samples by engine (comparator dual-device analog)."""
    labels: list[str] = []
    data: list[np.ndarray] = []
    for name, df in frames.items():
        a = metric_1d(df, metric)
        if a.size == 0:
            continue
        labels.append(name)
        data.append(a)
    if not data:
        raise ValueError(f"no finite '{metric}' samples to box-plot")
    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 * len(data) + 2), 4.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    if target is not None:
        ax.axhline(target, color="#9467bd", ls=":", lw=1.3, label=f"{target:.3f} V")
        ax.legend(fontsize=8)
    ylab = {"vout": r"$V_{\mathrm{OUT}}$ (V)", "headroom": "headroom (V)"}.get(metric, metric)
    ax.set_ylabel(ylab)
    ax.set_title(title or f"MC box — {metric}")
    ax.grid(True, axis="y", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_psrr_vs_cout(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """Sensitivity-vs-frequency analog: PSRR @ 120 Hz / 1 kHz / 10 kHz vs Cout."""
    c = np.logspace(-7, -5, 41)  # 0.1 µF … 10 µF
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for f, ls in ((120.0, "-"), (1e3, "--"), (1e4, "-.")):
        y = [float(psrr_db(f, cout=float(cc))[0]) for cc in c]
        ax.semilogx(1e6 * c, y, ls, lw=1.6, label=f"{f:.0f} Hz")
    ax.axvline(1e6 * p.cout, color="0.45", ls=":", lw=1.0, label="BOM Cout")
    ax.set_xlabel(r"$C_{\mathrm{OUT}}$ (µF)")
    ax.set_ylabel("PSRR (dB)")
    ax.set_title("PSRR vs Cout (G21 interpolation)")
    ax.legend(fontsize=8)
    _grid(ax)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_tran_family(
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Load-step family",
    t_scale: float = 1e3,
    t_label: str = "t (ms)",
    ylabel: str = r"$V_{\mathrm{OUT}}$ (V)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for lab, (t, v) in traces.items():
        ax.plot(t_scale * t, v, lw=1.5, label=lab)
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _grid(ax)
    first = next(iter(traces.values()))
    annotate_minmax(ax, t_scale * first[0], first[1], fmt="{:.3f}")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_cap_voltage_bars(
    df: pd.DataFrame,
    *,
    name_col: str = "ref",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "Capacitor voltage vs rating / 80% derate",
    exclude: Sequence[str] | None = None,
) -> plt.Figure:
    """Opamp 08 analog: cap V_applied vs derated and catalog limits."""
    d = _prep_stress(df, "V_applied_V", name_col, dropna=True, exclude=exclude)
    d = d[d[name_col].astype(str).str.startswith("C")]
    if d.empty:
        d = _prep_stress(df, "V_applied_V", name_col, dropna=True, exclude=exclude)
        if "P_rated_W" in d.columns:
            d = d[(d["P_rated_W"].fillna(0) <= 0) & (d["V_max_V"] > 0)]
    d = d.sort_values("V_applied_V", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, max(3.2, 0.42 * max(len(d), 1) + 1.2)))
    y = np.arange(len(d))
    ax.barh(y, d["V_applied_V"], color="C0", edgecolor="k", lw=0.4, height=0.45, label="applied")
    if "V_derated_V" in d.columns:
        ax.plot(d["V_derated_V"], y, "C3|", ms=14, mew=1.6, label="80% derated")
    if "V_max_V" in d.columns:
        ax.plot(d["V_max_V"], y, color="0.4", marker="|", ls="none", ms=14, mew=1.3,
                label="catalog Vmax")
    ax.set_yticks(y)
    ax.set_yticklabels(d[name_col])
    ax.set_xlabel("voltage (V)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_headroom_vs_vin(
    p: CircuitParams,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    from .analysis import headroom

    vin = np.linspace(5.5, 16.0, 80)
    h_typ = [headroom(p, vin=float(v), which="typ")["headroom"] for v in vin]
    h_wc = [headroom(p, vin=float(v), which="max_ot")["headroom"] for v in vin]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(vin, h_typ, "k-", lw=1.8, label="typ dropout")
    ax.plot(vin, h_wc, "C3--", lw=1.4, label="max dropout")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.axvline(p.op.vin_nom, color="0.45", ls="--", lw=0.9)
    ax.set_xlabel(r"$V_{\mathrm{IN}}$ (V)")
    ax.set_ylabel("headroom (V)")
    ax.set_title("Headroom vs Vin")
    ax.legend(fontsize=8)
    _grid(ax)
    annotate_ends(ax, vin, h_typ, fmt="{:.3f}")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)
