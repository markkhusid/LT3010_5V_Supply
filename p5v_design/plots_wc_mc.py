"""Comparator-style WC hypercube + range-bar plots, and 3-engine MC overlays.

Visual language (same as comparator-hysteresis / HV_monitor):

- Horizontal **range bars** with endpoint ticks and min/max labels
- One bar per engine (analytical / ngspice / LTspice)
- Hypercube corners as a scatter + axis-aligned envelope box
- MC histograms sharing bin edges, WC band fill, RSS dash-dot, target line
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .plots import apply_annotations, savefig, vfmt

COLOR_AN = "#333333"
COLOR_NG = "#1f77b4"
COLOR_LT = "#ff7f0e"
COLOR_WC = "#d62728"
COLOR_RSS = "#2ca02c"
COLOR_TGT = "#9467bd"
COLOR_NOM = "#111111"

ENGINE_STYLE = {
    "analytical": dict(color=COLOR_AN, label="analytical"),
    "ngspice": dict(color=COLOR_NG, label="ngspice"),
    "LTspice": dict(color=COLOR_LT, label="LTspice"),
    "ltspice": dict(color=COLOR_LT, label="LTspice"),
}

METRIC_LABELS = {
    "vout": r"$V_{\mathrm{OUT}}$ (V)",
    "headroom": "headroom (V)",
    "p_diss": r"$P_{\mathrm{diss}}$ (W)",
    "efficiency": "efficiency",
    "iin": r"$I_{\mathrm{IN}}$ (A)",
    "vdo": r"$V_{\mathrm{DO}}$ (V)",
}


def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    """First column if a rename created duplicates."""
    s = frame[name]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return _finite(s)


def band_from_array(x) -> tuple[float, float, float] | None:
    a = _finite(x)
    if a.size == 0:
        return None
    return float(np.mean(a)), float(np.min(a)), float(np.max(a))


def band_from_wc(wc, key: str) -> tuple[float, float, float]:
    return float(wc.nominal[key]), float(wc.minimum[key]), float(wc.maximum[key])


def _fmt(v: float, metric: str) -> str:
    if metric == "vout" or metric == "headroom":
        return f"{v:.3f}"
    if metric == "efficiency":
        return f"{100.0 * v:.2f}%"
    if metric == "p_diss":
        return f"{1e3 * v:.1f} mW"
    if abs(v) >= 1:
        return f"{v:.4g}"
    return f"{v:.3f}"


def plot_range_bars(
    bands: dict[str, tuple[float, float, float]],
    *,
    metric: str = "vout",
    target: float | None = 5.0,
    rss: tuple[float, float] | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
) -> plt.Figure:
    """One panel of comparator-style horizontal WC bars (one row per engine).

    ``bands[engine] = (nominal, min, max)``. Missing engines are skipped.
    """
    preferred = ("analytical", "ngspice", "LTspice", "ltspice")
    keys = [k for k in preferred if k in bands] + [k for k in bands if k not in preferred]
    seen = set()
    clean = []
    for k in keys:
        lab = ENGINE_STYLE.get(k, {}).get("label", k)
        if lab.lower() in seen:
            continue
        seen.add(lab.lower())
        clean.append((lab, k, bands[k]))
    fig, ax = plt.subplots(figsize=(9.6, max(2.6, 0.85 * len(clean) + 1.6)))
    ys = []
    all_x = []
    for i, (lab, key, (nom, lo, hi)) in enumerate(clean):
        y = float(len(clean) - 1 - i)
        ys.append(y)
        col = ENGINE_STYLE.get(key, {}).get("color", f"C{i}")
        ax.barh(
            y, hi - lo, left=lo, height=0.42,
            color=col, alpha=0.38, edgecolor=col, lw=1.5, zorder=2, label=lab,
        )
        ax.plot([lo, hi], [y, y], "|", color=COLOR_NOM, ms=13, mew=1.5, zorder=6)
        ax.plot([nom], [y], "o", color=COLOR_NOM, ms=8, zorder=7)
        ax.annotate(
            _fmt(lo, metric), xy=(lo, y), xytext=(-8, -16),
            textcoords="offset points", ha="right", va="top", fontsize=8, color="0.15",
        )
        ax.annotate(
            _fmt(hi, metric), xy=(hi, y), xytext=(8, -16),
            textcoords="offset points", ha="left", va="top", fontsize=8, color="0.15",
        )
        all_x.extend([lo, hi, nom])
    if rss is not None:
        ax.axvline(rss[0], color=COLOR_RSS, ls="-.", lw=1.3, label="RSS min/max")
        ax.axvline(rss[1], color=COLOR_RSS, ls="-.", lw=1.3)
        all_x.extend(list(rss))
    if target is not None:
        ax.axvline(target, color=COLOR_TGT, ls=":", lw=1.5,
                   label=f"{target:.3f} V" if metric == "vout" else "target")
        all_x.append(target)
    span = max(all_x) - min(all_x) if all_x else 1.0
    pad = max(0.18 * span, 0.008)
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in clean])
    ax.set_ylim(-0.85, len(clean) - 0.15)
    ax.set_xlabel(xlabel or METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"WC range — {METRIC_LABELS.get(metric, metric)}")
    ax.grid(True, axis="x", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=8, loc="best")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_metric_range_grid(
    bands_by_metric: dict[str, dict[str, tuple[float, float, float]]],
    *,
    rss_by_metric: dict[str, tuple[float, float]] | None = None,
    target_by_metric: dict[str, float] | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "WC range bars — analytical / ngspice / LTspice",
) -> plt.Figure:
    """2×2 (or 1×N) of range-bar panels. Comparator dual-device idea, three engines."""
    metrics = list(bands_by_metric)
    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.2 * nrows))
    axes_l = np.atleast_1d(axes).ravel()
    for ax, met in zip(axes_l, metrics):
        bands = bands_by_metric[met]
        rows = [(k, bands[k]) for k in ("analytical", "ngspice", "LTspice", "ltspice") if k in bands]
        seen = set()
        clean = []
        for k, b in rows:
            lab = ENGINE_STYLE.get(k, {}).get("label", k)
            if lab in seen:
                continue
            seen.add(lab)
            clean.append((lab, k, b))
        all_x: list[float] = []
        for i, (lab, key, (nom, lo, hi)) in enumerate(clean):
            y = float(len(clean) - 1 - i)
            col = ENGINE_STYLE.get(key, {}).get("color", f"C{i}")
            ax.barh(y, hi - lo, left=lo, height=0.42, color=col, alpha=0.38, edgecolor=col, lw=1.4)
            ax.plot([lo, hi], [y, y], "|", color=COLOR_NOM, ms=11, mew=1.3, zorder=6)
            ax.plot([nom], [y], "o", color=COLOR_NOM, ms=7, zorder=7)
            ax.annotate(_fmt(lo, met), xy=(lo, y), xytext=(-6, -14), textcoords="offset points",
                        ha="right", va="top", fontsize=7)
            ax.annotate(_fmt(hi, met), xy=(hi, y), xytext=(6, -14), textcoords="offset points",
                        ha="left", va="top", fontsize=7)
            all_x.extend([lo, hi, nom])
        rss = (rss_by_metric or {}).get(met)
        if rss:
            ax.axvline(rss[0], color=COLOR_RSS, ls="-.", lw=1.1)
            ax.axvline(rss[1], color=COLOR_RSS, ls="-.", lw=1.1)
            all_x.extend(list(rss))
        tgt = (target_by_metric or {}).get(met)
        if tgt is not None:
            ax.axvline(tgt, color=COLOR_TGT, ls=":", lw=1.3)
            all_x.append(tgt)
        span = max(all_x) - min(all_x) if all_x else 1.0
        pad = max(0.16 * span, 1e-4)
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_yticks(list(range(len(clean))))
        ax.set_yticklabels([c[0] for c in clean][::-1], fontsize=8)
        ax.set_xlabel(METRIC_LABELS.get(met, met), fontsize=9)
        ax.set_title(METRIC_LABELS.get(met, met), fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
    for ax in axes_l[n:]:
        ax.set_axis_off()
    fig.suptitle(title, y=1.02, fontsize=12)
    apply_annotations(axes_l[0], annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_span_grouped_bars(
    spans: dict[str, dict[str, float]],
    *,
    metrics: tuple[str, ...] | None = None,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str = "WC span (max − min) by engine",
    ylabel: str = "span",
) -> plt.Figure:
    """Grouped vertical bars: one group per metric, one color per engine."""
    engines = [e for e in ("analytical", "ngspice", "LTspice") if e in spans]
    engines += [e for e in spans if e not in engines]
    metrics = metrics or tuple(next(iter(spans.values())).keys())
    x = np.arange(len(metrics), dtype=float)
    width = 0.8 / max(len(engines), 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, eng in enumerate(engines):
        vals = [spans[eng].get(m, np.nan) for m in metrics]
        col = ENGINE_STYLE.get(eng, {}).get("color", f"C{i}")
        ax.bar(x + (i - 0.5 * (len(engines) - 1)) * width, vals, width=width,
               color=col, edgecolor="k", lw=0.4, label=eng, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_hypercube_scatter(
    frame: pd.DataFrame,
    *,
    x: str = "headroom",
    y: str = "vout",
    spice: pd.DataFrame | None = None,
    ltspice: pd.DataFrame | None = None,
    wc_box: tuple[float, float, float, float] | None = None,
    nominal: tuple[float, float] | None = None,
    target_y: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Each green point is one hypercube corner (comparator VL/VH idea)."""
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.scatter(
        frame[x], frame[y], s=42, alpha=0.75, c=COLOR_RSS,
        edgecolors="none", label="analytical corners", zorder=3,
    )
    if spice is not None and x in spice.columns and y in spice.columns:
        xs = np.asarray(spice[x], dtype=float).ravel()
        ys = np.asarray(spice[y], dtype=float).ravel()
        n = min(xs.size, ys.size)
        ax.scatter(
            xs[:n], ys[:n], s=56, marker="x", c=COLOR_NG, lw=1.4,
            label="ngspice corners", zorder=5,
        )
    if ltspice is not None and x in ltspice.columns and y in ltspice.columns:
        xs = np.asarray(ltspice[x], dtype=float).ravel()
        ys = np.asarray(ltspice[y], dtype=float).ravel()
        n = min(xs.size, ys.size)
        ax.scatter(
            xs[:n], ys[:n], s=48, marker="s", facecolors="none",
            edgecolors=COLOR_LT, lw=1.3, label="LTspice corners", zorder=5,
        )
    if wc_box is None and y in frame.columns and x in frame.columns:
        wc_box = (
            float(frame[x].min()), float(frame[x].max()),
            float(frame[y].min()), float(frame[y].max()),
        )
    if wc_box is not None:
        x0, x1, y0, y1 = wc_box
        ax.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color=COLOR_NOM, ls="--", lw=1.3, label="analytical envelope", zorder=4,
        )
        if y == "vout" or x == "headroom":
            ax.annotate(vfmt(y0), xy=(x0, y0), xytext=(-6, -12), textcoords="offset points",
                        ha="right", fontsize=7)
            ax.annotate(vfmt(y1), xy=(x1, y1), xytext=(6, 8), textcoords="offset points",
                        ha="left", fontsize=7)
    if nominal is not None:
        ax.scatter([nominal[0]], [nominal[1]], s=110, c="k", marker="o", zorder=6, label="nominal")
    if target_y is not None and y == "vout":
        ax.axhline(target_y, color=COLOR_TGT, ls=":", lw=1.4, label=f"{target_y:.3f} V")
        ax.scatter([nominal[0] if nominal else frame[x].mean()], [target_y],
                   s=90, c="0.35", marker="D", zorder=6, label="goal")
    ax.set_xlabel(METRIC_LABELS.get(x, x))
    ax.set_ylabel(METRIC_LABELS.get(y, y))
    ax.set_title(title or "Hypercube corners")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_hypercube_params(
    frame: pd.DataFrame,
    *,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """R1 vs R2 and Vadj vs Vout — the knobs of the 16-corner cube."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    sc = axes[0].scatter(frame["r1"], frame["r2"], c=frame["vout"], cmap="viridis", s=70, edgecolors="k", lw=0.4)
    fig.colorbar(sc, ax=axes[0], label=r"$V_{\mathrm{OUT}}$ (V)")
    axes[0].set_xlabel(r"$R_1$ (Ω)")
    axes[0].set_ylabel(r"$R_2$ (Ω)")
    axes[0].set_title("Hypercube in resistor plane")
    axes[0].grid(True, alpha=0.3)
    sc2 = axes[1].scatter(
        frame["vadj"], frame["vout"], c=frame["iadj"] * 1e9, cmap="magma",
        s=70, edgecolors="k", lw=0.4,
    )
    cb = fig.colorbar(sc2, ax=axes[1])
    cb.set_label(r"$I_{\mathrm{ADJ}}$ (nA)")
    axes[1].axhline(5.0, color=COLOR_TGT, ls=":", lw=1.2)
    axes[1].set_xlabel(r"$V_{\mathrm{ADJ}}$ (V)")
    axes[1].set_ylabel(r"$V_{\mathrm{OUT}}$ (V)")
    axes[1].set_title("Vadj box × divider")
    axes[1].grid(True, alpha=0.3)
    apply_annotations(axes[1], annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_corner_strip(
    frame: pd.DataFrame,
    *,
    metric: str = "vout",
    spice: pd.DataFrame | None = None,
    target: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Every hypercube corner as a stem — comparator 'green points' in 1-D."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    y = _col(frame, metric)
    x = np.arange(len(y))
    ax.vlines(x, y.min() if y.size else 0, y, color=COLOR_RSS, alpha=0.35, lw=1.0)
    ax.plot(x, y, "o", color=COLOR_RSS, ms=7, label="analytical")
    if spice is not None and metric in spice.columns:
        ys = _col(spice, metric)
        xs = np.arange(len(ys))
        ax.plot(xs, ys, "x", color=COLOR_NG, ms=8, mew=1.4, label="ngspice")
    if target is not None and metric == "vout":
        ax.axhline(target, color=COLOR_TGT, ls=":", lw=1.4, label=f"{target:.3f} V")
    ax.axhline(float(np.mean(y)), color=COLOR_NOM, ls="--", lw=1.0, label="mean")
    if y.size:
        ax.annotate(vfmt(float(np.min(y))) if metric in ("vout", "headroom") else f"{np.min(y):.4g}",
                    xy=(0, float(np.min(y))), xytext=(8, -12), textcoords="offset points",
                    ha="left", fontsize=7)
        ax.annotate(vfmt(float(np.max(y))) if metric in ("vout", "headroom") else f"{np.max(y):.4g}",
                    xy=(int(np.argmax(y)), float(np.max(y))), xytext=(8, 8),
                    textcoords="offset points", ha="left", fontsize=7)
    ax.set_xticks(x)
    labels = [str(c) for c in frame.get("corner", pd.Series(x))]
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"Hypercube corners — {METRIC_LABELS.get(metric, metric)}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_an_vs_spice(
    an: np.ndarray,
    sp: np.ndarray,
    *,
    engine: str = "ngspice",
    metric: str = "vout",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    col = ENGINE_STYLE.get(engine, ENGINE_STYLE["ngspice"])["color"]
    axes[0].plot(an, an, "k-", lw=1.0, label="y = x")
    axes[0].scatter(an, sp, s=40, c=col, zorder=3, label=engine)
    axes[0].set_xlabel(f"analytical {METRIC_LABELS.get(metric, metric)}")
    axes[0].set_ylabel(f"{engine} {METRIC_LABELS.get(metric, metric)}")
    axes[0].set_title("Corner-by-corner overlay")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    err = 1e3 * (sp - an) if metric in ("vout", "headroom") else (sp - an)
    axes[1].stem(np.arange(len(err)), err, basefmt="k-")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("corner index")
    axes[1].set_ylabel("error (mV)" if metric in ("vout", "headroom") else "error")
    axes[1].set_title(f"{engine} − analytical")
    axes[1].grid(True, alpha=0.3)
    apply_annotations(axes[0], annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_wc_rss_band(
    *,
    nom: float,
    wc: tuple[float, float],
    rss: tuple[float, float],
    target: float | None = 5.0,
    mc: np.ndarray | None = None,
    metric: str = "vout",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """1-D WC (fill) vs RSS (hatch) band, optional MC rug. Comparator transfer-band idea."""
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.axvspan(wc[0], wc[1], color=COLOR_WC, alpha=0.14, label=f"WC [{wc[0]:.3f}, {wc[1]:.3f}]")
    ax.axvspan(rss[0], rss[1], color=COLOR_RSS, alpha=0.22, hatch="///", label=f"RSS [{rss[0]:.3f}, {rss[1]:.3f}]")
    ax.axvline(wc[0], color=COLOR_WC, ls="-", lw=1.3)
    ax.axvline(wc[1], color=COLOR_WC, ls="-", lw=1.3)
    ax.axvline(rss[0], color=COLOR_RSS, ls="-.", lw=1.2)
    ax.axvline(rss[1], color=COLOR_RSS, ls="-.", lw=1.2)
    ax.axvline(nom, color=COLOR_NOM, ls="-", lw=1.6, label=f"nominal {nom:.3f}")
    if target is not None:
        ax.axvline(target, color=COLOR_TGT, ls=":", lw=1.5, label=f"target {target:.3f}")
    if metric in ("vout", "headroom"):
        ax.annotate(vfmt(wc[0]), xy=(wc[0], 0), xytext=(-6, 12), textcoords="offset points",
                    ha="right", fontsize=8)
        ax.annotate(vfmt(wc[1]), xy=(wc[1], 0), xytext=(6, 12), textcoords="offset points",
                    ha="left", fontsize=8)
    if mc is not None:
        a = _finite(mc)
        ax.plot(a, np.full_like(a, 0.0), "|", color=COLOR_NG, ms=10, alpha=0.35, label="MC samples")
    ax.set_yticks([])
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or "WC envelope vs RSS vs target")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_mc_hist_engines(
    frames: dict[str, pd.DataFrame],
    *,
    metric: str = "vout",
    wc: tuple[float, float] | None = None,
    rss: tuple[float, float] | None = None,
    nominal: float | None = None,
    target: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
    n_bins: int = 40,
) -> plt.Figure:
    """Shared-bin density histograms for every engine that has samples."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    data = {}
    lo, hi = np.inf, -np.inf
    for name, fr in frames.items():
        if metric not in fr.columns:
            continue
        a = _col(fr, metric)
        if a.size == 0:
            continue
        data[name] = a
        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    if wc:
        lo, hi = min(lo, wc[0]), max(hi, wc[1])
    if target is not None:
        lo, hi = min(lo, target), max(hi, target)
    pad = 0.08 * (hi - lo) if hi > lo else 1e-3
    bins = np.linspace(lo - pad, hi + pad, n_bins + 1)
    for name, a in data.items():
        st = ENGINE_STYLE.get(name, ENGINE_STYLE["analytical"])
        filled = name == "analytical"
        ax.hist(
            a, bins=bins, density=True,
            color=st["color"], alpha=0.55 if filled else 1.0,
            edgecolor="white" if filled else st["color"],
            histtype="bar" if filled else "step",
            lw=1.8 if not filled else 0.4,
            label=f"{st['label']} (N={len(a)})",
        )
    if wc:
        ax.axvspan(wc[0], wc[1], color=COLOR_WC, alpha=0.08, zorder=0)
        ax.axvline(wc[0], color=COLOR_WC, ls="--", lw=1.4, label=f"WC {wc[0]:.3f}…{wc[1]:.3f}")
        ax.axvline(wc[1], color=COLOR_WC, ls="--", lw=1.4)
    if rss:
        ax.axvline(rss[0], color=COLOR_RSS, ls="-.", lw=1.3, label=f"RSS {rss[0]:.3f}…{rss[1]:.3f}")
        ax.axvline(rss[1], color=COLOR_RSS, ls="-.", lw=1.3)
    if nominal is not None:
        ax.axvline(nominal, color=COLOR_NOM, ls="-", lw=1.4, label=f"nominal {nominal:.3f}")
    if target is not None:
        ax.axvline(target, color=COLOR_TGT, ls=":", lw=1.5, label=f"{target:.3f} V")
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_ylabel("density")
    ax.set_title(title or f"MC overlay — {METRIC_LABELS.get(metric, metric)}")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=8)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_mc_cloud(
    frames: dict[str, pd.DataFrame],
    *,
    x: str = "headroom",
    y: str = "vout",
    wc_box: tuple[float, float, float, float] | None = None,
    target_y: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for name, fr in frames.items():
        if x not in fr.columns or y not in fr.columns:
            continue
        st = ENGINE_STYLE.get(name, ENGINE_STYLE["analytical"])
        xs, ys = _col(fr, x), _col(fr, y)
        n = min(xs.size, ys.size)
        kw: dict[str, Any] = dict(s=8 if name == "analytical" else 22, alpha=0.25 if name == "analytical" else 0.85,
                                  c=st["color"], label=st["label"], zorder=3)
        if name != "analytical":
            kw.update(marker="x", lw=1.1, alpha=0.9)
        ax.scatter(xs[:n], ys[:n], **kw)
    if wc_box:
        x0, x1, y0, y1 = wc_box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=1.2, label="WC envelope")
    if target_y is not None:
        ax.axhline(target_y, color=COLOR_TGT, ls=":", lw=1.3, label=f"{target_y:.3f} V")
    ax.set_xlabel(METRIC_LABELS.get(x, x))
    ax.set_ylabel(METRIC_LABELS.get(y, y))
    ax.set_title(title or "MC cloud")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_mc_vs_run(
    frame: pd.DataFrame,
    *,
    metric: str = "vout",
    wc: tuple[float, float] | None = None,
    target: float | None = 5.0,
    engine: str = "analytical",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    y = _col(frame, metric)
    ax.plot(np.arange(len(y)), y, ".", color=ENGINE_STYLE.get(engine, ENGINE_STYLE["analytical"])["color"],
            ms=3.5, alpha=0.7)
    if wc:
        ax.axhspan(wc[0], wc[1], color=COLOR_WC, alpha=0.08)
        ax.axhline(wc[0], color=COLOR_WC, ls="--", lw=1.2)
        ax.axhline(wc[1], color=COLOR_WC, ls="--", lw=1.2)
        if metric in ("vout", "headroom"):
            ax.annotate(vfmt(wc[0]), xy=(0, wc[0]), xytext=(8, -12), textcoords="offset points",
                        fontsize=7, ha="left")
            ax.annotate(vfmt(wc[1]), xy=(0, wc[1]), xytext=(8, 8), textcoords="offset points",
                        fontsize=7, ha="left")
    if target is not None:
        ax.axhline(target, color=COLOR_TGT, ls=":", lw=1.3)
    ax.set_xlabel("sample / run")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{engine} MC vs run")
    ax.grid(True, alpha=0.3)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_cdf(
    frames: dict[str, pd.DataFrame],
    *,
    metric: str = "vout",
    wc: tuple[float, float] | None = None,
    target: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for name, fr in frames.items():
        if metric not in fr.columns:
            continue
        a = np.sort(_col(fr, metric))
        if a.size == 0:
            continue
        y = np.linspace(0.0, 1.0, a.size)
        st = ENGINE_STYLE.get(name, ENGINE_STYLE["analytical"])
        ax.plot(a, y, color=st["color"], lw=1.8, label=st["label"])
    if wc:
        ax.axvline(wc[0], color=COLOR_WC, ls="--", lw=1.2)
        ax.axvline(wc[1], color=COLOR_WC, ls="--", lw=1.2)
        if metric in ("vout", "headroom"):
            ax.annotate(vfmt(wc[0]), xy=(wc[0], 0.02), xytext=(-6, 8), textcoords="offset points",
                        ha="right", fontsize=7)
            ax.annotate(vfmt(wc[1]), xy=(wc[1], 0.02), xytext=(6, 8), textcoords="offset points",
                        ha="left", fontsize=7)
    if target is not None:
        ax.axvline(target, color=COLOR_TGT, ls=":", lw=1.4)
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_ylabel("empirical CDF")
    ax.set_title("MC cumulative distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)


def plot_percentile_bars(
    frames: dict[str, pd.DataFrame],
    *,
    metric: str = "vout",
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
) -> plt.Figure:
    """p01–p99 as a range bar per engine (MC analog of the WC range bar)."""
    bands = {}
    for name, fr in frames.items():
        if metric not in fr.columns:
            continue
        a = _col(fr, metric)
        if a.size == 0:
            continue
        bands[name] = (float(np.median(a)), float(np.percentile(a, 1)), float(np.percentile(a, 99)))
    return plot_range_bars(
        bands, metric=metric, target=5.0 if metric == "vout" else None,
        annotations=annotations, path=path,
        title=f"MC p01–p99 range — {METRIC_LABELS.get(metric, metric)}",
    )


def plot_whiskers_engines(
    bands: dict[str, tuple[float, float, float]],
    *,
    metric: str = "vout",
    target: float | None = 5.0,
    annotations: list[dict] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """HV_monitor-style vertical whiskers (nom + min/max) per engine."""
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    rows = [(k, bands[k]) for k in ("analytical", "ngspice", "LTspice") if k in bands]
    for i, (name, (nom, lo, hi)) in enumerate(rows):
        col = ENGINE_STYLE[name]["color"]
        ax.plot([i, i], [lo, hi], color=col, lw=3.2, solid_capstyle="round")
        ax.plot(i, nom, "o", color=col, ms=9, zorder=3, label=f"{name}  {_fmt(lo, metric)} … {_fmt(hi, metric)}")
        ax.plot(i, lo, "_", color=col, ms=16, mew=2)
        ax.plot(i, hi, "_", color=col, ms=16, mew=2)
        ax.annotate(_fmt(lo, metric), xy=(i, lo), xytext=(8, -10), textcoords="offset points",
                    fontsize=8, ha="left")
        ax.annotate(_fmt(hi, metric), xy=(i, hi), xytext=(8, 8), textcoords="offset points",
                    fontsize=8, ha="left")
    if target is not None:
        ax.axhline(target, color=COLOR_TGT, ls=":", lw=1.3, label=f"{target:.3f} V")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or "WC min / nom / max by engine")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    apply_annotations(ax, annotations)
    fig.tight_layout()
    return savefig(fig, path)
