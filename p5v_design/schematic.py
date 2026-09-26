"""Annotated matplotlib schematic of the P5V ISO LT3010 regulator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

from .params import CircuitParams, format_eng


def draw_schematic(
    path: str | Path | None = None,
    p: CircuitParams | None = None,
    figsize: tuple[float, float] = (12.2, 5.2),
    annotations: list[dict] | None = None,
) -> Path | None:
    """Block-level schematic. Optional ``annotations`` are extra text labels
    in axes coordinates (0–1) so they stay put if you only tweak words:

        dict(text="...", xy=(0.12, 0.08), fontsize=8, color="0.2")
    """
    if p is None:
        p = CircuitParams()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "LT3010-5 datasheet example — 5 V supply with shutdown   "
        f"{p.op.vin_min:.1f}–{p.op.vin_max:.0f} V in, "
        f"{p.op.vout_target:.0f} V @ {p.op.iload_op * 1e3:.0f} mA",
        fontsize=12,
        pad=8,
    )

    def wire(x1, y1, x2, y2, **kw):
        ax.plot([x1, x2], [y1, y2], color=kw.get("color", "k"), lw=kw.get("lw", 1.35), zorder=2)

    def gnd(x, y):
        ax.plot([x - 0.22, x + 0.22], [y, y], color="k", lw=1.3)
        ax.plot([x - 0.14, x + 0.14], [y - 0.08, y - 0.08], color="k", lw=1.3)
        ax.plot([x - 0.06, x + 0.06], [y - 0.16, y - 0.16], color="k", lw=1.3)

    def box(x, y, w, h, text, fc="#f5f5f5", fontsize=8):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.04", facecolor=fc, edgecolor="k", lw=1.2, zorder=3
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4)

    def cap_label(x, y, name, val, esr=None):
        ax.text(x, y, name, ha="center", va="bottom", fontsize=7.5, color="0.15")
        extra = f"\nESR {format_eng(esr, 'Ω')}" if esr is not None else ""
        ax.text(x, y - 0.02, format_eng(val, "F") + extra, ha="center", va="top", fontsize=7, color="0.25")

    # Input source
    box(0.25, 3.35, 1.7, 1.15, f"VIN\n{p.op.vin_nom:.0f} V", fc="#ffe8e8", fontsize=9)
    wire(1.95, 3.92, 2.45, 3.92)

    # Cin
    wire(2.45, 3.92, 2.45, 2.55)
    ax.add_patch(Rectangle((2.32, 2.15), 0.26, 0.40, fill=False, lw=1.2))
    wire(2.45, 2.15, 2.45, 1.55)
    gnd(2.45, 1.55)
    cap_label(2.45, 1.22, "C4 Cin", p.cin, p.esr_in)
    wire(2.45, 3.92, 3.15, 3.92)

    # LT3010
    box(3.15, 2.85, 3.5, 2.15, "U1  LT3010-5\nIN  OUT\nSHDN  SENSE  GND", fc="#e8ffe8", fontsize=8.5)
    ax.annotate(
        "SHDN = VIN (always on)",
        xy=(3.3, 4.85),
        xytext=(3.3, 5.55),
        fontsize=7.5,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7),
        ha="left",
    )
    wire(6.65, 4.35, 7.35, 4.35)  # OUT
    wire(6.65, 3.35, 8.55, 3.35)  # ADJ
    wire(4.90, 2.85, 4.90, 1.55)
    gnd(4.90, 1.55)

    # Cff across R1
    wire(7.35, 4.35, 7.35, 4.85)
    wire(7.35, 4.85, 9.85, 4.85)
    ax.add_patch(Rectangle((8.45, 4.72), 0.40, 0.26, fill=False, lw=1.2))
    cff_txt = "no Cff" if p.cff <= 0 else f"C1 Cff  {format_eng(p.cff, 'F')}"
    ax.text(8.65, 5.22, cff_txt, ha="center", fontsize=7.5)

    # R1 top
    top = "SENSE tied\nto OUT" if p.r1 < 1.0 else f"R1  {format_eng(p.r1, 'Ω')}\nOUT → ADJ"
    box(7.35, 3.85, 2.5, 0.95, top, fc="#e8f0ff", fontsize=8)
    wire(9.85, 4.35, 11.15, 4.35)
    wire(9.85, 4.35, 9.85, 3.35)
    wire(8.55, 3.35, 9.85, 3.35)

    # R2 bot
    bot = "no external\ndivider" if p.r1 < 1.0 else f"R2  {format_eng(p.r2, 'Ω')}\nADJ → GND"
    box(8.55, 2.15, 2.0, 0.95, bot, fc="#e8f0ff", fontsize=8)
    wire(9.55, 2.15, 9.55, 1.55)
    gnd(9.55, 1.55)

    # Cout
    wire(11.15, 4.35, 11.15, 2.55)
    ax.add_patch(Rectangle((11.02, 2.15), 0.26, 0.40, fill=False, lw=1.2))
    wire(11.15, 2.15, 11.15, 1.55)
    gnd(11.15, 1.55)
    cap_label(11.15, 1.22, "C2 Cout", p.cout, p.esr_out)

    wire(11.15, 4.35, 12.0, 4.35)
    box(12.0, 3.55, 1.9, 1.35, f"Rload\n{format_eng(p.r_load, 'Ω')}\n{p.op.iload_op * 1e3:.0f} mA", fc="#fff3cd", fontsize=8)
    wire(13.90, 4.22, 14.45, 4.22)
    box(14.45, 3.55, 1.35, 1.35, "VOUT\n5 V", fc="#fde8ff", fontsize=9)
    wire(12.95, 3.55, 12.95, 1.55)
    gnd(12.95, 1.55)

    vout = p.vadj * p.gain + p.iadj * p.r1
    ax.text(
        8.0,
        0.38,
        f"LT3010-5, SENSE tied to OUT.  VOUT typical = {vout:.3f} V"
        if p.r1 < 1.0
        else (
            f"VOUT = {p.vadj:.3f} (1 + R1/R2) + IADJ·R1  =  {vout:.4f} V   "
            f"(target {p.op.vout_target:.3f} V,  bias {1e3 * (vout - p.op.vout_target):+.1f} mV)"
        ),
        fontsize=8,
        ha="center",
        color="0.15",
    )

    if annotations:
        for it in annotations:
            ax.text(
                it["xy"][0],
                it["xy"][1],
                it.get("text", ""),
                fontsize=it.get("fontsize", 8),
                color=it.get("color", "0.2"),
                ha=it.get("ha", "left"),
                va=it.get("va", "center"),
                transform=ax.transAxes if it.get("axes_frac") else ax.transData,
            )

    fig.tight_layout()
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
