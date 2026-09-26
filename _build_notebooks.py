"""Build Jupyter notebooks 01–10 for the LT3010-5 datasheet example."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def _jupyter_math(s: str) -> str:
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"\x00MATH{len(blocks) - 1}\x00"

    s = re.sub(r"(?m)^\$\$[ \t]*\n(?:.*\n)*?^\$\$[ \t]*$", _stash, s)
    s = re.sub(r"\\\((.+?)\\\)", r"$\1$", s, flags=re.S)
    s = re.sub(
        r"\\\[(.+?)\\\]",
        lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n",
        s,
        flags=re.S,
    )

    def _dd(m: re.Match) -> str:
        inner = m.group(1)
        line_start = s.rfind("\n", 0, m.start()) + 1
        line_end = s.find("\n", m.end())
        if line_end < 0:
            line_end = len(s)
        before = s[line_start : m.start()]
        after = s[m.end() : line_end]
        if before.strip() == "" and after.strip() == "":
            return m.group(0)
        return f"${inner}$"

    s = re.sub(r"\$\$(.+?)\$\$", _dd, s)
    s = re.sub(r"(?<!\$)\$([^$\n]+?)\$\$", r"$\1$", s)
    s = re.sub(r"\$\$([^$\n]+?)\$(?!\$)", r"$\1$", s)
    for i, b in enumerate(blocks):
        s = s.replace(f"\x00MATH{i}\x00", b)
    return s


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(_jupyter_math(source))}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(s: str) -> list[str]:
    s = s.strip("\n")
    if not s:
        return []
    lines = s.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def write_nb(name: str, cells: list[dict]) -> Path:
    path = NB_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print("wrote", path)
    return path


SETUP_MD = r"""## Setup — editable BOM and run knobs

This cell is shared by every notebook. It puts `P5V_Regulator/` on `sys.path`
and builds **one** `CircuitParams` object. Change a resistor, capacitor, or
tolerance here and re-run: analytical, SPICE, WC, and MC all follow this object.

**Plot callouts:** each figure cell has an `ann = [dict(...), ...]` list.
Edit `text`, `xy` (data coords of the arrow tip), and `offset=(dx, dy)` in
**points** to nudge a label. Use `xytext=(x, y)` instead of `offset` for data
coordinates. `arrow=False` for a plain label. `ann = []` hides all callouts.

Tolerances are **fractions** (0.01 = 1 %). Figures go to `results/figures/`.
"""

BOOT = r'''
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p5v_design.params import CircuitParams, ToleranceSpec, format_eng, LdoParams
from p5v_design import analysis, schematic, plots, stability
from p5v_design import plots_wc_mc as pwc
from p5v_design.ldo_model import dropout_v, ignd_a, tj_c, psrr_db, vadj_typical_vs_temp
from dataclasses import replace

# --- EDITABLE BOM ---
# Datasheet TA01, "5V Supply with Shutdown" (LT3010/LT3010-5 Rev. J page 1).
# Fixed 5 V part: SENSE tied to OUT (R1 = 0). No external divider (R2 open).
# Capacitors are the 1 µF parts drawn on that figure. Load is the labeled 50 mA.
# VIN is printed as 5.4 V to 80 V; 12 V is the single-point bias inside that range.
R1 = 0.0             # SENSE shorted to OUT
R2 = 1.0e12          # open — no bottom resistor
CFF = 0.0            # no feedforward capacitor on the fixed part
COUT = 1.0e-6
CIN = 1.0e-6
ESR_OUT = 10e-3      # ceramic assumption, not printed on TA01
ESR_IN = 10e-3
RLOAD = 100.0        # 5 V / 50 mA
R_TOL = 0.01         # 1 %
C_TOL = 0.10         # 10 %
VIN_NOM = 12.0
VIN_MIN = 5.4
VIN_MAX = 80.0
ILOAD_OP = 50e-3
N_MC = 2000
N_SPICE_MC = 40
N_LTSPICE_MC = 200
SKIP_SPICE_WC_MC = False
RUN_LTSPICE_BATCH = False
SPICE_WORKERS = None
LTSPICE_WORKERS = None

p = CircuitParams(
    r1=R1, r2=R2, cff=CFF, cout=COUT, cin=CIN,
    esr_out=ESR_OUT, esr_in=ESR_IN, r_load=RLOAD,
    r_tol=ToleranceSpec(R_TOL), c_tol=ToleranceSpec(C_TOL),
)
p = replace(p, op=replace(p.op, vin_nom=VIN_NOM, vin_min=VIN_MIN, vin_max=VIN_MAX,
                           iload_op=ILOAD_OP, r_load=RLOAD))

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
print("gain =", p.gain, "  Vout nom =", analysis.vout_dc(p))
print("SPICE_WORKERS =", SPICE_WORKERS, "  N_MC =", N_MC)
'''


def nb01():
    return [
        md("# 01 — Analytical setpoint (LT3010-5 datasheet example)\n\n"
           "Fixed LT3010-5, datasheet Rev. J page 1 typical application "
           '"5V Supply with Shutdown": 5 V at 50 mA, 1 µF input and output '
           "capacitors, VIN printed as 5.4 V to 80 V. Single-point calculations "
           "use 12 V, which is inside that range. Page 3 is the source of every "
           "IC min/typ/max used below."),
        md(r"""## Datasheet example

Page 1, figure TA01, "5V Supply with Shutdown":

| Item | Value on the figure |
|------|---------------------|
| Part | LT3010-5 |
| VIN | 5.4 V to 80 V |
| VOUT | 5 V |
| Load | 50 mA |
| Input capacitor | 1 µF |
| Output capacitor | 1 µF |
| SHDN | above 2.0 V on, below 0.3 V off |

SENSE is tied to OUT, so there is no external divider. The page-3
over-temperature output box (4.850–5.150 V) already includes line, load,
and temperature. Default WC/MC uses that envelope and does not stack
line and load on top of it.
"""),
        md(SETUP_MD),
        code(BOOT),
        md("## Schematic and nominal setpoint\n\n"
           "SENSE is tied to OUT, so the typical setpoint is the page-3 5.000 V, "
           "before IC and resistor tolerances. That bias is reported, not “fixed”."),
        code("""
from IPython.display import display, Markdown
sp = analysis.setpoint(p)
tbl = analysis.setpoint_table(p)
display(analysis.style_metrics(tbl))
display(Markdown("$$" + analysis.latex_vout() + "$$"))
print(f"setpoint bias vs 5.000 V: {1e3*sp['setpoint_bias']:+.2f} mV  ({sp['setpoint_bias_pct']:+.2f} %)")
sch = schematic.draw_schematic(FIG / "01_schematic.png", p)
print("saved", sch)
"""),
        md("## Divider current vs $I_{\\mathrm{ADJ}}$\n\n"
           "Datasheet: keep the bottom resistor under 250 kΩ so $I_{\\mathrm{ADJ}}$ "
           "is a small error. Our R2 = 2.2 kΩ is comfortably below that."),
        code("""
stab = stability.stability_table(p)
display(stab)
ann_div = [
    dict(text="operate Idiv", xy=(p.r2, 1e6 * p.i_div), offset=(18, 12), arrow=True),
]
# Idiv vs R2 family
r2g = np.logspace(3, 5.4, 80)
idiv = p.vadj / r2g
fig, ax = plt.subplots(figsize=(8.0, 4.2))
ax.loglog(r2g, 1e6 * idiv, "k-", lw=1.8, label=r"$I_{div}=V_{ADJ}/R_2$")
ax.axhline(p.ldo.iadj_typ * 1e6, color="C3", ls="--", lw=1.1, label="Iadj typ 50 nA")
ax.axhline(p.ldo.iadj_max * 1e6, color="C1", ls=":", lw=1.1, label="Iadj max 100 nA")
ax.axvline(p.r2, color="0.45", ls="--")
ax.axvline(250e3, color="C4", ls="-.", lw=1.0, label="250 kΩ guideline")
ax.set_xlabel(r"$R_2$ (Ω)")
ax.set_ylabel(r"$I_{div}$ (µA)")
ax.set_title("Feedback divider current vs bottom resistor")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plots.apply_annotations(ax, ann_div)
plots.savefig(fig, FIG / "01_idiv_vs_r2.png")
plt.show()
"""),
        md("## Cff 10 kHz rule and Cout/ESR window"),
        code("""
f, z, rbot = stability.cff_impedance_sweep(p)
ann_cff = [
    dict(text="10 kHz rule", xy=(10e3, rbot), offset=(12, 14), arrow=True),
    dict(text="|Xcff|", xy=(10e3, z[np.argmin(np.abs(f-10e3))]), offset=(12, -18), arrow=True),
]
fig, ax = plt.subplots(figsize=(8.0, 4.2))
ax.loglog(f, z, "k-", lw=1.8, label=r"$|X_{Cff}|$")
ax.axhline(rbot, color="C3", ls="--", label=f"R2 = {format_eng(rbot, 'Ω')}")
ax.axvline(10e3, color="0.45", ls=":")
ax.set_xlabel("f (Hz)")
ax.set_ylabel("Ω")
ax.set_title("Cff impedance vs datasheet 10 kHz rule")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plots.apply_annotations(ax, ann_cff)
plots.savefig(fig, FIG / "01_cff_z.png")
plt.show()
print(f"Cout {format_eng(p.cout,'F')}  (min {format_eng(p.ldo.cout_min_f,'F')})   "
      f"ESR {format_eng(p.esr_out,'Ω')}  (max {p.ldo.esr_max_ohm} Ω)")
"""),
        md("## Page-3 error budget — envelope vs stacked\n\n"
           "Envelope terms go into WC/MC. Stacked extras (line, load) are "
           "**already inside** the over-temp $V_{\\mathrm{ADJ}}$ box."),
        code("""
bud_25 = analysis.error_budget_table(p, over_temp=False)
bud_ot = analysis.error_budget_table(p, over_temp=True)
print("25 °C envelope RSS half-span", f"{bud_25.attrs['rss_half_span_mV']:.2f} mV")
print("over-temp envelope RSS half-span", f"{bud_ot.attrs['rss_half_span_mV']:.2f} mV")
display(bud_25)
ann_bud = []  # tornado is self-labelled; add dicts here if you want callouts
plots.plot_error_budget(bud_25, annotations=ann_bud, path=FIG/"01_budget_envelope.png",
                        title="Envelope OAT half-span (25 °C)")
plt.show()
plots.plot_error_budget_signed(bud_25, annotations=[], path=FIG/"01_budget_signed.png")
plt.show()
env = analysis.envelope_bounds(p, over_temp=False)
env_ot = analysis.envelope_bounds(p, over_temp=True)
stk = analysis.stacked_bounds(p, over_temp=True)
print("envelope 25 °C  Vout", env)
print("envelope over temp", env_ot)
print("stacked (pessimistic)", stk)
"""),
        md("## $V_{\\mathrm{OUT}}$ vs $V_{\\mathrm{IN}}$ (dropout knee + line)"),
        code("""
vin, v_typ = analysis.vout_vs_vin(p, which_do="typ")
_, v_wc = analysis.vout_vs_vin(p, which_do="max_ot")
vdo_op = dropout_v(p, which="max_ot")
ann_vin = [
    dict(text="12 V bias", xy=(12.0, analysis.vout_dc(p)), offset=(16, -22), arrow=True, color="C3"),
    dict(text="dropout knee", xy=(5.0 + vdo_op, analysis.vout_dc(p) * 0.92), offset=(16, -18), arrow=True),
    dict(text="target 5 V", xy=(11.0, 5.0), offset=(0, 12), arrow=False),
]
plots.plot_vout_vs_vin(
    vin, v_typ, vout_wc=v_wc, vin_nom=p.op.vin_nom, vout_target=p.op.vout_target,
    vdo=vdo_op, annotations=ann_vin, path=FIG/"01_vout_vs_vin.png",
)
plt.show()
hr = analysis.headroom(p, which="max_ot")
display(analysis.style_metrics(analysis.metrics_table(hr)))
plots.plot_headroom_bar(p, annotations=[
    dict(text=f"headroom {hr['headroom']:.2f} V", xy=(p.op.vin_nom*0.72, 0.0), offset=(0, 18), arrow=False),
], path=FIG/"01_headroom.png")
plt.show()
"""),
        md("## Load regulation, dropout, and $I_{\\mathrm{GND}}$"),
        code("""
i, vo = analysis.vout_vs_iload(p)
ann_load = [
    dict(text="50 mA load", xy=(50.0, analysis.vout_dc(p)), offset=(12, 12), arrow=True),
]
plots.plot_vout_vs_iload(i, vo, iload_op=p.op.iload_op, annotations=ann_load,
                         path=FIG/"01_vout_vs_i.png")
plt.show()
ann_do = [
    dict(text="50 mA", xy=(50.0, 1e3*dropout_v(p, which="typ")), offset=(10, 14), arrow=True),
    dict(text="max over temp", xy=(50.0, 550), offset=(-40, 10), arrow=True, color="C3"),
]
plots.plot_dropout_vs_i(p, annotations=ann_do, path=FIG/"01_dropout.png")
plt.show()
ann_ig = [
    dict(text="operate", xy=(35.0, 1e3*ignd_a(p, which="typ")), offset=(12, 12), arrow=True),
]
plots.plot_ignd_vs_i(p, annotations=ann_ig, path=FIG/"01_ignd.png")
plt.show()
"""),
        md("## Regulation families (datasheet G-curve style)"),
        code("""
ann_f = [dict(text="12 V / 50 mA", xy=(12.0, analysis.vout_dc(p)), offset=(12, -18), arrow=True)]
plots.plot_regulation_family(p, which="vin", annotations=ann_f, path=FIG/"01_family_vin.png")
plt.show()
plots.plot_regulation_family(p, which="i", path=FIG/"01_family_i.png")
plt.show()
"""),
        md("## PSRR, noise, thermal, current-limit margin"),
        code("""
ann_psrr = [
    dict(text="120 Hz spec 75 dB typ", xy=(120, 75), offset=(14, 8), arrow=True),
]
plots.plot_psrr(p, annotations=ann_psrr, path=FIG/"01_psrr.png")
plt.show()
ann_n = [
    dict(text=r"~1.5 µV/√Hz", xy=(1e3, 1.5), offset=(12, 10), arrow=True),
]
plots.plot_noise_psd(p, annotations=ann_n, path=FIG/"01_noise_psd.png")
plt.show()
th = tj_c(p, which="typ")
display(analysis.style_metrics(analysis.metrics_table(th)))
print(f"Ilim min over temp {1e3*p.ldo.ilim_min_ot_a:.0f} mA   operate {1e3*p.op.iload_op:.0f} mA   "
      f"margin {1e3*(p.ldo.ilim_min_ot_a-p.op.iload_op):.0f} mA")
m = analysis.summary_metrics(p)
display(analysis.style_metrics(analysis.metrics_table(m)))
ripple = analysis.psrr_ripple_out(p, vin_ripple_pp=0.05, f_hz=120.0)
print(f"50 mVpp @ 120 Hz on VIN → {1e3*ripple:.3f} mVpp on VOUT (datasheet PSRR curve)")
"""),
        md("## Extra datasheet families (from HV_monitor / Opamp / comparator notebooks)\n\n"
           "PSRR 1 µF vs 10 µF (G21), integrated noise vs bandwidth, current-limit "
           "margin, headroom vs Vin, and the over-temp error-budget tornado."),
        code("""
ann_pf = [dict(text="BOM 10 µF", xy=(120, 75), offset=(14, 8), arrow=True)]
plots.plot_psrr_family(p, annotations=ann_pf, path=FIG/"01_psrr_family.png")
plt.show()
ann_nrms = [dict(text="10 kHz", xy=(1e4, 150), offset=(12, 10), arrow=True)]
plots.plot_noise_rms_vs_bw(p, annotations=ann_nrms, path=FIG/"01_noise_rms_bw.png")
plt.show()
plots.plot_ilim_margin(p, annotations=[
    dict(text="50 mA load", xy=(50, 0), offset=(12, 12), arrow=True),
], path=FIG/"01_ilim.png")
plt.show()
plots.plot_headroom_vs_vin(p, annotations=[
    dict(text="12 V bias", xy=(12.0, analysis.headroom(p)["headroom"]), offset=(12, 10), arrow=True),
], path=FIG/"01_headroom_vs_vin.png")
plt.show()
plots.plot_error_budget(bud_ot, annotations=[], path=FIG/"01_budget_overtemp.png",
                        title="Envelope OAT half-span (over temp)")
plt.show()
plots.plot_wc_vout_vs_vin(p, annotations=[
    dict(text="dropout knee", xy=(5.4, 4.7), offset=(14, -12), arrow=True),
], path=FIG/"01_wc_vs_vin.png")
plt.show()
"""),
    ]


def nb02():
    return [
        md("# 02 — ngspice vs analytical\n\n"
           "Behavioral LDO (`LT3010_BEHAV`) uses the **external divider**, so R1/R2 "
           "and $I_{\\mathrm{ADJ}}$ match the closed form. Dynamics (PSRR pole, "
           "load-step recovery) are a sanity check — the vendor macromodel in "
           "notebook 05 is the dynamics reference."),
        md(SETUP_MD),
        code(BOOT),
        code("""
from p5v_design.simulate import find_ngspice, simulate_op, simulate_dc_vin, simulate_dc_i, simulate_ac_psrr

try:
    print("ngspice", find_ngspice())
    HAVE_SPICE = True
except FileNotFoundError as e:
    print(e)
    HAVE_SPICE = False
"""),
        md("## Operating point"),
        code("""
an = analysis.summary_metrics(p)
print("analytical Vout", an["vout"])
if HAVE_SPICE:
    op = simulate_op(p, workdir=ROOT/"results"/"ngspice_nom"/"op",
                     mirror=ROOT/"netlists"/"spice_nom"/"p5v_op.cir")
    print("ngspice nodes", op.nodes)
    vsp = op.nodes.get("out", list(op.nodes.values())[0] if op.nodes else float("nan"))
    print(f"error {1e3*(vsp-an['vout']):.3f} mV")
else:
    print("skip ngspice OP")
"""),
        md("## Vin sweep overlay"),
        code("""
vin, v_an = analysis.vout_vs_vin(p, which_do="typ")
ann = [
    dict(text="12 V bias", xy=(12.0, analysis.vout_dc(p)), offset=(12, 10), arrow=True),
]
if HAVE_SPICE:
    dc = simulate_dc_vin(p, workdir=ROOT/"results"/"ngspice_nom"/"dc_vin",
                         mirror=ROOT/"netlists"/"spice_nom"/"p5v_dc_vin.cir")
    v_sp = np.interp(vin, dc.x, dc.vout)
    plots.plot_overlay_dc(vin, v_an, v_sp, xlabel=r"$V_{IN}$ (V)", title="Vin sweep",
                          annotations=ann, path=FIG/"02_vin_overlay.png")
else:
    plots.plot_vout_vs_vin(vin, v_an, annotations=ann, path=FIG/"02_vin_overlay.png")
plt.show()
"""),
        md("## Iload sweep overlay"),
        code("""
i, vo = analysis.vout_vs_iload(p)
ann = [dict(text="50 mA", xy=(50.0, analysis.vout_dc(p)), offset=(10, 10), arrow=True)]
if HAVE_SPICE:
    dci = simulate_dc_i(p, workdir=ROOT/"results"/"ngspice_nom"/"dc_i",
                        mirror=ROOT/"netlists"/"spice_nom"/"p5v_dc_i.cir")
    # sweep variable is Iload (A)
    v_sp = np.interp(i, dci.x, dci.vout)
    plots.plot_overlay_dc(1e3*i, vo, v_sp, xlabel="Iload (mA)", title="Iload sweep",
                          annotations=ann, path=FIG/"02_i_overlay.png")
else:
    plots.plot_vout_vs_iload(i, vo, annotations=ann, path=FIG/"02_i_overlay.png")
plt.show()
"""),
        md("## PSRR — datasheet curve vs behavioral AC"),
        code("""
ann = [dict(text="120 Hz", xy=(120, 75), offset=(12, 8), arrow=True)]
if HAVE_SPICE:
    ac = simulate_ac_psrr(p, workdir=ROOT/"results"/"ngspice_nom"/"psrr",
                          mirror=ROOT/"netlists"/"spice_nom"/"p5v_psrr.cir")
    plots.plot_psrr(p, spice_f=ac.f_hz, spice_db=ac.psrr_db, annotations=ann,
                    path=FIG/"02_psrr.png")
else:
    plots.plot_psrr(p, annotations=ann, path=FIG/"02_psrr.png")
plt.show()
"""),
        md("## Residual and OP bars (HV_monitor / Opamp notebook 02 style)\n\n"
           "PSRR error = ngspice − datasheet G21. OP bars compare closed-form "
           "$V_{\\mathrm{OUT}}$ to the behavioral OP."),
        code("""
an = analysis.summary_metrics(p)
rows = {"analytical": {"vout": an["vout"]}}
if HAVE_SPICE:
    ac = simulate_ac_psrr(p, workdir=ROOT/"results"/"ngspice_nom"/"psrr")
    f = np.logspace(1, 6, 241)
    ds = psrr_db(f, cout=p.cout)
    sp = np.interp(f, ac.f_hz, ac.psrr_db)
    plots.plot_error_db(
        f, sp - ds, ylabel="ngspice − G21 (dB)",
        title="PSRR residual  ngspice − datasheet",
        path=FIG/"02_psrr_error.png",
    )
    plt.show()
    try:
        vsp = float(op.nodes.get("out", next(iter(op.nodes.values()))))
        rows["ngspice"] = {"vout": vsp}
        print(f"OP error {1e3*(vsp-an['vout']):.3f} mV")
    except NameError:
        print("OP result not in namespace; skip OP bars overlay")
plots.plot_op_compare_bars(
    rows, metrics=("vout",), units={"vout": "V"},
    title="OP  Vout  analytical vs ngspice", path=FIG/"02_op_bars.png",
)
plt.show()
"""),
    ]


def nb03():
    return [
        md(r"""# 03 — Worst-case hypercube (analytical + ngspice)

Endpoint **hypercube** on the envelope sources $V_{\mathrm{ADJ}}$, $I_{\mathrm{ADJ}}$,
$R_1$, $R_2$ ($2^4 = 16$ corners), plus RSS of one-at-a-time half-spans.

This notebook is the regulator analog of the comparator $V_L$/$V_H$ WC chapter:

| Picture | Comparator | This regulator |
|---------|------------|----------------|
| Range bars | $V_L$, $V_H$, $V_{\mathrm{HYS}}$ min…max | $V_{\mathrm{OUT}}$, headroom, $P_{\mathrm{diss}}$, efficiency |
| Green points | one $(V_L,V_H)$ per hypercube corner | one $({\mathrm{headroom}}, V_{\mathrm{OUT}})$ per corner |
| Envelope box | axis-aligned min/max in the $V_L$–$V_H$ plane | same, in headroom–$V_{\mathrm{OUT}}$ |
| RSS hatch | $\sqrt{\sum \delta_i^2}$ inside the box | same formula on $V_{\mathrm{OUT}}$ |

The hypercube is **pessimistic** (every part at a worst endpoint together).
RSS is the usual independent-error bound.

$$
\mathrm{RSS}(m)=\sqrt{\sum_i \delta_i(m)^2},\qquad
\delta_i(m)=\tfrac12\bigl|m(x_i^{\max})-m(x_i^{\min})\bigr|
$$
"""),
        md(SETUP_MD),
        code(BOOT),
        md(r"""## Walk the 16-corner envelope

Signs: $R_1$ lo/hi × $R_2$ lo/hi × $V_{\mathrm{ADJ}}$ lo/hi × $I_{\mathrm{ADJ}}$ lo/hi.
$V_{\mathrm{OUT}}$ rises with $V_{\mathrm{ADJ}}$, $I_{\mathrm{ADJ}}$, $R_1$ and falls with $R_2$.
"""),
        code("""
from p5v_design.worst_case import (
    analytical_worst_case, oat_contributions, rss_bounds, rss_bounds_many,
    envelope_worst_case, spice_wc_corners, corners_frame,
)
from p5v_design.simulate import find_ngspice

wc = analytical_worst_case(p, over_temp=False)
wc_ot = analytical_worst_case(p, over_temp=True)
rss = rss_bounds_many(p, over_temp=False)
oat = oat_contributions(p, metrics=("vout", "headroom", "p_diss", "efficiency"))
cdf = corners_frame(wc)
print("16 corners + nominal rows:", len(wc.rows))
display(pd.DataFrame([wc.nominal, wc.minimum, wc.maximum], index=["nom", "min", "max"]).T)
display(cdf.sort_values("vout")[["corner", "r1", "r2", "vadj", "iadj", "vout", "headroom", "p_diss"]].head(8))
display(cdf.sort_values("vout")[["corner", "vout", "headroom"]].tail(8))
print("RSS Vout", rss["vout"])
"""),
        md(r"""## Range bars — analytical WC (comparator threshold-band chart)

Each bar is **min…max** of the hypercube. Black ticks are the endpoints;
the black circle is nominal. Purple dotted line is the 5 V goal.
Edit `ann_bar` offsets if a label collides.
"""),
        code("""
ann_bar = [
    dict(text="5 V goal", xy=(5.0, 0.0), offset=(8, 18), arrow=True, color="#9467bd"),
]
pwc.plot_range_bars(
    {"analytical": pwc.band_from_wc(wc, "vout")},
    metric="vout", target=5.0, rss=(rss["vout"]["low"], rss["vout"]["high"]),
    annotations=ann_bar, path=FIG/"03_range_vout.png",
    title=r"Analytical WC range  $V_{\mathrm{OUT}}$",
)
plt.show()
pwc.plot_metric_range_grid(
    {
        "vout": {"analytical": pwc.band_from_wc(wc, "vout")},
        "headroom": {"analytical": pwc.band_from_wc(wc, "headroom")},
        "p_diss": {"analytical": pwc.band_from_wc(wc, "p_diss")},
        "efficiency": {"analytical": pwc.band_from_wc(wc, "efficiency")},
    },
    rss_by_metric={k: (rss[k]["low"], rss[k]["high"]) for k in rss},
    target_by_metric={"vout": 5.0},
    path=FIG/"03_range_grid.png",
    title="Analytical WC range bars (four metrics)",
)
plt.show()
pwc.plot_whiskers_engines(
    {"analytical": pwc.band_from_wc(wc, "vout")},
    metric="vout", target=5.0, path=FIG/"03_whiskers_vout.png",
)
plt.show()
"""),
        md(r"""## WC fill vs RSS hatch vs 5 V (1-D band)

Same idea as the comparator transfer plot: **WC** is the wide light band,
**RSS** is the hatched inner band, nominal is the solid line, goal is dotted.
"""),
        code("""
ann_band = [
    dict(text="WC envelope", xy=(wc.minimum["vout"], 0), offset=(10, 16), arrow=True, color="#d62728"),
    dict(text="RSS", xy=(rss["vout"]["low"], 0), offset=(-20, 16), arrow=True, color="#2ca02c"),
]
pwc.plot_wc_rss_band(
    nom=wc.nominal["vout"], wc=(wc.minimum["vout"], wc.maximum["vout"]),
    rss=(rss["vout"]["low"], rss["vout"]["high"]), target=5.0,
    annotations=ann_band, path=FIG/"03_wc_rss_band.png",
)
plt.show()
"""),
        md(r"""## Hypercube scatter — each green point is one corner

Comparator notebook 01 plots $(V_L, V_H)$ for every corner. Here the natural
plane is **headroom vs $V_{\mathrm{OUT}}$**: both move when the divider or
$V_{\mathrm{ADJ}}$ hits an endpoint. The dashed box is the axis-aligned envelope.
The diamond is the 5 V goal (at nominal headroom).
"""),
        code("""
ann_sc = [
    dict(text="nominal", xy=(wc.nominal["headroom"], wc.nominal["vout"]), offset=(12, -16), arrow=True),
    dict(text="5 V goal", xy=(wc.nominal["headroom"], 5.0), offset=(14, 10), arrow=True, color="#9467bd"),
]
pwc.plot_hypercube_scatter(
    cdf, x="headroom", y="vout",
    nominal=(wc.nominal["headroom"], wc.nominal["vout"]),
    target_y=5.0, annotations=ann_sc, path=FIG/"03_hypercube_scatter.png",
    title=r"Hypercube corners  (headroom, $V_{\mathrm{OUT}}$)",
)
plt.show()
pwc.plot_hypercube_scatter(
    cdf, x="gain", y="vout",
    nominal=(wc.nominal["gain"], wc.nominal["vout"]),
    target_y=5.0, path=FIG/"03_hypercube_gain.png",
    title=r"Hypercube  $G=1+R_1/R_2$ vs $V_{\mathrm{OUT}}$",
)
plt.show()
pwc.plot_hypercube_params(cdf, path=FIG/"03_hypercube_params.png")
plt.show()
"""),
        md(r"""## Stem plot of all 16 corners

Every corner on one axis, sorted in the hypercube walk order. The 5 V line
shows the page-3 output-voltage box with SENSE tied to OUT.
"""),
        code("""
ann_st = [dict(text="lowest corner", xy=(0, cdf["vout"].min()), offset=(12, -14), arrow=True)]
pwc.plot_corner_strip(cdf, metric="vout", target=5.0, annotations=ann_st,
                      path=FIG/"03_corner_strip_vout.png")
plt.show()
pwc.plot_corner_strip(cdf, metric="headroom", target=None,
                      path=FIG/"03_corner_strip_headroom.png")
plt.show()
pwc.plot_corner_strip(cdf, metric="p_diss", target=None,
                      path=FIG/"03_corner_strip_pdiss.png")
plt.show()
"""),
        md(r"""## Tornado (OAT) — where to spend tolerance money

Each bar is the half-span with **everything else held nominal**. $V_{\mathrm{ADJ}}$
should dominate $V_{\mathrm{OUT}}$; $I_{\mathrm{ADJ}}$ is almost invisible at 2.2 kΩ.
"""),
        code("""
for met, xlab, fname in (
    ("vout", "half-span (V)", "03_tornado_vout.png"),
    ("headroom", "half-span (V)", "03_tornado_headroom.png"),
    ("p_diss", "half-span (W)", "03_tornado_pdiss.png"),
    ("efficiency", "half-span", "03_tornado_eta.png"),
):
    plots.plot_tornado(oat[met], xlabel=xlab, title=f"OAT tornado — {met}",
                       path=FIG/fname)
    plt.show()
# Span grouped: envelope vs over-temp vs RSS
pwc.plot_span_grouped_bars(
    {
        "analytical": {
            "vout": wc.span("vout"),
            "headroom": wc.span("headroom"),
            "p_diss": wc.span("p_diss"),
            "efficiency": wc.span("efficiency"),
        },
    },
    path=FIG/"03_span_bars.png",
    title="Analytical WC span (25 °C envelope)",
    ylabel="span (native units)",
)
plt.show()
"""),
        md(r"""## 25 °C vs over-temp envelope

MP grade is specified −55 °C to 125 °C. The over-temp $V_{\mathrm{ADJ}}$ box
(1.237–1.313 V) **includes** line and load, so it is the datasheet-faithful
hot/cold envelope — do not stack line/load on top of it.
"""),
        code("""
pwc.plot_range_bars(
    {"analytical": pwc.band_from_wc(wc, "vout")},
    metric="vout", target=5.0, rss=(rss["vout"]["low"], rss["vout"]["high"]),
    annotations=[dict(text="25 °C envelope", xy=(wc.nominal["vout"], 0), offset=(10, 16), arrow=True)],
    path=FIG/"03_range_25C.png", title=r"$V_{\mathrm{OUT}}$ WC  25 °C envelope",
)
plt.show()
pwc.plot_range_bars(
    {"analytical": pwc.band_from_wc(wc_ot, "vout")},
    metric="vout", target=5.0,
    path=FIG/"03_range_overtemp.png", title=r"$V_{\mathrm{OUT}}$ WC  over-temp envelope (MP)",
)
plt.show()
pwc.plot_span_grouped_bars(
    {
        "25 °C": {m: wc.span(m) for m in ("vout", "headroom", "p_diss")},
        "over temp": {m: wc_ot.span(m) for m in ("vout", "headroom", "p_diss")},
        "RSS 25 °C": {m: 2 * rss[m]["delta_rss"] for m in ("vout", "headroom", "p_diss")},
    },
    path=FIG/"03_span_25_vs_ot.png",
    title="Span: 25 °C hypercube vs over-temp vs RSS",
)
plt.show()
plots.plot_wc_vout_vs_vin(p, annotations=[
    dict(text="25 °C envelope", xy=(12.0, analysis.envelope_bounds(p)["max"]),
         offset=(12, 10), arrow=True),
], path=FIG/"03_wc_vs_vin.png")
plt.show()
ann_span = [
    dict(text="5.000 V goal", xy=(5.0, 1.0), offset=(10, 16), arrow=True, color="#9467bd"),
]
plots.plot_wc_span_compare(
    [
        {"name": "25 °C", "min": wc.minimum["vout"], "max": wc.maximum["vout"],
         "nominal": wc.nominal["vout"]},
        {"name": "over temp", "min": wc_ot.minimum["vout"], "max": wc_ot.maximum["vout"],
         "nominal": wc_ot.nominal["vout"]},
        {"name": "RSS 25 °C", "min": rss["vout"]["low"], "max": rss["vout"]["high"],
         "nominal": wc.nominal["vout"]},
    ],
    metric=r"$V_{\mathrm{OUT}}$ (V)", title="WC span  25 °C vs over-temp vs RSS",
    target=5.0, annotations=ann_span, path=FIG/"03_wc_span_whiskers.png",
)
plt.show()
"""),
        md(r"""## ngspice hypercube (same 16 corners)

Behavioral LDO OP at every analytical corner. Expect $V_{\mathrm{OUT}}$ within
a millivolt of the closed form. Set `SKIP_SPICE_WC_MC = True` to skip.
"""),
        code("""
ng_df = None
HAVE_SPICE = False
try:
    find_ngspice()
    HAVE_SPICE = True
except FileNotFoundError as e:
    print(e)

if HAVE_SPICE and not SKIP_SPICE_WC_MC:
    rows = spice_wc_corners(
        p, workdir=ROOT/"results"/"ngspice_wc", max_workers=SPICE_WORKERS, full=True,
    )
    ng_df = pd.DataFrame(rows)
    display(ng_df[["corner", "vout_an", "vout_spice", "r1", "r2", "vadj"]])
    print("max |spice − an| mV", 1e3 * np.nanmax(np.abs(ng_df.vout_spice - ng_df.vout_an)))
    ng_band = pwc.band_from_array(ng_df["vout_spice"])
    pwc.plot_range_bars(
        {
            "analytical": pwc.band_from_wc(wc, "vout"),
            "ngspice": ng_band,
        },
        metric="vout", target=5.0, rss=(rss["vout"]["low"], rss["vout"]["high"]),
        annotations=[dict(text="ngspice vs closed form", xy=(ng_band[0], 0), offset=(10, 14), arrow=True)],
        path=FIG/"03_range_an_ng.png",
        title=r"$V_{\mathrm{OUT}}$ WC  analytical vs ngspice",
    )
    plt.show()
    pwc.plot_hypercube_scatter(
        cdf, x="headroom", y="vout", spice=ng_df.rename(columns={"vout_spice": "vout"}),
        nominal=(wc.nominal["headroom"], wc.nominal["vout"]), target_y=5.0,
        path=FIG/"03_hypercube_an_ng.png",
        title="Hypercube  analytical + ngspice",
    )
    plt.show()
    pwc.plot_corner_strip(
        cdf, metric="vout", spice=ng_df.rename(columns={"vout_spice": "vout"}),
        target=5.0, path=FIG/"03_strip_an_ng.png",
    )
    plt.show()
    pwc.plot_an_vs_spice(
        ng_df["vout_an"].to_numpy(), ng_df["vout_spice"].to_numpy(),
        engine="ngspice", metric="vout", path=FIG/"03_an_vs_ng.png",
    )
    plt.show()
    pwc.plot_whiskers_engines(
        {"analytical": pwc.band_from_wc(wc, "vout"), "ngspice": ng_band},
        metric="vout", target=5.0, path=FIG/"03_whiskers_an_ng.png",
    )
    plt.show()
else:
    print("ngspice WC skipped")
"""),
    ]


def nb04():
    return [
        md(r"""# 04 — Monte Carlo (uniform) vs WC / RSS

Independent **uniform** draws on every tolerance interval (a datasheet box is
not a $3\sigma$ Gaussian). Histograms share bin edges. Overlay lines:

| Line | Meaning |
|------|---------|
| black solid | analytical nominal |
| red dashed + fill | hypercube WC min…max (notebook 03) |
| green dash-dot | RSS min…max |
| purple dotted | 5 V goal |

A well-behaved MC cloud sits **inside RSS** and well inside the WC box.
ngspice MC redraws the same passives + $V_{\mathrm{ADJ}}$/$I_{\mathrm{ADJ}}$
and scores $V_{\mathrm{OUT}}$ from an OP. LTspice MC lives in notebook 05
and is overlaid here if its `.log` is already on disk.
"""),
        md(SETUP_MD),
        code(BOOT),
        md(r"""## Analytical MC at the operate point

Each sample redraws $R_1$, $R_2$, $C$, ESR, $V_{\mathrm{ADJ}}$ (25 °C box),
and $I_{\mathrm{ADJ}}$ (0…max). Seed 42. Cache: `results/mc_analytical/`.
"""),
        code("""
from p5v_design.monte_carlo import sample_metrics, spice_monte_carlo
from p5v_design.worst_case import analytical_worst_case, rss_bounds_many, corners_frame
from p5v_design.ltspice_io import meas_frame
from p5v_design.simulate import find_ngspice

wc = analytical_worst_case(p, over_temp=False)
rss = rss_bounds_many(p)
mc = sample_metrics(p, n=N_MC, seed=42, over_temp=False)
mc.save(ROOT/"results"/"mc_analytical")
display(mc.summary)
print("MC Vout p01…p99", mc.frame.vout.quantile(0.01), mc.frame.vout.quantile(0.99))
print("WC Vout", wc.minimum["vout"], wc.maximum["vout"])
print("fraction of MC outside WC",
      ((mc.frame.vout < wc.minimum["vout"]) | (mc.frame.vout > wc.maximum["vout"])).mean())
"""),
        md(r"""## Histograms with WC / RSS / target (vout, headroom, dissipation)

Same layout as the comparator $V_H$/$V_L$/$V_{\mathrm{HYS}}$ trio.
"""),
        code("""
ann_h = [dict(text="5 V target", xy=(5.0, 0), offset=(10, 36), arrow=True, color="#9467bd")]
for met in ("vout", "headroom", "p_diss", "efficiency"):
    tgt = 5.0 if met == "vout" else None
    pwc.plot_mc_hist_engines(
        {"analytical": mc.frame}, metric=met,
        wc=(wc.minimum[met], wc.maximum[met]),
        rss=(rss[met]["low"], rss[met]["high"]),
        nominal=wc.nominal[met], target=tgt,
        annotations=ann_h if met == "vout" else [],
        path=FIG/f"04_mc_hist_{met}.png",
        title=f"Analytical MC — {met}",
    )
    plt.show()
"""),
        md(r"""## MC cloud, CDF, vs-run, and p01–p99 range bars"""),
        code("""
cdf = corners_frame(wc)
pwc.plot_mc_cloud(
    {"analytical": mc.frame},
    x="headroom", y="vout",
    wc_box=(wc.minimum["headroom"], wc.maximum["headroom"], wc.minimum["vout"], wc.maximum["vout"]),
    target_y=5.0, path=FIG/"04_mc_cloud.png",
    title=r"MC cloud  (headroom, $V_{\mathrm{OUT}}$)  + WC box",
)
plt.show()
pwc.plot_cdf(
    {"analytical": mc.frame}, metric="vout",
    wc=(wc.minimum["vout"], wc.maximum["vout"]), target=5.0,
    path=FIG/"04_mc_cdf_vout.png",
)
plt.show()
pwc.plot_mc_vs_run(
    mc.frame, metric="vout", wc=(wc.minimum["vout"], wc.maximum["vout"]),
    target=5.0, engine="analytical", path=FIG/"04_mc_vs_run.png",
)
plt.show()
pwc.plot_percentile_bars(
    {"analytical": mc.frame}, metric="vout",
    annotations=[dict(text="p01–p99 vs 5 V", xy=(5.0, 0), offset=(8, 16), arrow=True)],
    path=FIG/"04_mc_p01_p99.png",
)
plt.show()
pwc.plot_wc_rss_band(
    nom=wc.nominal["vout"], wc=(wc.minimum["vout"], wc.maximum["vout"]),
    rss=(rss["vout"]["low"], rss["vout"]["high"]), target=5.0,
    mc=mc.frame["vout"].to_numpy(),
    path=FIG/"04_mc_on_wc_band.png",
    title="MC rug on WC / RSS band",
)
plt.show()
plots.plot_mc_box(
    {"analytical": mc.frame}, metric="vout", target=5.0,
    path=FIG/"04_mc_box_vout.png", title="Analytical MC box — Vout",
)
plt.show()
"""),
        md(r"""## ngspice MC overlay

Same seed, smaller $N$ (`N_SPICE_MC`). Density histograms share bins with
analytical. The WC/RSS lines stay the **analytical** envelope.
"""),
        code("""
spice = None
HAVE_SPICE = False
try:
    find_ngspice()
    HAVE_SPICE = True
except FileNotFoundError as e:
    print(e)

if HAVE_SPICE and not SKIP_SPICE_WC_MC:
    spice = spice_monte_carlo(
        p, n=N_SPICE_MC, seed=42,
        workdir=ROOT/"results"/"ngspice_mc", max_workers=SPICE_WORKERS,
    )
    display(spice.summary)
    frames = {"analytical": mc.frame, "ngspice": spice.frame.rename(columns={"vout_spice": "vout"})}
    pwc.plot_mc_hist_engines(
        frames, metric="vout",
        wc=(wc.minimum["vout"], wc.maximum["vout"]),
        rss=(rss["vout"]["low"], rss["vout"]["high"]),
        nominal=wc.nominal["vout"], target=5.0,
        path=FIG/"04_mc_hist_an_ng.png",
        title="MC  analytical + ngspice",
    )
    plt.show()
    pwc.plot_cdf(frames, metric="vout", wc=(wc.minimum["vout"], wc.maximum["vout"]),
                 target=5.0, path=FIG/"04_mc_cdf_an_ng.png")
    plt.show()
    pwc.plot_mc_cloud(
        frames, x="headroom", y="vout",
        wc_box=(wc.minimum["headroom"], wc.maximum["headroom"], wc.minimum["vout"], wc.maximum["vout"]),
        target_y=5.0, path=FIG/"04_mc_cloud_an_ng.png",
    )
    plt.show()
    pwc.plot_mc_vs_run(spice.frame.rename(columns={"vout_spice": "vout"}), metric="vout",
                       wc=(wc.minimum["vout"], wc.maximum["vout"]), target=5.0,
                       engine="ngspice", path=FIG/"04_mc_vs_run_ng.png")
    plt.show()
    pwc.plot_percentile_bars(frames, metric="vout", path=FIG/"04_p01_p99_an_ng.png")
    plt.show()
    pwc.plot_span_grouped_bars(
        {
            "WC": {"vout": wc.span("vout"), "headroom": wc.span("headroom")},
            "MC 1–99%": {
                "vout": float(mc.frame.vout.quantile(0.99) - mc.frame.vout.quantile(0.01)),
                "headroom": float(mc.frame.headroom.quantile(0.99) - mc.frame.headroom.quantile(0.01)),
            },
            "RSS": {"vout": 2 * rss["vout"]["delta_rss"], "headroom": 2 * rss["headroom"]["delta_rss"]},
        },
        path=FIG/"04_span_wc_mc_rss.png",
        title="Span: WC box vs MC p01–p99 vs RSS",
    )
    plt.show()
else:
    print("ngspice MC skipped")
    frames = {"analytical": mc.frame}
"""),
        md(r"""## LTspice MC overlay (if notebook 05 has already written a log)

This cell is read-only: it does not launch LTspice. Run notebook 05 (or
`python LTSpice/run_ltspice_batch.py mc`) then re-run this cell.
"""),
        code("""
lt_log = ROOT/"LTSpice"/"MC"/"P5V_Regulator_MC.log"
if lt_log.is_file():
    lt_mc = meas_frame(lt_log, "vout_op", "vout")
    print("LTspice MC N", len(lt_mc), "min/max", lt_mc.vout.min(), lt_mc.vout.max())
    frames3 = dict(frames)
    frames3["LTspice"] = lt_mc
    pwc.plot_mc_hist_engines(
        frames3, metric="vout",
        wc=(wc.minimum["vout"], wc.maximum["vout"]),
        rss=(rss["vout"]["low"], rss["vout"]["high"]),
        nominal=wc.nominal["vout"], target=5.0,
        path=FIG/"04_mc_hist_three_engines.png",
        title="MC  analytical + ngspice + LTspice",
    )
    plt.show()
    pwc.plot_cdf(frames3, metric="vout", wc=(wc.minimum["vout"], wc.maximum["vout"]),
                 target=5.0, path=FIG/"04_mc_cdf_three.png")
    plt.show()
    pwc.plot_percentile_bars(frames3, metric="vout", path=FIG/"04_p01_p99_three.png")
    plt.show()
    pwc.plot_mc_vs_run(lt_mc, metric="vout", wc=(wc.minimum["vout"], wc.maximum["vout"]),
                       target=5.0, engine="LTspice", path=FIG/"04_mc_vs_run_lt.png")
    plt.show()
    plots.plot_mc_box(frames3, metric="vout", target=5.0, path=FIG/"04_mc_box_three.png",
                      title="MC box — three engines")
    plt.show()
else:
    print("no LTspice MC log yet — run notebook 05")
"""),
    ]


def nb05():
    return [
        md(r"""# 05 — Three engines: analytical / ngspice / LTspice

Nominal, **WC hypercube**, and **uniform MC** on the same `CircuitParams`.

| Engine | What it is | WC | MC |
|--------|------------|----|----|
| Analytical | closed-form $V_{\mathrm{OUT}}$ | 16-corner envelope | `N_MC` Uniform |
| ngspice | behavioral LDO, external divider | same 16 corners (OP) | `N_SPICE_MC` OP |
| LTspice | vendor `LT3010.lib` | 16 grouped-sign corners on ADJ injection | `N_LTSPICE_MC` `mc()` |

LTspice **breaks the ADJ net** and injects `VADJ_ERR` (delta from typical
1.275 V) plus `IADJ_EXTRA` so the macromodel's typical reference is not
double-counted.

Set `RUN_LTSPICE_BATCH = True` in the setup cell to batch-run. If logs
already exist they are parsed either way.
"""),
        md(SETUP_MD),
        code(BOOT),
        md(r"""## Write decks and (optionally) batch-run LTspice"""),
        code("""
from p5v_design.ltspice_io import (
    find_ltspice, run_ltspice_deck, run_ltspice_decks, parse_meas_log,
    parse_meas_tables, parse_ltspice_raw_tran, meas_frame, meas_values,
)
from p5v_design.worst_case import (
    analytical_worst_case, rss_bounds_many, spice_wc_corners, corners_frame,
)
from p5v_design.monte_carlo import sample_metrics, spice_monte_carlo
from p5v_design.simulate import find_ngspice
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_ltspice_batch", ROOT / "LTSpice" / "run_ltspice_batch.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.write_all(p, mc_n=N_LTSPICE_MC)
print("decks written")

HAVE_LT = False
try:
    print("LTspice", find_ltspice())
    HAVE_LT = True
except FileNotFoundError as e:
    print(e)

HAVE_SPICE = False
try:
    print("ngspice", find_ngspice())
    HAVE_SPICE = True
except FileNotFoundError as e:
    print(e)

op_cir = ROOT/"LTSpice"/"Nominal"/"P5V_Regulator_op.cir"
tran_cir = ROOT/"LTSpice"/"Nominal"/"P5V_Regulator_tran.cir"
wc_cir = ROOT/"LTSpice"/"WC"/"P5V_Regulator_WC.cir"
mc_cir = ROOT/"LTSpice"/"MC"/"P5V_Regulator_MC.cir"
if HAVE_LT and RUN_LTSPICE_BATCH:
    run_ltspice_decks([op_cir, wc_cir, mc_cir], max_workers=LTSPICE_WORKERS)
else:
    print("LTspice batch skipped (RUN_LTSPICE_BATCH=False). Existing logs still parsed.")
"""),
        md(r"""## Load all three engines into frames"""),
        code("""
wc = analytical_worst_case(p)
rss = rss_bounds_many(p)
cdf = corners_frame(wc)
mc = sample_metrics(p, n=N_MC, seed=42)
mc.save(ROOT/"results"/"mc_analytical")

ng_wc = None
ng_mc = None
if HAVE_SPICE and not SKIP_SPICE_WC_MC:
    ng_rows = spice_wc_corners(p, workdir=ROOT/"results"/"ngspice_wc",
                               max_workers=SPICE_WORKERS, full=True)
    ng_wc = pd.DataFrame(ng_rows)
    ng_mc = spice_monte_carlo(p, n=N_SPICE_MC, seed=42,
                              workdir=ROOT/"results"/"ngspice_mc",
                              max_workers=SPICE_WORKERS)
    display(ng_wc[["corner", "vout_an", "vout_spice"]].head())
    display(ng_mc.summary)

lt_wc = meas_frame(ROOT/"LTSpice"/"WC"/"P5V_Regulator_WC.log") if (ROOT/"LTSpice"/"WC"/"P5V_Regulator_WC.log").is_file() else pd.DataFrame()
lt_mc = meas_frame(ROOT/"LTSpice"/"MC"/"P5V_Regulator_MC.log") if (ROOT/"LTSpice"/"MC"/"P5V_Regulator_MC.log").is_file() else pd.DataFrame()
print("LTspice WC N", len(lt_wc), "MC N", len(lt_mc))

def _band_frame(fr, col="vout"):
    if fr is None or len(fr) == 0 or col not in fr.columns:
        return None
    return pwc.band_from_array(fr[col])

bands_vout = {"analytical": pwc.band_from_wc(wc, "vout")}
if ng_wc is not None:
    bands_vout["ngspice"] = _band_frame(ng_wc, "vout_spice")
if len(lt_wc):
    bands_vout["LTspice"] = _band_frame(lt_wc, "vout")
print("Vout bands", bands_vout)
"""),
        md(r"""## WC range bars — three engines (comparator dual-device chart)

One horizontal bar per engine, endpoint ticks, min/max labels, 5 V goal.
This is the picture that should match notebook 01 of the hysteresis project.
"""),
        code("""
ann_r = [dict(text="5 V", xy=(5.0, 1.0), offset=(8, 14), arrow=True, color="#9467bd")]
pwc.plot_range_bars(
    {k: v for k, v in bands_vout.items() if v is not None},
    metric="vout", target=5.0, rss=(rss["vout"]["low"], rss["vout"]["high"]),
    annotations=ann_r, path=FIG/"05_wc_range_three.png",
    title=r"WC $V_{\mathrm{OUT}}$  analytical / ngspice / LTspice",
)
plt.show()
pwc.plot_whiskers_engines(
    {k: v for k, v in bands_vout.items() if v is not None},
    metric="vout", target=5.0, path=FIG/"05_wc_whiskers_three.png",
)
plt.show()

# Grid of metrics that exist on all analytical + ngspice
grid = {
    "vout": {"analytical": pwc.band_from_wc(wc, "vout")},
    "headroom": {"analytical": pwc.band_from_wc(wc, "headroom")},
    "p_diss": {"analytical": pwc.band_from_wc(wc, "p_diss")},
    "efficiency": {"analytical": pwc.band_from_wc(wc, "efficiency")},
}
if ng_wc is not None:
    grid["vout"]["ngspice"] = _band_frame(ng_wc, "vout_spice")
if len(lt_wc):
    grid["vout"]["LTspice"] = _band_frame(lt_wc, "vout")
pwc.plot_metric_range_grid(
    grid, rss_by_metric={k: (rss[k]["low"], rss[k]["high"]) for k in rss},
    target_by_metric={"vout": 5.0},
    path=FIG/"05_wc_range_grid.png",
    title="WC range bars by metric and engine",
)
plt.show()

spans = {
    "analytical": {m: wc.span(m) for m in ("vout", "headroom", "p_diss")},
}
if ng_wc is not None:
    spans["ngspice"] = {"vout": float(ng_wc.vout_spice.max() - ng_wc.vout_spice.min()),
                        "headroom": np.nan, "p_diss": np.nan}
if len(lt_wc):
    spans["LTspice"] = {"vout": float(lt_wc.vout.max() - lt_wc.vout.min()),
                        "headroom": np.nan, "p_diss": np.nan}
pwc.plot_span_grouped_bars(spans, metrics=("vout",), path=FIG/"05_wc_span_vout.png",
                           title="WC Vout span by engine", ylabel="span (V)")
plt.show()
"""),
        md(r"""## Hypercube scatter + stem: analytical corners vs ngspice / LTspice"""),
        code("""
spice_sc = None
if ng_wc is not None:
    spice_sc = ng_wc.rename(columns={"vout_spice": "vout"})
pwc.plot_hypercube_scatter(
    cdf, x="headroom", y="vout", spice=spice_sc,
    nominal=(wc.nominal["headroom"], wc.nominal["vout"]), target_y=5.0,
    annotations=[dict(text="nominal", xy=(wc.nominal["headroom"], wc.nominal["vout"]), offset=(12, -14), arrow=True)],
    path=FIG/"05_hypercube_scatter.png",
    title="WC hypercube  analytical + ngspice",
)
plt.show()
pwc.plot_corner_strip(
    cdf, metric="vout", spice=spice_sc, target=5.0,
    path=FIG/"05_corner_strip.png",
)
plt.show()
if ng_wc is not None:
    pwc.plot_an_vs_spice(
        ng_wc["vout_an"].to_numpy(), ng_wc["vout_spice"].to_numpy(),
        engine="ngspice", path=FIG/"05_an_vs_ng_corners.png",
    )
    plt.show()
if len(lt_wc):
    # LTspice WC is a .step sweep, not 1:1 with our corner names — compare as a distribution
    pwc.plot_range_bars(
        {
            "analytical": pwc.band_from_wc(wc, "vout"),
            "LTspice": _band_frame(lt_wc, "vout"),
        },
        metric="vout", target=5.0, path=FIG/"05_range_an_lt.png",
        title="Analytical hypercube vs LTspice 16-corner .step",
    )
    plt.show()
    pwc.plot_wc_rss_band(
        nom=wc.nominal["vout"], wc=(wc.minimum["vout"], wc.maximum["vout"]),
        rss=(rss["vout"]["low"], rss["vout"]["high"]), target=5.0,
        mc=lt_wc["vout"].to_numpy(),
        path=FIG/"05_lt_wc_on_band.png",
        title="LTspice WC corners as a rug on the analytical WC/RSS band",
    )
    plt.show()
"""),
        md(r"""## MC histograms — three engines, shared bins"""),
        code("""
frames = {"analytical": mc.frame}
if ng_mc is not None:
    frames["ngspice"] = ng_mc.frame.rename(columns={"vout_spice": "vout"})
if len(lt_mc):
    frames["LTspice"] = lt_mc

pwc.plot_mc_hist_engines(
    frames, metric="vout",
    wc=(wc.minimum["vout"], wc.maximum["vout"]),
    rss=(rss["vout"]["low"], rss["vout"]["high"]),
    nominal=wc.nominal["vout"], target=5.0,
    annotations=[dict(text="5 V", xy=(5.0, 0), offset=(10, 32), arrow=True, color="#9467bd")],
    path=FIG/"05_mc_hist_three.png",
    title="MC $V_{OUT}$  analytical / ngspice / LTspice",
)
plt.show()
pwc.plot_cdf(frames, metric="vout", wc=(wc.minimum["vout"], wc.maximum["vout"]),
             target=5.0, path=FIG/"05_mc_cdf_three.png")
plt.show()
pwc.plot_percentile_bars(frames, metric="vout", path=FIG/"05_mc_p01_p99_three.png")
plt.show()
pwc.plot_mc_cloud(
    {k: v for k, v in frames.items() if "headroom" in v.columns},
    x="headroom", y="vout",
    wc_box=(wc.minimum["headroom"], wc.maximum["headroom"], wc.minimum["vout"], wc.maximum["vout"]),
    target_y=5.0, path=FIG/"05_mc_cloud_three.png",
)
plt.show()
for name, fr in frames.items():
    pwc.plot_mc_vs_run(
        fr, metric="vout", wc=(wc.minimum["vout"], wc.maximum["vout"]),
        target=5.0, engine=name, path=FIG/f"05_mc_vs_run_{name}.png",
    )
    plt.show()
"""),
        md(r"""## Span summary: WC box vs MC p01–p99 vs RSS, by engine"""),
        code("""
span_map = {
    "analytical WC": {"vout": wc.span("vout")},
    "analytical MC p01–p99": {
        "vout": float(mc.frame.vout.quantile(0.99) - mc.frame.vout.quantile(0.01))
    },
    "RSS": {"vout": 2 * rss["vout"]["delta_rss"]},
}
if ng_wc is not None:
    span_map["ngspice WC"] = {"vout": float(ng_wc.vout_spice.max() - ng_wc.vout_spice.min())}
if ng_mc is not None and "vout_spice" in ng_mc.frame.columns:
    q = ng_mc.frame.vout_spice.quantile
    span_map["ngspice MC p01–p99"] = {"vout": float(q(0.99) - q(0.01))}
if len(lt_wc):
    span_map["LTspice WC"] = {"vout": float(lt_wc.vout.max() - lt_wc.vout.min())}
if len(lt_mc):
    q = lt_mc.vout.quantile
    span_map["LTspice MC p01–p99"] = {"vout": float(q(0.99) - q(0.01))}
display(pd.DataFrame(span_map).T.rename(columns={"vout": "Vout span (V)"}))
pwc.plot_span_grouped_bars(span_map, metrics=("vout",), path=FIG/"05_span_all.png",
                           title="Vout span  WC / MC / RSS  × engine", ylabel="span (V)")
plt.show()
"""),
        md(r"""## Nominal LTspice OP / startup (vendor macromodel)

Typical dynamics live here. WC/MC above used ADJ injection; this block is
the unmodified LT3010.lib at the BOM values.
"""),
        code("""
ann = [dict(text="settled Vout", xy=(8.0, analysis.vout_dc(p)), offset=(10, 12), arrow=True)]
if HAVE_LT and (RUN_LTSPICE_BATCH or op_cir.with_suffix(".log").is_file()):
    if RUN_LTSPICE_BATCH or not op_cir.with_suffix(".log").is_file():
        log = run_ltspice_deck(op_cir)
    else:
        log = op_cir.with_suffix(".log")
    print("OP meas", parse_meas_log(log))
    raw = tran_cir.with_suffix(".raw")
    if RUN_LTSPICE_BATCH or not raw.is_file():
        try:
            run_ltspice_deck(tran_cir)
        except Exception as e:
            print("tran skip", e)
    if raw.is_file():
        tr = parse_ltspice_raw_tran(raw)
        plots.plot_tran(tr.t, tr.vout, vin=tr.vin, lt_t=tr.t, lt_v=tr.vout,
                        annotations=ann, path=FIG/"05_startup.png", title="LTspice startup (typical model)")
        plt.show()
else:
    print("LTspice nominal not run. Set RUN_LTSPICE_BATCH = True.")
print("done three-engine comparison")
"""),
        md(r"""## OP bars and PSRR overlay (HV_monitor notebook 05 style)

Grouped $V_{\mathrm{OUT}}$ at the operate point. PSRR overlay uses the
datasheet G21 curve plus ngspice AC; LTspice PSRR if `P5V_Regulator_psrr`
has been run.
"""),
        code("""
rows = {"analytical": {"vout": analysis.vout_dc(p)}}
if HAVE_SPICE:
    try:
        from p5v_design.simulate import simulate_op
        op_ng = simulate_op(p, workdir=ROOT/"results"/"ngspice_nom"/"op")
        rows["ngspice"] = {"vout": float(op_ng.nodes.get("out", next(iter(op_ng.nodes.values()))))}
    except Exception as e:
        print("ngspice OP skip", e)
lt_op = ROOT/"LTSpice"/"Nominal"/"P5V_Regulator_op.log"
if HAVE_LT and lt_op.is_file():
    try:
        m = parse_meas_log(lt_op)
        if "vout_op" in m:
            rows["LTspice"] = {"vout": float(m["vout_op"])}
    except Exception as e:
        print("LTspice OP skip", e)
plots.plot_op_compare_bars(rows, metrics=("vout",), title="OP Vout  three engines",
                           path=FIG/"05_op_bars.png")
plt.show()
ann = [dict(text="120 Hz", xy=(120, 75), offset=(12, 8), arrow=True)]
if HAVE_SPICE:
    try:
        from p5v_design.simulate import simulate_ac_psrr
        ac = simulate_ac_psrr(p, workdir=ROOT/"results"/"ngspice_nom"/"psrr")
        plots.plot_psrr(p, spice_f=ac.f_hz, spice_db=ac.psrr_db, annotations=ann,
                        path=FIG/"05_psrr_overlay.png")
        f = np.logspace(1, 6, 241)
        ds = psrr_db(f, cout=p.cout)
        sp = np.interp(f, ac.f_hz, ac.psrr_db)
        plots.plot_error_db(f, sp - ds, ylabel="ngspice − G21 (dB)",
                            title="PSRR residual", path=FIG/"05_psrr_error.png")
        plt.show()
    except Exception as e:
        print("PSRR overlay skip", e)
        plots.plot_psrr(p, annotations=ann, path=FIG/"05_psrr_overlay.png")
else:
    plots.plot_psrr(p, annotations=ann, path=FIG/"05_psrr_overlay.png")
plt.show()
"""),
    ]


def nb06():
    return [
        md("# 06 — Temperature sweep (−40 °C … +125 °C)\n\n"
           "Passives scale with TCR. Typical $V_{\\mathrm{ADJ}}(T)$ follows G05. "
           "Envelope WC at each T uses the **25 °C Electrical Characteristics box** "
           "at 25 °C and the **over-temperature box** everywhere else."),
        md(SETUP_MD),
        code(BOOT),
        md(r"""## What this cell plots

Seven figures from one analytical sweep `sweep_analytical(p)` on the MP grid
(−55 °C … +125 °C, 20 °C steps plus the usual corners −40 / 0 / 25 / 85 / 125).

| Figure | File | What it is |
|--------|------|------------|
| $V_{\mathrm{OUT}}$ vs $T$ | `06_vout_vs_t.png` | typical setpoint (black) + WC envelope (red band) |
| Dropout vs $T$ | `06_vdo_vs_t.png` | typ $V_{\mathrm{DO}}(T)$ vs datasheet max-over-temp |
| Temperature family | `06_temp_grid.png` | $V_{\mathrm{OUT}}$, $V_{\mathrm{ADJ}}$, $V_{\mathrm{DO}}$, $I_{\mathrm{GND}}$, $\eta$, $T_j$ |
| $I_{\mathrm{GND}}$ vs $T$ | `06_ignd_vs_t.png` | typ vs max GND-pin current |
| $P_{\mathrm{diss}}$ vs $T$ | `06_pdiss_vs_t.png` | $I_{\mathrm{LOAD}}(V_{\mathrm{IN}}-V_{\mathrm{OUT}})+I_{\mathrm{GND}}V_{\mathrm{IN}}$ |
| $T_j$ vs $T_A$ | `06_tj_vs_t.png` | $T_A + P_{\mathrm{tot}}\theta_{JA}$ vs 125 °C limit |
| PSRR | `06_psrr_vs_t.png` | G21 typ vs min (typ − 10 dB over-temp class) |

### Black line — typical $V_{\mathrm{OUT}}(T)$

At each grid temperature:

1. Scale $R_1$ and $R_2$ from 25 °C with the same TCR (default 100 ppm/°C).
   Same-sign TCR on both resistors leaves $R_1/R_2$ almost unchanged; the
   leftover is only the $\pm 1\,\%$ tolerance, not the TCR itself.
2. Replace $V_{\mathrm{ADJ}}$ with the **typical** G05 curve
   `vadj_typical_vs_temp(T)` (a few millivolts of bow).
3. Evaluate
   \(V_{\mathrm{OUT}}=V_{\mathrm{ADJ}}(1+R_1/R_2)+I_{\mathrm{ADJ}}R_1\)
   with typical $I_{\mathrm{ADJ}}=50\,\mathrm{nA}$.

That is why the black trace is a gentle arch, ~4.92 V at the cold and hot
ends. The fixed 5 V part is 5.000 V typical at 25 °C.

### Red band — worst-case envelope, and the notch at 25 °C

The band is **not** a Monte Carlo cloud and it is **not** a continuous
temperature coefficient of the LT3010 spec. It is the datasheet
**hypercube** of $V_{\mathrm{ADJ}}$, $I_{\mathrm{ADJ}}$, $R_1$, $R_2$
evaluated at each $T$:

\[
V_{\min}=V_{\mathrm{ADJ,lo}}\,(1+R_{1,\mathrm{lo}}/R_{2,\mathrm{hi}})
\qquad
V_{\max}=V_{\mathrm{ADJ,hi}}\,(1+R_{1,\mathrm{hi}}/R_{2,\mathrm{lo}})+I_{\mathrm{ADJ,hi}}R_{1,\mathrm{hi}}
\]

**Which $V_{\mathrm{ADJ}}$ box** is a datasheet switch, not a physical dip:

| Condition | Code | $V_{\mathrm{ADJ}}$ box (Rev. J p. 3) |
|-----------|------|--------------------------------------|
| $\lvert T-25\,^\circ\mathrm{C}\rvert\le 1\,^\circ\mathrm{C}$ | `over_temp=False` | **1.258 … 1.292 V** (25 °C) |
| everywhere else on the MP range | `over_temp=True` | **1.237 … 1.313 V** (over temp) |

The over-temp ADJ limits **already include line, load, and temperature**
(Notes 2, 3, 10). This plot does **not** stack extra line/load on top of
that box (`error_mode='envelope'`).

The V-shaped notch is therefore:

- At **5 °C** and **45 °C** (20 °C grid neighbors of 25 °C) the code still
  uses the **wide** over-temp box → band ≈ 4.731 … 5.175 V.
- At the single grid point **25 °C** it uses the **tighter** 25 °C box →
  band ≈ 4.812 … 5.091 V.
- Matplotlib draws **straight lines** between those three points, so you
  see a notch rather than a step. It is an artifact of a discrete spec
  table + linear interpolation, not a real 25 °C “suck-in” of the silicon.

$I_{\mathrm{ADJ,hi}}$ also grows above ~100 °C (G15), which slightly
widens the hot end of the upper bound; TCR on $R_1,R_2$ is a much smaller
effect because both resistors move together.

### How to read it

- The **green dotted line** is the 5.000 V goal. Typical (black) sits
  the page-3 typical output, 5.000 V, at every $T$ before the temperature bow.
- If the red band at 25 °C is the production screen, the over-temp band
  is the guarantee the customer sees across −55 … +125 °C.
- Dropout, $I_{\mathrm{GND}}$, $P_{\mathrm{diss}}$, and $T_j$ in the
  later axes of this cell do **not** use that ADJ box switch — they use
  the datasheet typ vs max-over-temp columns vs load/temperature.
"""),
        code("""
from p5v_design.temperature import sweep_analytical
sw = sweep_analytical(p)
sw.save(ROOT/"results"/"temp_analytical.csv")
display(sw.frame.head())
ann = [
    dict(text="25 °C", xy=(25, float(sw.frame.loc[sw.frame.temp_c==25, "vout"].iloc[0])),
         offset=(12, 10), arrow=True),
    dict(text="E min −40 °C", xy=(-40, float(sw.frame.iloc[0]["vout"])), offset=(12, -12), arrow=True),
]
plots.plot_temp_sweep(
    sw.frame.temp_c.to_numpy(), sw.frame.vout.to_numpy(),
    vout_lo=sw.frame.vout_wc_lo.to_numpy(), vout_hi=sw.frame.vout_wc_hi.to_numpy(),
    annotations=ann, path=FIG/"06_vout_vs_t.png",
)
plt.show()
"""),
        md(r"""## ngspice and LTspice WC vs temperature

Same ADJ-box switch as the analytical band (25 °C box only at 25 °C).

| Engine | Typical | WC |
|--------|---------|----|
| ngspice behavioral | OP on the **full** $T$ grid, G05 $V_{\mathrm{ADJ}}(T)$ | 16 corners at each $T$ |
| LTspice vendor lib | unbroken ADJ, macromodel $V_{\mathrm{ADJ}}(T)$ | 16 injected corners at **−55 / 25 / 125 °C** only |

LTspice WC is drawn as **error bars**, not a filled band, so three temperatures
do not invent a notch by interpolation. Set `SKIP_SPICE_WC_MC = True` to skip
ngspice; set `RUN_LTSPICE_BATCH = True` (or pre-run
`python LTSpice/run_ltspice_batch.py temp`) for vendor-lib logs.

The vendor macromodel is **typical-only** and is not guaranteed at MP cold.
Expect LTspice typical/WC at −55 °C to sit well below the datasheet envelope
(the lib’s ADJ voltage collapses); 25 °C and 125 °C should track analytical
with the same slight low bias already seen in notebook 05.
"""),
        code("""
from p5v_design.temperature import (
    sweep_ngspice_nominal, sweep_ngspice_wc, sweep_ltspice_from_logs, temp_tag,
)
from p5v_design.simulate import find_ngspice

T_LTSPICE = [-40.0, 25.0, 125.0]
ng_nom = ng_wc = None
lt_nom = lt_wc = None

HAVE_SPICE = False
try:
    find_ngspice()
    HAVE_SPICE = True
except FileNotFoundError as e:
    print(e)

if HAVE_SPICE and not SKIP_SPICE_WC_MC:
    ng_nom = sweep_ngspice_nominal(
        p, workdir=ROOT/"results"/"ngspice_temp", root=ROOT,
    )
    ng_wc = sweep_ngspice_wc(
        p, workdir=ROOT/"results"/"ngspice_wc_temp", root=ROOT,
        max_workers=SPICE_WORKERS,
    )
    ng_nom.to_csv(ROOT/"results"/"temp_ngspice_nom.csv", index=False)
    ng_wc.to_csv(ROOT/"results"/"temp_ngspice_wc.csv", index=False)
    display(ng_wc)
else:
    print("ngspice temperature WC skipped")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_ltspice_batch", ROOT/"LTSpice"/"run_ltspice_batch.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.write_temp_decks(p, temps=T_LTSPICE)
if RUN_LTSPICE_BATCH:
    from p5v_design.ltspice_io import find_ltspice, run_ltspice_decks
    try:
        find_ltspice()
        decks = []
        for T in T_LTSPICE:
            d = ROOT/"LTSpice"/"Temp"/temp_tag(T)
            decks += [d/"P5V_Regulator_op.cir", d/"P5V_Regulator_WC.cir"]
        run_ltspice_decks(decks, max_workers=LTSPICE_WORKERS)
    except FileNotFoundError as e:
        print(e)
else:
    print("LTspice temp batch skipped (RUN_LTSPICE_BATCH=False). Existing logs still parsed.")

lt_nom, lt_wc = sweep_ltspice_from_logs(p, temp_root=ROOT/"LTSpice"/"Temp", temps=T_LTSPICE)
display(lt_nom)
display(lt_wc)

kw = dict(
    vout_lo=sw.frame.vout_wc_lo.to_numpy(), vout_hi=sw.frame.vout_wc_hi.to_numpy(),
)
if ng_nom is not None:
    kw["ng_temps"] = ng_nom.temp_c.to_numpy()
    kw["ng_vout"] = ng_nom.vout.to_numpy()
if ng_wc is not None:
    kw["ng_lo"] = ng_wc.vout_lo.to_numpy()
    kw["ng_hi"] = ng_wc.vout_hi.to_numpy()
if lt_nom is not None and lt_nom.vout.notna().any():
    kw["lt_temps"] = lt_nom.temp_c.to_numpy()
    kw["lt_vout"] = lt_nom.vout.to_numpy()
if lt_wc is not None and lt_wc.vout_lo.notna().any():
    kw["lt_lo"] = lt_wc.vout_lo.to_numpy()
    kw["lt_hi"] = lt_wc.vout_hi.to_numpy()
ann_ov = [
    dict(text="25 °C box (notch)", xy=(25.0, float(sw.frame.loc[np.isclose(sw.frame.temp_c, 25), "vout_wc_hi"].iloc[0])),
         offset=(14, 12), arrow=True, color="C3"),
    dict(text="5.000 V", xy=(80.0, 5.0), offset=(8, 10), arrow=True, color="#2ca02c"),
]
plots.plot_temp_sweep(
    sw.frame.temp_c.to_numpy(), sw.frame.vout.to_numpy(),
    annotations=ann_ov, path=FIG/"06_vout_vs_t_engines.png",
    title=r"$V_{\mathrm{OUT}}$ vs $T$  analytical / ngspice / LTspice",
    **kw,
)
plt.show()

# Three-temperature comparison table
rows = []
for T in T_LTSPICE:
    an = sw.frame.loc[np.isclose(sw.frame.temp_c, T)].iloc[0]
    rec = {"T_C": T, "an_nom": an.vout, "an_lo": an.vout_wc_lo, "an_hi": an.vout_wc_hi}
    if ng_nom is not None:
        rec["ng_nom"] = float(ng_nom.loc[np.isclose(ng_nom.temp_c, T), "vout"].iloc[0])
    if ng_wc is not None:
        rec["ng_lo"] = float(ng_wc.loc[np.isclose(ng_wc.temp_c, T), "vout_lo"].iloc[0])
        rec["ng_hi"] = float(ng_wc.loc[np.isclose(ng_wc.temp_c, T), "vout_hi"].iloc[0])
    if lt_nom is not None and lt_nom.vout.notna().any():
        rec["lt_nom"] = float(lt_nom.loc[np.isclose(lt_nom.temp_c, T), "vout"].iloc[0])
    if lt_wc is not None and lt_wc.vout_lo.notna().any():
        rec["lt_lo"] = float(lt_wc.loc[np.isclose(lt_wc.temp_c, T), "vout_lo"].iloc[0])
        rec["lt_hi"] = float(lt_wc.loc[np.isclose(lt_wc.temp_c, T), "vout_hi"].iloc[0])
    rows.append(rec)
cmp = pd.DataFrame(rows)
if "ng_nom" in cmp.columns:
    cmp["ng-an_mV"] = 1e3 * (cmp["ng_nom"] - cmp["an_nom"])
display(cmp)
"""),
        md("## Dropout, $I_{\\mathrm{GND}}$, dissipation, $T_j$, PSRR vs $T$"),
        code("""
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.plot(sw.frame.temp_c, 1e3*sw.frame.vdo_typ, "k-", lw=1.7, label="Vdo typ")
ax.plot(sw.frame.temp_c, 1e3*sw.frame.vdo_wc, "C3--", lw=1.3, label="Vdo max over temp")
ax.set_xlabel("T (°C)")
ax.set_ylabel("dropout (mV)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plots.apply_annotations(ax, [dict(text="50 mA", xy=(25, 1e3*sw.frame.vdo_typ.iloc[len(sw.frame)//2]), offset=(10, 10), arrow=True)])
plots.savefig(fig, FIG/"06_vdo_vs_t.png")
plt.show()
ann_g = [dict(text="25 °C", xy=(25, float(sw.frame.loc[np.isclose(sw.frame.temp_c, 25), "vout"].iloc[0])), offset=(10, 12), arrow=True)]
plots.plot_temp_grid(sw.frame, annotations=ann_g, path=FIG/"06_temp_grid.png")
plt.show()
fig, ax = plt.subplots(figsize=(8.0, 3.8))
ax.plot(sw.frame.temp_c, 1e3*sw.frame.ignd_typ, "k-", lw=1.7, label="typ")
ax.plot(sw.frame.temp_c, 1e3*sw.frame.ignd_max, "C3--", lw=1.3, label="max")
ax.set_xlabel("T (°C)"); ax.set_ylabel("Ignd (mA)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plots.savefig(fig, FIG/"06_ignd_vs_t.png"); plt.show()
fig, ax = plt.subplots(figsize=(8.0, 3.8))
ax.plot(sw.frame.temp_c, 1e3*sw.frame.p_diss, "k-", lw=1.7)
ax.set_xlabel("T (°C)"); ax.set_ylabel("Pdiss (mW)")
ax.set_title("LDO dissipation vs T (typ Ignd)")
ax.grid(True, alpha=0.3)
plots.savefig(fig, FIG/"06_pdiss_vs_t.png"); plt.show()
fig, ax = plt.subplots(figsize=(8.0, 3.8))
ax.plot(sw.frame.temp_c, sw.frame.tj_c, "k-", lw=1.7, label="Tj")
ax.axhline(p.ldo.tj_max_c, color="C3", ls="--", label="Tjmax")
ax.set_xlabel("T (°C)"); ax.set_ylabel("°C"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("Junction temperature vs ambient")
plots.savefig(fig, FIG/"06_tj_vs_t.png"); plt.show()
# PSRR min over-temp is datasheet typ−10 dB (page 3)
f = np.logspace(1, 6, 241)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.semilogx(f, psrr_db(f, cout=p.cout), "k-", lw=1.8, label="typ 25 °C (G21)")
ax.semilogx(f, psrr_db(f, cout=p.cout, typ=False), "C3--", lw=1.4, label="min (over temp class)")
ax.axvline(120, color="0.5", ls=":", lw=0.9)
ax.set_xlabel("f (Hz)"); ax.set_ylabel("PSRR (dB)")
ax.set_title("PSRR  25 °C typ vs min (over-temp class)")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
plots.savefig(fig, FIG/"06_psrr_vs_t.png"); plt.show()
"""),
    ]


def nb07():
    return [
        md("# 07 — Sensitivity\n\n"
           "OAT tornado and normalized $S_x^{V_{\\mathrm{OUT}}}=(x/V)\\,\\partial V/\\partial x$. "
           "Use this to decide whether 0.1 % feedback resistors are worth it."),
        md(SETUP_MD),
        code(BOOT),
        code("""
from p5v_design.sensitivity import full_sensitivity
sens = full_sensitivity(p)
display(sens.oat)
display(sens.normalized)
ann = []
torn = dict(zip(sens.oat["name"], sens.oat["half_span_vout_mV"]))
plots.plot_tornado(torn, xlabel="half-span (mV)", title="Vout sensitivity",
                   annotations=ann, path=FIG/"07_tornado.png")
plt.show()
# What-if 0.1 % resistors
p01 = p.with_values(r_tol=ToleranceSpec(0.001))
s01 = full_sensitivity(p01)
print("RSS 1 %", np.sqrt(np.sum(np.square(sens.oat.half_span_vout_mV))))
print("RSS 0.1 %", np.sqrt(np.sum(np.square(s01.oat.half_span_vout_mV))))
plots.plot_oat_share(sens.oat, path=FIG/"07_oat_share.png")
plt.show()
from p5v_design.worst_case import oat_contributions
a = oat_contributions(p)["vout"]
b = oat_contributions(p01)["vout"]
plots.plot_tornado_compare(a, b, path=FIG/"07_tornado_1pct_vs_0p1.png")
plt.show()
display(sens.normalized)
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.barh(sens.normalized["name"], sens.normalized["S"], color="C2", edgecolor="k", lw=0.4)
ax.set_xlabel(r"$S_x^{V_{OUT}}=(x/V)\,\partial V/\partial x$")
ax.set_title("Normalized DC sensitivity")
ax.grid(True, axis="x", alpha=0.3)
plots.savefig(fig, FIG/"07_normalized_S.png")
plt.show()
# Other-metric tornados (HV_monitor 07)
from p5v_design.worst_case import oat_contributions as oat_c
oat_all = oat_c(p, metrics=("vout", "headroom", "p_diss", "efficiency"))
for met, xlab, fn in (
    ("headroom", "half-span (V)", "07_tornado_headroom.png"),
    ("p_diss", "half-span (W)", "07_tornado_pdiss.png"),
    ("efficiency", "half-span", "07_tornado_eta.png"),
):
    plots.plot_tornado(oat_all[met], xlabel=xlab, title=f"OAT tornado — {met}", path=FIG/fn)
    plt.show()
s_ot = full_sensitivity(p, over_temp=True)
plots.plot_tornado_compare(
    dict(zip(sens.oat["name"], sens.oat["half_span_vout"])),
    dict(zip(s_ot.oat["name"], s_ot.oat["half_span_vout"])),
    label_a="25 °C", label_b="over temp",
    title="Vout sensitivity  25 °C vs over-temp envelope",
    path=FIG/"07_tornado_25_vs_ot.png",
)
plt.show()
plots.plot_psrr_vs_cout(p, annotations=[
    dict(text="BOM 10 µF", xy=(10.0, 75), offset=(12, -14), arrow=True),
], path=FIG/"07_psrr_vs_cout.png")
plt.show()
"""),
    ]


def nb08():
    return [
        md(r"""# 08 — Derating, thermal, component stress

Datasheet: $P = I_{\mathrm{OUT}}(V_{\mathrm{IN}}-V_{\mathrm{OUT}}) + I_{\mathrm{GND}} V_{\mathrm{IN}}$.
θJA from Table 1 vs copper area. SOA at $T_A=25$ and $85\,^\circ\mathrm{C}$.

Horizontal **stress bars** match comparator notebook 06:

| Chart | Axis | Guideline |
|-------|------|-----------|
| Power dissipation | mW | — |
| Power utilization | $P/P_{\mathrm{rated}}$ (%) | 50 % use |
| Voltage utilization | $V / (0.80\,V_{\max})$ (%) | 100 % = 80 % derate |
| Thermal utilization (U1) | $P / P_{\mathrm{Tjmax}}$ (%) | 100 % = $T_j=125\,^\circ\mathrm{C}$ |
| $\Delta T$ | °C | — |

Grouped bars repeat the same metrics at **5.4 / 12 / 24 V** (inside the datasheet 5.4 V to 80 V range)
and **1 / 35 / 50 mA**.
"""),
        md(SETUP_MD),
        code(BOOT),
        md(r"""## Power / voltage table at the 12 V / 50 mA bias

Ambient $T_A=25\,^\circ\mathrm{C}$. Status: `OK`, `P>50%`, `V>80%`, `Tj>max`.
CSV: `results/thermal_operate.csv`.
"""),
        code("""
from p5v_design.thermal import power_table, copper_table, power_tables_vs_vin, power_tables_vs_load, bom_only
from p5v_design.derating import power_derating_curve

pt = power_table(p, ta_C=25.0)
bom = bom_only(pt)
display(pt[[
    "ref", "package", "P_mW", "P_rated_W", "utilization_pct",
    "V_applied_V", "V_derated_V", "V_util_derated_pct",
    "P_util_thermal_pct", "dT_C", "Tj_C", "status", "note",
]])
pt.to_csv(ROOT/"results"/"thermal_operate.csv", index=False)
display(copper_table())
"""),
        md(r"""## Stress bars at operate — power, voltage, ΔT (comparator 06)

R1/R2 dissipation is tiny (divider ~0.6 mA). **U1** is the thermal part.
Cout/Cin voltage stress uses the 5 V output and the 12 V bias. The datasheet input can be as high as 80 V, so the input capacitor voltage rating is an assumption, not a value printed on the example.

Dummy `Rload` is an Iload equivalent, not a board part — it is omitted from
voltage / utilization charts. Black ticks on the applied-voltage chart are
**80% derated** $V_{\max}$; gray ticks are catalog $V_{\max}$.
"""),
        code("""
ann_p = [dict(text="U1 is the wattage", xy=(float(pt.loc[pt.ref=="U1", "P_mW"].iloc[0]), "U1"), offset=(12, 10), arrow=True)]
plots.plot_stress_bars(
    pt, value_col="P_mW", xlabel="Power (mW)", title="BOM power dissipation at 12 V / 50 mA",
    color="C0", annotations=ann_p, path=FIG/"08_stress_PmW.png",
)
plt.show()
plots.plot_stress_bars(
    bom[bom["P_rated_W"] > 0], value_col="utilization_pct",
    xlabel="P / P_rated (%)", title="Resistor power utilization",
    vline=50, vline_label="50% use guideline", color="C4",
    path=FIG/"08_stress_Putil.png",
)
plt.show()
plots.plot_stress_bars(
    bom[bom["V_max_V"] > 0], value_col="V_util_derated_pct",
    xlabel="V / (0.80 Vmax) (%)", title="Voltage stress vs 80% derated rating",
    vline=100, vline_label="100% of 80% derate",
    vline2=80, vline2_label="80% of derated V (extra margin)",
    color="C1", path=FIG/"08_stress_Vutil.png",
)
plt.show()
plots.plot_applied_voltage(
    bom, title="Applied voltage at 12 V / 50 mA", path=FIG/"08_stress_Vapplied.png",
)
plt.show()
plots.plot_stress_bars(
    bom[bom["dT_C"].notna()], value_col="dT_C",
    xlabel="ΔT (°C)", title="Junction / package temperature rise",
    color="C3", path=FIG/"08_stress_dT.png",
)
plt.show()
plots.plot_stress_bars(
    bom[bom["P_util_thermal_pct"].notna()], value_col="P_util_thermal_pct",
    xlabel="P / P(Tj=125 °C) (%)", title="Thermal-budget utilization",
    vline=100, vline_label="Tjmax", color="C5",
    path=FIG/"08_stress_Pthermal.png",
)
plt.show()
plots.plot_cap_voltage_bars(bom, path=FIG/"08_cap_voltage.png")
plt.show()
"""),
        md(r"""## Grouped bars across the datasheet input range and vs load

Same idea as comparator 06 (operate / UVLO / full-scale): here **5.4 / 12 / 24 V**
and **1 / 35 / 50 mA**. U1 $P$ and $\Delta T$ move; the divider barely does.
"""),
        code("""
vin_tabs = power_tables_vs_vin(p, ta_C=25.0)
load_tabs = power_tables_vs_load(p, ta_C=25.0)
vin_bom = {k: bom_only(df) for k, df in vin_tabs.items()}
load_bom = {k: bom_only(df) for k, df in load_tabs.items()}
for key, df in vin_tabs.items():
    df.to_csv(ROOT/"results"/f"thermal_{key.replace(' ', '')}.csv", index=False)
plots.plot_stress_grouped(
    vin_tabs, value_col="P_mW", xlabel="Power (mW)",
    title="Power vs Vin (5.4 / 12 / 24 V)", path=FIG/"08_stress_P_vs_vin.png",
)
plt.show()
plots.plot_stress_grouped(
    vin_bom, value_col="utilization_pct", xlabel="P / P_rated (%)",
    title="Power utilization vs Vin", vline=50, vline_label="50% use",
    path=FIG/"08_stress_Putil_vs_vin.png",
)
plt.show()
plots.plot_stress_grouped(
    vin_bom, value_col="V_util_derated_pct", xlabel="V / (0.80 Vmax) (%)",
    title="Voltage stress vs Vin", vline=100, vline_label="100% of 80% derate",
    vline2=80, vline2_label="80% of derated V",
    path=FIG/"08_stress_V_vs_vin.png",
)
plt.show()
plots.plot_applied_voltage_grouped(
    vin_bom, title="Applied voltage vs Vin (5.4 / 12 / 24 V)",
    path=FIG/"08_stress_Vapplied_vs_vin.png",
)
plt.show()
plots.plot_stress_grouped(
    vin_bom, value_col="dT_C", xlabel="ΔT (°C)",
    title="ΔT vs Vin", path=FIG/"08_stress_dT_vs_vin.png",
)
plt.show()
plots.plot_stress_grouped(
    vin_bom, value_col="P_util_thermal_pct", xlabel="P / P(Tj=125 °C) (%)",
    title="Thermal-budget utilization vs Vin", vline=100, vline_label="Tjmax",
    path=FIG/"08_stress_Pthermal_vs_vin.png",
)
plt.show()
plots.plot_stress_grouped(
    load_tabs, value_col="P_mW", xlabel="Power (mW)",
    title="Power vs load (1 / 35 / 50 mA)", path=FIG/"08_stress_P_vs_load.png",
)
plt.show()
plots.plot_stress_grouped(
    load_bom, value_col="utilization_pct", xlabel="P / P_rated (%)",
    title="Power utilization vs load", vline=50, vline_label="50% use",
    path=FIG/"08_stress_Putil_vs_load.png",
)
plt.show()
plots.plot_stress_grouped(
    load_bom, value_col="V_util_derated_pct", xlabel="V / (0.80 Vmax) (%)",
    title="Voltage stress vs load", vline=100, vline_label="80% derate",
    path=FIG/"08_stress_V_vs_load.png",
)
plt.show()
plots.plot_applied_voltage_grouped(
    load_bom, title="Applied voltage vs load (1 / 35 / 50 mA)",
    path=FIG/"08_stress_Vapplied_vs_load.png",
)
plt.show()
plots.plot_stress_grouped(
    load_bom, value_col="dT_C", xlabel="ΔT (°C)",
    title="ΔT vs load", path=FIG/"08_stress_dT_vs_load.png",
)
plt.show()
plots.plot_stress_grouped(
    load_bom, value_col="P_util_thermal_pct", xlabel="P / P(Tj=125 °C) (%)",
    title="Thermal-budget utilization vs load", vline=100, vline_label="Tjmax",
    path=FIG/"08_stress_Pthermal_vs_load.png",
)
plt.show()
"""),
        md(r"""## SOA, $T_j$ map, copper area, 0805 derating curve"""),
        code("""
ann_soa = [dict(text="12 V / 50 mA", xy=(12.0, 35.0), offset=(14, 12), arrow=True, color="w")]
plots.plot_soa(p, ta_c=25.0, annotations=ann_soa, path=FIG/"08_soa_25C.png")
plt.show()
plots.plot_soa(p, ta_c=85.0, annotations=[dict(text="85 °C ambient", xy=(9, 35), offset=(12, 10), arrow=True)],
               path=FIG/"08_soa_85C.png")
plt.show()
ann_map = [dict(text="12 V / 50 mA", xy=(12.0, 50.0), offset=(12, 10), arrow=True, color="w")]
plots.plot_tj_map(p, ta_c=25.0, annotations=ann_map, path=FIG/"08_tj_map.png")
plt.show()
plots.plot_copper_rth(annotations=[dict(text="2500 mm² → 40 °C/W", xy=(2500, 40), offset=(-80, 14), arrow=True)],
                      path=FIG/"08_copper.png")
plt.show()
cur = power_derating_curve(p)
fig, ax = plt.subplots(figsize=(8.0, 3.8))
ax.plot(cur.temp_c, cur.p_rated_derated_W, "k-", lw=1.7, label="rated (derated)")
ax.plot(cur.temp_c, cur.p_use_W, "C3--", lw=1.3, label="50 % use")
ax.set_xlabel("T (°C)"); ax.set_ylabel("W")
ax.set_title("0805 resistor power derating")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plots.savefig(fig, FIG/"08_r_derate.png"); plt.show()
"""),
        md(r"""## Ceramic DC-bias (datasheet Figs. 3–4)

A 16 V 10 µF **Y5V** can look like 1–2 µF at 5 V. **X7R/X5R** is required for
Cout. This design uses 10 µF / 5 mΩ — stay on X7R and verify C at 5 V DC.
"""),
        code("""
ann_c = [dict(text="5 V Cout", xy=(5.0, 85), offset=(12, 10), arrow=True)]
plots.plot_ceramic_dc_bias(annotations=ann_c, path=FIG/"08_ceramic_dc_bias.png")
plt.show()
"""),
    ]


def nb09():
    return [
        md("# 09 — Input current and efficiency\n\n"
           "The input source sees $I_{\\mathrm{IN}} = I_{\\mathrm{OUT}} + I_{\\mathrm{GND}}$. "
           "At 12 V in and 50 mA out, a 5 V linear regulator is about 42 % efficient — expected."),
        md(SETUP_MD),
        code(BOOT),
        code("""
from p5v_design.supply_budget import rail_budget, budget_vs_load, budget_vs_vin
b = rail_budget(p)
bmax = rail_budget(p, which="max")
typ = analysis.metrics_table(b).rename(columns={"value": "typ"})
mx = analysis.metrics_table(bmax).rename(columns={"value": "max Ignd"})
op_tbl = typ.join(mx[["max Ignd"]])[["typ", "max Ignd", "unit"]]
display(analysis.style_metrics(op_tbl))
bl = budget_vs_load(p)
display(analysis.style_metrics(analysis.wide_metrics_table(bl, header="iload")))
bl.to_csv(ROOT/"results"/"supply_budget.csv", index=False)
ann = [dict(text="50 mA load", xy=(p.op.vin_nom, 100*b["efficiency"]), offset=(12, 10), arrow=True)]
plots.plot_efficiency(p, annotations=ann, path=FIG/"09_eta.png")
plt.show()
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.plot(1e3*bl.iload, 1e3*bl.iin, "k-", lw=1.7, label="Iin")
ax.plot(1e3*bl.iload, 1e3*bl.iload, "C0--", lw=1.2, label="Iout")
ax.plot(1e3*bl.iload, 1e3*bl.ignd, "C3-.", lw=1.3, label="Ignd")
ax.set_xlabel("Iout (mA)")
ax.set_ylabel("mA")
ax.set_title("Current drawn from VIN")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plots.apply_annotations(ax, [dict(text="Ignd << Iout at 50 mA", xy=(50, 1.3), offset=(10, 12), arrow=True)])
plots.savefig(fig, FIG/"09_iin.png")
plt.show()
plots.plot_rail_stack(b, path=FIG/"09_rail_stack.png")
plt.show()
bv = budget_vs_vin(p)
bvmax = budget_vs_vin(p, which="max")
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.plot(bv.vin, 1e3*bv.iin, "k-", lw=1.7, label="Iin typ")
ax.plot(bvmax.vin, 1e3*bvmax.iin, "C3--", lw=1.4, label="Iin max Ignd")
ax.set_xlabel("Vin (V)"); ax.set_ylabel("mA"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("Input current vs Vin (typ vs max Ignd)")
plots.savefig(fig, FIG/"09_iin_vs_vin.png"); plt.show()
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.stackplot(bv.vin, bv.p_out, bv.p_diss, labels=["Pout", "Pdiss"], colors=["#2ca02c", "#ff7f0e"], alpha=0.85)
ax.set_xlabel("Vin (V)"); ax.set_ylabel("W"); ax.legend(fontsize=8)
ax.set_title("Pin = Pout + Pdiss")
ax.grid(True, alpha=0.3)
plots.savefig(fig, FIG/"09_pin_stack.png"); plt.show()
plots.plot_budget_grouped(b, bmax, path=FIG/"09_budget_grouped.png")
plt.show()
plots.plot_efficiency_vs_load(p, annotations=[
    dict(text="50 mA", xy=(50.0, 100*b["efficiency"]), offset=(12, 10), arrow=True),
], path=FIG/"09_eta_vs_load.png")
plt.show()
# Stacked current at 8 / 9 / 12 V (comparator 07 analog)
vins = [5.4, 12.0, 24.0]
iout = [1e3 * rail_budget(p, vin=v)["iload"] for v in vins]
ig = [1e3 * rail_budget(p, vin=v)["ignd"] for v in vins]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(len(vins))
ax.bar(x, iout, color="C0", edgecolor="k", lw=0.4, label="Iout")
ax.bar(x, ig, bottom=iout, color="C3", edgecolor="k", lw=0.4, label="Ignd")
ax.set_xticks(x); ax.set_xticklabels([f"{v:g} V" for v in vins])
ax.set_ylabel("mA"); ax.set_title(r"$I_{\mathrm{IN}}=I_{\mathrm{OUT}}+I_{\mathrm{GND}}$")
ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
plots.savefig(fig, FIG/"09_iin_stacked.png"); plt.show()
"""),
    ]


def nb10():
    return [
        md("# 10 — Time domain\n\n"
           "Analytical Cout/ESR estimates vs ngspice behavioral vs (optional) LTspice "
           "vendor model. Load-step recovery time `tau_loop` is an **editable** "
           "stand-in; G26 is ~200 µs with 1 µF."),
        md(SETUP_MD),
        code(BOOT),
        code("""
from p5v_design.timedomain import load_step, startup, line_step
from p5v_design.simulate import find_ngspice, simulate_tran

TAU_LOOP = 80e-6   # editable recovery time constant
ls = load_step(p, tau_loop=TAU_LOOP)
ann = [
    dict(text="1 → 50 mA", xy=(0.05, ls.vout.min()), offset=(12, -14), arrow=True),
    dict(text="ESR spike", xy=(0.05, ls.vout.min()), offset=(40, 20), arrow=True),
]
plots.plot_tran(ls.t, ls.vout, iout=ls.iout, annotations=ann, path=FIG/"10_load_step.png",
                title="Load step 1→50 mA (analytical)", t_scale=1e3, t_label="t (ms)")
plt.show()
st = startup(p)
ann_s = [dict(text="Ilim charge", xy=(0.2, 2.0), offset=(12, 10), arrow=True)]
plots.plot_tran(st.t, st.vout, vin=st.vin, annotations=ann_s, path=FIG/"10_startup.png",
                title="Startup (Ilim onto Cout)", t_scale=1e3)
plt.show()
ln = line_step(p)
ann_l = [dict(text="9 → 12 V", xy=(0.1, ln.vout[0]), offset=(12, 10), arrow=True)]
plots.plot_tran(ln.t, ln.vout, vin=ln.vin, annotations=ann_l, path=FIG/"10_line_step.png",
                title="Line step", t_scale=1e3)
plt.show()
try:
    find_ngspice()
    sp = simulate_tran(p, kind="load", workdir=ROOT/"results"/"ngspice_nom"/"tran_load")
    plots.plot_tran(ls.t, ls.vout, spice_t=sp.t, spice_v=sp.vout, iout=ls.iout,
                    annotations=[dict(text="ngspice overlay", xy=(0.2, analysis.vout_dc(p)), offset=(10, 10), arrow=True)],
                    path=FIG/"10_load_spice.png", title="Load step vs ngspice")
    plt.show()
except FileNotFoundError as e:
    print(e)
except Exception as e:
    print("ngspice load step failed:", e)
try:
    find_ngspice()
    sp_ln = simulate_tran(p, kind="line", workdir=ROOT/"results"/"ngspice_nom"/"tran_line")
    plots.plot_tran(ln.t, ln.vout, spice_t=sp_ln.t, spice_v=sp_ln.vout, vin=ln.vin,
                    path=FIG/"10_line_spice.png", title="Line step vs ngspice")
    plt.show()
    sp_st = simulate_tran(p, kind="startup", workdir=ROOT/"results"/"ngspice_nom"/"tran_start")
    plots.plot_tran(st.t, st.vout, spice_t=sp_st.t, spice_v=sp_st.vout, vin=st.vin,
                    path=FIG/"10_startup_spice.png", title="Startup vs ngspice")
    plt.show()
except Exception as e:
    print("ngspice extra transients:", e)
dip = float(analysis.vout_dc(p) - ls.vout.min())
print(f"analytical load-step undershoot {1e3*dip:.1f} mV")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.barh(["undershoot (mV)", "ESR spike (mV)", "Cout dip (mV)"],
        [1e3*dip, 1e3*(p.op.iload_op-1e-3)*p.esr_out, 1e3*(p.op.iload_op-1e-3)*80e-6/p.cout],
        color=["C3", "C0", "C1"], edgecolor="k")
ax.set_xlabel("mV")
ax.set_title("Load-step budget (analytical)")
ax.grid(True, axis="x", alpha=0.3)
plots.savefig(fig, FIG/"10_step_budget.png"); plt.show()
# Load-step family (HV_monitor 01 analog)
fam = {}
for i1, lab in ((10e-3, "1→10 mA"), (50e-3, "1→50 mA"), (50e-3, "1→50 mA")):
    tr = load_step(p, i0=1e-3, i1=i1, tau_loop=TAU_LOOP)
    fam[lab] = (tr.t, tr.vout)
plots.plot_tran_family(fam, title="Load-step family (analytical)", path=FIG/"10_load_family.png")
plt.show()
fall = load_step(p, i0=p.op.iload_op, i1=1e-3, tau_loop=TAU_LOOP)
plots.plot_tran(fall.t, fall.vout, iout=fall.iout,
                path=FIG/"10_load_fall.png", title="Load step 35→1 mA (analytical)",
                t_scale=1e3, t_label="t (ms)")
plt.show()
esr_fam = {}
for esr, lab in ((1e-3, "1 mΩ"), (5e-3, "5 mΩ BOM"), (50e-3, "50 mΩ"), (300e-3, "300 mΩ max")):
    pc = replace(p, esr_out=esr)
    tr = load_step(pc, tau_loop=TAU_LOOP)
    esr_fam[lab] = (tr.t, tr.vout)
plots.plot_tran_family(esr_fam, title="Load step vs Cout ESR", path=FIG/"10_esr_family.png")
plt.show()
c_fam = {}
for c, lab in ((1e-6, "1 µF"), (3.3e-6, "3.3 µF"), (10e-6, "10 µF BOM"), (22e-6, "22 µF")):
    pc = replace(p, cout=c)
    tr = load_step(pc, tau_loop=TAU_LOOP)
    c_fam[lab] = (tr.t, tr.vout)
plots.plot_tran_family(c_fam, title="Load step vs Cout", path=FIG/"10_cout_family.png")
plt.show()
"""),
    ]


def main() -> None:
    write_nb("01_analytical_setpoint.ipynb", nb01())
    write_nb("02_ngspice_vs_analytical.ipynb", nb02())
    write_nb("03_worst_case.ipynb", nb03())
    write_nb("04_monte_carlo.ipynb", nb04())
    write_nb("05_ltspice_nominal_mc_wc.ipynb", nb05())
    write_nb("06_temperature_sweep.ipynb", nb06())
    write_nb("07_sensitivity.ipynb", nb07())
    write_nb("08_derating_thermal.ipynb", nb08())
    write_nb("09_supply_budget.ipynb", nb09())
    write_nb("10_time_domain.ipynb", nb10())


if __name__ == "__main__":
    main()
