"""Generate ngspice (and LTspice-twin) netlists for the P5V ISO regulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from .ldo_model import dropout_v, ignd_a
from .params import CircuitParams

DEFAULT_BEHAV_LIB = Path("models") / "lt3010" / "lt3010_behav.lib"


@dataclass
class NetlistOptions:
    ldo_model: str = "behav"  # "behav"
    analysis: str = "op"  # op | dc_vin | dc_i | ac_psrr | tran_startup | tran_load | tran_line
    include_nonideals: bool = True
    ac_fstart: float = 10.0
    ac_fstop: float = 1e6
    ac_points: int = 40
    temp_c: float = 25.0
    title: str = "LT3010-5 datasheet 5 V / 50 mA"
    vin_start: float = 3.0
    vin_stop: float = 12.0
    vin_step: float = 0.05
    i_start: float = 0.0
    i_stop: float = 0.05
    i_step: float = 0.0005
    tran_tstep: float = 1e-6
    tran_tstop: float = 2e-3
    load_i0: float = 1e-3
    load_i1: float = 50e-3
    line_v0: float = 12.0
    line_v1: float = 24.0
    spice_engine: str = "ngspice"  # ngspice | ltspice


def _param_block(p: CircuitParams) -> str:
    sp = p.as_spice_params()
    vdo = dropout_v(p, which="typ")
    ign = ignd_a(p, which="typ")
    sp["VDO"] = vdo
    sp["IGN0"] = ign
    return "\n".join(f".param {k}={v:.12g}" for k, v in sp.items())


def behav_subckt() -> str:
    """Behavioral LDO: divider is EXTERNAL so R1/R2 tolerances move Vout.

    Loop: high-gain error amp servos V(ADJ) to VADJ.
    Dropout clamp: V(OUT) <= V(VIN) - VDO.
    Line: LINEK * (Vin - VINNOM) added to the reference (page-3 line reg).
    PSRR AC: Vin filtered by Rpsr/Cpsr before LINEK.
    IADJ flows into the ADJ pin.
    """
    return dedent(
        """\
        * Behavioral LT3010 (ngspice). Pins: OUT ADJ GND SHDN VIN
        .subckt LT3010_BEHAV OUT ADJ GND SHDN VIN
        .param GM=5e-3 RCOMP=2e7 CC=5n ROUT=0.05
        * Reference + line term (PSRR / line regulation)
        Rpsr VIN vin_lp 1k
        Cpsr vin_lp GND 160n
        Bref nref GND V = {VADJ} + {LINEK}*(V(vin_lp) - {VINNOM})
        * ADJ bias current into the pin
        Iadj ADJ GND {IADJ}
        * Error amp
        Gerr GND ncomp nref ADJ {GM}
        Rcomp ncomp GND {RCOMP}
        Ccomp ncomp GND {CC}
        * Pass element: voltage source + small Rout, dropout clamp
        Bout npre GND V = min(V(VIN) - {VDO}, max(0, V(ncomp)))
        Rout npre OUT {ROUT}
        * GND pin current (operate-point value)
        Ignd VIN GND {IGN0}
        * SHDN: unused (tied high in this application)
        Rshdn SHDN GND 100meg
        .ends LT3010_BEHAV
        """
    )


def _circuit_body(p: CircuitParams | None = None) -> str:
    """External network. A fixed LT3010-5 (r1 == 0) ties SENSE/ADJ to OUT."""
    cff_line = "Cff OUT ADJ {CFF}\n" if p is None or p.cff > 1e-15 else "* no Cff on the fixed 5 V example\n"
    r1_line = "R1 OUT ADJ 1u\n" if p is not None and p.r1 <= 1e-9 else "R1 OUT ADJ {R1}\n"
    return (
        "Vin VIN 0 {VIN}\n"
        "Resrin VIN ncin {ESRIN}\n"
        "Cin ncin 0 {CIN}\n"
        "XU1 OUT ADJ 0 VIN VIN LT3010_BEHAV\n"
        + r1_line
        + "R2 ADJ 0 {R2}\n"
        + cff_line
        + "Resrout OUT ncout {ESROUT}\n"
        + "Cout ncout 0 {COUT}\n"
        + "Rload OUT 0 {RLOAD}\n"
    )


def _analysis_ngspice(opt: NetlistOptions) -> str:
    if opt.analysis == "op":
        return dedent(
            f"""\
            .temp {opt.temp_c}
            .op
            .control
            run
            print v(out) v(adj) v(vin)
            wrdata op_nodes v(out) v(adj) v(vin)
            quit
            .endc
            """
        )
    if opt.analysis == "dc_vin":
        return dedent(
            f"""\
            .temp {opt.temp_c}
            .dc Vin {opt.vin_start} {opt.vin_stop} {opt.vin_step}
            .control
            run
            wrdata dc_vin v(out) v(vin) v(adj)
            quit
            .endc
            """
        )
    if opt.analysis == "dc_i":
        # Replace Rload with a current source for an Iload sweep
        return (
            f"* Iload sweep uses Iload current source; Rload still present\n"
            f"Iload OUT 0 0\n"
            f".temp {opt.temp_c}\n"
            f".dc Iload {opt.i_start} {opt.i_stop} {opt.i_step}\n"
            ".control\n"
            "run\n"
            "wrdata dc_i v(out) v(adj)\n"
            "quit\n"
            ".endc\n"
        )
    if opt.analysis == "ac_psrr":
        return dedent(
            f"""\
            Vin VIN 0 DC {{VIN}} AC 1
            .temp {opt.temp_c}
            .ac dec {opt.ac_points} {opt.ac_fstart} {opt.ac_fstop}
            .control
            run
            wrdata vout_complex real(v(out)) imag(v(out)) real(v(vin)) imag(v(vin))
            wrdata vout_ac vdb(out) vp(out)
            quit
            .endc
            """
        )
    if opt.analysis == "tran_startup":
        return dedent(
            f"""\
            .temp {opt.temp_c}
            .tran {opt.tran_tstep} {opt.tran_tstop} uic
            .control
            run
            wrdata vout_tran v(out) v(vin)
            quit
            .endc
            """
        )
    if opt.analysis == "tran_load":
        return (
            f"Istep OUT 0 PWL(0 {opt.load_i0} 100u {opt.load_i0} 100.5u {opt.load_i1} {opt.tran_tstop} {opt.load_i1})\n"
            f".temp {opt.temp_c}\n"
            f".tran {opt.tran_tstep} {opt.tran_tstop}\n"
            ".control\n"
            "run\n"
            "wrdata vout_tran v(out) v(vin)\n"
            "quit\n"
            ".endc\n"
        )
    if opt.analysis == "tran_line":
        return (
            f"Vin VIN 0 PWL(0 {opt.line_v0} 100u {opt.line_v0} 101u {opt.line_v1} {opt.tran_tstop} {opt.line_v1})\n"
            f".temp {opt.temp_c}\n"
            f".tran {opt.tran_tstep} {opt.tran_tstop}\n"
            ".control\n"
            "run\n"
            "wrdata vout_tran v(out) v(vin)\n"
            "quit\n"
            ".endc\n"
        )
    raise ValueError(f"Unknown analysis {opt.analysis!r}")


def _analysis_ltspice(opt: NetlistOptions) -> str:
    if opt.analysis == "op":
        return f".temp {opt.temp_c}\n.op\n.meas OP vout_op FIND V(OUT)\n.meas OP vadj_op FIND V(ADJ)\n"
    if opt.analysis == "dc_vin":
        return (
            f".temp {opt.temp_c}\n"
            f".dc Vin {opt.vin_start} {opt.vin_stop} {opt.vin_step}\n"
            ".meas DC vout_12 FIND V(OUT) AT 12\n"
        )
    if opt.analysis == "ac_psrr":
        return (
            f"Vin VIN 0 DC {{VIN}} AC 1\n"
            f".temp {opt.temp_c}\n"
            f".ac dec {opt.ac_points} {opt.ac_fstart} {opt.ac_fstop}\n"
            ".meas AC psrr_120 FIND mag(V(OUT)) AT 120\n"
        )
    if opt.analysis.startswith("tran"):
        return f".temp {opt.temp_c}\n.tran {opt.tran_tstep} {opt.tran_tstop} startup\n.meas TRAN vout_end FIND V(OUT) AT {opt.tran_tstop}\n"
    return f".temp {opt.temp_c}\n.op\n"


def build_netlist(
    p: CircuitParams | None = None,
    opt: NetlistOptions | None = None,
) -> str:
    if p is None:
        p = CircuitParams()
    if opt is None:
        opt = NetlistOptions()
    body = _circuit_body(p)
    extra = ""
    if opt.analysis in ("ac_psrr", "tran_line"):
        body = "\n".join(ln for ln in body.splitlines() if not ln.startswith("Vin "))
    if opt.analysis in ("dc_i", "tran_load"):
        # current-source load; drop Rload so Iout is the sweep/step
        body = "\n".join(ln for ln in body.splitlines() if not ln.startswith("Rload "))
    if opt.spice_engine == "ltspice":
        anal = _analysis_ltspice(opt)
    else:
        anal = _analysis_ngspice(opt)
    return (
        f"* {opt.title}\n"
        + _param_block(p)
        + "\n"
        + behav_subckt()
        + "\n"
        + body
        + "\n"
        + extra
        + anal
        + ".end\n"
    )


def write_netlist(
    dest: Path | str,
    p: CircuitParams | None = None,
    opt: NetlistOptions | None = None,
    *,
    mirror: Path | str | None = None,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = build_netlist(p, opt)
    dest.write_text(text, encoding="utf-8")
    if mirror is not None:
        m = Path(mirror)
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(text, encoding="utf-8")
    return dest


def write_behav_lib(dest: Path | str) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("* LT3010 behavioral (see netlist.behav_subckt)\n" + behav_subckt(), encoding="utf-8")
    return dest
