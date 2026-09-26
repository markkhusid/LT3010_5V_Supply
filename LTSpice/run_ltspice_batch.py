"""Write LTspice decks for the LT3010-5 datasheet example and optionally run them.

The vendor macromodel is not part of this repository. Point LT3010_LIB at a
local copy of Analog Devices' LT3010.lib before running.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p5v_design.params import CircuitParams
from p5v_design.temperature import temp_tag

LT = ROOT / "LTSpice"
NOM = LT / "Nominal"
MC = LT / "MC"
WC = LT / "WC"
TEMP = LT / "Temp"


def _lib_src() -> Path | None:
    env = os.environ.get("LT3010_LIB", "")
    candidates = [
        Path(env) if env else None,
        ROOT / "models" / "lt3010" / "LT3010.lib",
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None


def _copy_lib(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    src = _lib_src()
    if src is None:
        raise FileNotFoundError(
            "LT3010.lib not found. Set LT3010_LIB to a local copy. "
            "The vendor model is not redistributed."
        )
    shutil.copy2(src, dest / "LT3010.lib")


def _deck(p: CircuitParams, verr: str) -> str:
    """LT3010-5, SENSE tied to the IC output, load on VOUT.

    verr is an LTspice expression (volts) added in series with the
    macromodel output. The macromodel is a typical device. The same
    page-3 output-tolerance term used by the analytical and ngspice
    engines is applied here so worst-case and Monte Carlo compare
    the same box.
    """
    return f"""* LT3010-5 datasheet TA01
V1 VIN 0 {p.op.vin_nom:g}
XU1 NIC SENSE NC1 0 VIN NC2 NC3 VIN LT3010-5
Rshort SENSE NIC 1u
Verr NIC VOUT {verr}
Cin VIN 0 {p.cin:g} Rser={p.esr_in:g}
Cout VOUT 0 {p.cout:g} Rser={p.esr_out:g}
Rload VOUT 0 {p.r_load:g}
.lib LT3010.lib
"""


def write_all(p: CircuitParams | None = None, mc_n: int = 200) -> None:
    if p is None:
        p = CircuitParams()
    _copy_lib(NOM)
    _copy_lib(WC)
    _copy_lib(MC)
    (NOM / "P5V_ISO_Regulator_op.cir").write_text(
        _deck(p, "0") + ".tran 1u\n.meas TRAN vout_op FIND V(VOUT) AT 1u\n.end\n",
        encoding="utf-8",
    )
    (NOM / "P5V_ISO_Regulator_tran.cir").write_text(
        _deck(p, "0") + ".tran 0 5m 0 1u startup\n.end\n",
        encoding="utf-8",
    )
    # 25 C page-3 box: 4.925 V to 5.075 V around the 5.000 V typical.
    half = 0.075
    (WC / "P5V_ISO_Regulator_WC.cir").write_text(
        _deck(p, "{s*0.075}")
        + ".param s=1\n"
        + ".tran 1u\n"
        + ".step param s list -1 1\n"
        + ".meas TRAN vout_op FIND V(VOUT) AT 1u\n.end\n",
        encoding="utf-8",
    )
    (MC / "P5V_ISO_Regulator_MC.cir").write_text(
        _deck(p, "{(mc(1,1)-1)*0.075}")
        + f".tran 1u\n.step param run 1 {int(mc_n)} 1\n"
        + ".meas TRAN vout_op FIND V(VOUT) AT 1u\n.end\n",
        encoding="utf-8",
    )
    del half


def write_temp_decks(p: CircuitParams, temps: list[float]) -> None:
    """Operating-point and worst-case decks at each temperature."""
    for t in temps:
        dest = TEMP / temp_tag(t)
        _copy_lib(dest)
        (dest / "P5V_ISO_Regulator_op.cir").write_text(
            _deck(p, "0") + f".temp {t:g}\n.tran 1u\n.meas TRAN vout_op FIND V(VOUT) AT 1u\n.end\n",
            encoding="utf-8",
        )
        (dest / "P5V_ISO_Regulator_WC.cir").write_text(
            _deck(p, "{s*0.075}")
            + f".param s=1\n.temp {t:g}\n.tran 1u\n"
            + ".step param s list -1 1\n"
            + ".meas TRAN vout_op FIND V(VOUT) AT 1u\n.end\n",
            encoding="utf-8",
        )
