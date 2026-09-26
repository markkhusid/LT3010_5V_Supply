"""Circuit parameters, LT3010 page-3 limits, tolerances, temperature, derating.

HOW TO ADAPT
------------
1. Change fields on ``CircuitParams`` for R/C/load/Vin.
2. ``LdoParams`` is the LT3010E/MP electrical table (adjustable). Do not
   mix H-grade numbers into this object.
3. ``as_spice_params()`` names must stay in sync with ``netlist.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

ErrorMode = Literal["envelope", "stacked"]


@dataclass(frozen=True)
class ToleranceSpec:
    """Relative tolerance as a fraction (0.01 = 1%), not percent."""

    rel_tol: float = 0.01

    def __post_init__(self) -> None:
        if self.rel_tol < 0:
            raise ValueError("rel_tol must be >= 0")
        if self.rel_tol >= 1:
            raise ValueError("rel_tol must be < 1 (use fraction, e.g. 0.01 for 1%)")

    @property
    def percent(self) -> float:
        return 100.0 * self.rel_tol

    def bounds(self, nom: float) -> tuple[float, float]:
        return nom * (1.0 - self.rel_tol), nom * (1.0 + self.rel_tol)

    def scale(self, nom: float, u: float) -> float:
        lo, hi = self.bounds(nom)
        return lo + (hi - lo) * float(u)


@dataclass(frozen=True)
class LdoParams:
    """LT3010 / LT3010MP adjustable, datasheet Rev. J page 3.

    Over-temp ADJ min/max already include line, load, and temperature
    (Notes 2, 3, 10). Use ``error_mode='envelope'`` for WC/MC so those
    terms are not stacked on top of the box.
    """

    # LT3010-5 fixed 5 V (datasheet Rev. J page 1, TA01, and page 3).
    # The vadj_* fields hold the regulated OUTPUT box. SENSE is tied to
    # OUT, so the closed-loop gain is 1 and these volts appear at Vout.
    name: str = "LT3010-5"
    grade: str = "E"
    vadj_typ: float = 5.000
    vadj_min_25: float = 4.925
    vadj_max_25: float = 5.075
    vadj_min_ot: float = 4.850
    vadj_max_ot: float = 5.150
    # Line regulation at Vout, ΔVin = 5.5 V → 80 V, Iload = 1 mA
    line_typ_v: float = 3e-3
    line_max_v: float = 15e-3
    line_dv_in: float = 74.5  # 80 - 5.5
    line_vin_lo: float = 5.5
    line_vin_hi: float = 80.0
    # Load regulation at Vout, ΔIload = 1 mA → 50 mA, Vin = 6 V
    load_typ_v: float = 25e-3
    load_max_v_25: float = 50e-3
    load_max_v_ot: float = 90e-3
    load_di_a: float = 49e-3  # 50 mA - 1 mA
    load_i_lo: float = 1e-3
    load_i_hi: float = 50e-3
    # ADJ bias current, into the pin (Note 7)
    iadj_typ: float = 0.0  # fixed 5 V part has no ADJ pin
    iadj_max: float = 0.0
    # Dropout (Notes 4, 5): Iload breakpoints 1 / 10 / 50 mA
    vdo_i_a: tuple[float, ...] = (1e-3, 10e-3, 50e-3)
    vdo_typ_v: tuple[float, ...] = (100e-3, 200e-3, 300e-3)
    vdo_max_25_v: tuple[float, ...] = (150e-3, 260e-3, 370e-3)
    vdo_max_ot_v: tuple[float, ...] = (190e-3, 350e-3, 550e-3)
    # GND pin current (Notes 4, 6)
    ignd_i_a: tuple[float, ...] = (0.0, 1e-3, 10e-3, 50e-3)
    ignd_typ_a: tuple[float, ...] = (30e-6, 100e-6, 400e-6, 1.8e-3)
    ignd_max_a: tuple[float, ...] = (60e-6, 180e-6, 700e-6, 3.3e-3)
    # Noise / PSRR / limit
    noise_rms_v: float = 100e-6  # 10 Hz–100 kHz, Cout=10 µF, 50 mA
    psrr_typ_db: float = 68.0  # LT3010-5, 120 Hz, 0.5 Vpp, 50 mA
    psrr_min_db: float = 60.0
    ilim_typ_a: float = 140e-3
    ilim_min_ot_a: float = 60e-3
    vin_min_typ: float = 3.0
    vin_min_max: float = 4.0
    # Thermal, MS8E exposed pad
    tj_max_c: float = 125.0
    tmin_c: float = -40.0  # LT3010E / LT3010-5 row
    tmax_c: float = 125.0
    t0_c: float = 25.0
    rth_ja: float = 40.0  # °C/W, datasheet pin-config / 2500 mm² copper
    rth_jc: float = 16.0
    # SHDN
    shdn_on_max_v: float = 2.0
    shdn_off_min_v: float = 0.3
    # Cout / ESR (applications)
    cout_min_f: float = 1e-6
    esr_max_ohm: float = 3.0
    rbot_max_ohm: float = 250e3  # Iadj error guideline
    iadj_clamp_max_a: float = 5e-3  # ADJ pin current if OUT is forced high

    def vadj_bounds(self, *, over_temp: bool = False) -> tuple[float, float]:
        if over_temp:
            return self.vadj_min_ot, self.vadj_max_ot
        return self.vadj_min_25, self.vadj_max_25

    def iadj_bounds(self, temp_c: float | None = None) -> tuple[float, float]:
        """Current into ADJ. Lower bound 0 (current cannot reverse)."""
        hi = self.iadj_max_at_temp(self.t0_c if temp_c is None else temp_c)
        return 0.0, hi

    def iadj_max_at_temp(self, temp_c: float) -> float:
        """G15: ~50 nA at 25 °C, rising sharply above ~100 °C.

        Piecewise-linear through (25 °C, 50 nA), (100 °C, 50 nA),
        (150 °C, 180 nA), with a mild increase below 0 °C (~60 nA at −50 °C).
        Floor at the 25 °C max (100 nA) for WC so we never under-bound.
        """
        t = float(temp_c)
        if t <= 25.0:
            typ = 50e-9 + (60e-9 - 50e-9) * (25.0 - t) / 75.0
        elif t <= 100.0:
            typ = 50e-9
        else:
            typ = 50e-9 + (180e-9 - 50e-9) * (t - 100.0) / 50.0
        scale = typ / 50e-9
        return self.iadj_max * max(scale, 1.0)

    def load_max_v(self, *, over_temp: bool = False) -> float:
        return self.load_max_v_ot if over_temp else self.load_max_v_25

    def summary_rows(self) -> list[tuple[str, str]]:
        return [
            ("Device", self.name),
            ("Grade", f"{self.grade}  ({self.tmin_c:.0f}…{self.tmax_c:.0f} °C)"),
            ("Vadj typ", f"{self.vadj_typ:.3f} V"),
            ("Vadj 25 °C", f"{self.vadj_min_25:.3f} … {self.vadj_max_25:.3f} V"),
            ("Vadj over temp", f"{self.vadj_min_ot:.3f} … {self.vadj_max_ot:.3f} V"),
            ("Iadj typ / max", f"{self.iadj_typ * 1e9:.0f} / {self.iadj_max * 1e9:.0f} nA into ADJ"),
            ("Line (ADJ)", f"{self.line_typ_v * 1e3:.0f} typ, {self.line_max_v * 1e3:.0f} max mV / {self.line_dv_in:.0f} V"),
            ("Load (ADJ)", f"{self.load_typ_v * 1e3:.0f} typ, {self.load_max_v_25 * 1e3:.0f} max mV / 49 mA"),
            ("PSRR 120 Hz", f"{self.psrr_typ_db:.0f} typ, {self.psrr_min_db:.0f} min dB"),
            ("Noise", f"{self.noise_rms_v * 1e6:.0f} µVrms (10 Hz–100 kHz)"),
            ("Ilim typ / min", f"{self.ilim_typ_a * 1e3:.0f} / {self.ilim_min_ot_a * 1e3:.0f} mA"),
            ("θJA / θJC", f"{self.rth_ja:.0f} / {self.rth_jc:.0f} °C/W"),
        ]


@dataclass(frozen=True)
class OperatingPoint:
    # TA01 prints VIN as "5.4V TO 80V" and the load as 5 V / 50 mA.
    # 12 V is the single-point bias used for calculations; it sits inside
    # that printed range (the figure does not label one input voltage).
    vin_nom: float = 12.0
    vin_min: float = 5.4
    vin_max: float = 80.0
    vout_target: float = 5.0
    iload_op: float = 50e-3
    iload_min: float = 1e-3
    iload_max: float = 50e-3
    temp_c: float = 25.0
    r_load: float = 100.0  # 5 V / 50 mA

    @property
    def iload_from_r(self) -> float:
        return self.vout_target / max(self.r_load, 1e-12)


@dataclass(frozen=True)
class TempcoParams:
    tcr_ppm_per_c: float = 100.0  # 1 % thick-film class
    tcc_ppm_per_c: float = 0.0  # X7R handled as % tables, not ppm
    temp_min_c: float = -40.0
    temp_max_c: float = 125.0
    temp_step_c: float = 20.0
    t_ref_c: float = 25.0

    def grid(self) -> list[float]:
        vals: list[float] = []
        t = self.temp_min_c
        while t <= self.temp_max_c + 1e-9:
            vals.append(round(t, 6))
            t += self.temp_step_c
        for extra in (self.t_ref_c, -55.0, -40.0, 0.0, 25.0, 85.0, 125.0):
            if self.temp_min_c - 1e-9 <= extra <= self.temp_max_c + 1e-9:
                vals.append(extra)
        return sorted(set(vals))

    def scale_passive(self, nom: float, temp_c: float, ppm: float) -> float:
        return nom * (1.0 + ppm * 1e-6 * (temp_c - self.t_ref_c))


@dataclass(frozen=True)
class DeratingParams:
    resistor_package: str = "0805"
    resistor_power_rated_w: float = 0.125
    resistor_full_power_to_c: float = 70.0
    resistor_zero_power_c: float = 155.0
    resistor_power_use_fraction: float = 0.5
    capacitor_v_rated_out: float = 10.0  # assumption; TA01 does not print a voltage rating
    capacitor_v_rated_in: float = 100.0
    capacitor_v_use_fraction: float = 0.5
    v_derate_frac: float = 0.80
    copper_area_mm2: float = 2500.0

    def derated_power_w(self, temp_c: float) -> float:
        p = self.resistor_power_rated_w
        t1 = self.resistor_full_power_to_c
        t0 = self.resistor_zero_power_c
        if temp_c <= t1:
            return p
        if temp_c >= t0:
            return 0.0
        return p * (t0 - temp_c) / (t0 - t1)


@dataclass(frozen=True)
class CircuitParams:
    """LT3010-5 datasheet TA01 passives + IC + operate point.

    The fixed 5 V part ties SENSE to OUT. There is no external divider:
    r1 = 0 is that short, r2 is an open (so it does not load the output),
    and cff is absent. Vout equals the page-3 regulated-output box.
    """

    r1: float = 0.0  # SENSE shorted to OUT
    r2: float = 1.0e12  # open; no bottom resistor on the fixed part
    cff: float = 0.0
    cout: float = 1.0e-6
    cin: float = 1.0e-6
    esr_out: float = 10e-3  # ceramic assumption; not printed on TA01
    esr_in: float = 10e-3
    r_load: float = 100.0

    r_tol: ToleranceSpec = field(default_factory=lambda: ToleranceSpec(0.01))
    c_tol: ToleranceSpec = field(default_factory=lambda: ToleranceSpec(0.10))
    esr_tol: ToleranceSpec = field(default_factory=lambda: ToleranceSpec(0.50))

    ldo: LdoParams = field(default_factory=LdoParams)
    op: OperatingPoint = field(default_factory=OperatingPoint)
    tempco: TempcoParams = field(default_factory=TempcoParams)
    derating: DeratingParams = field(default_factory=DeratingParams)
    error_mode: ErrorMode = "envelope"

    # Injected IC state (nominal sim uses typ / 0 extra)
    vadj: float = 5.0
    iadj: float = 0.0
    vadj_err: float = 0.0  # delta from typ, for SPICE overlay
    iadj_extra: float = 0.0

    def with_values(self, **kwargs: Any) -> "CircuitParams":
        return replace(self, **kwargs)

    @property
    def gain(self) -> float:
        """Closed-loop Vout / Vadj, ignoring Iadj."""
        return 1.0 + self.r1 / max(self.r2, 1e-30)

    @property
    def i_div(self) -> float:
        return self.ldo.vadj_typ / max(self.r2, 1e-30)

    def passive_dict(self) -> dict[str, float]:
        return {
            "R1": self.r1,
            "R2": self.r2,
            "Cff": self.cff,
            "Cout": self.cout,
            "Cin": self.cin,
            "ESRout": self.esr_out,
            "ESRin": self.esr_in,
            "Rload": self.r_load,
        }

    def apply_passives(self, d: dict[str, float]) -> "CircuitParams":
        kw: dict[str, Any] = {}
        mapping = {
            "R1": "r1",
            "R2": "r2",
            "Cff": "cff",
            "Cout": "cout",
            "Cin": "cin",
            "ESRout": "esr_out",
            "ESRin": "esr_in",
            "Rload": "r_load",
        }
        for k, v in d.items():
            if k in mapping:
                kw[mapping[k]] = float(v)
        return replace(self, **kw)

    def tol_for(self, name: str) -> ToleranceSpec:
        if name in ("R1", "R2", "Rload"):
            return self.r_tol
        if name in ("Cff", "Cout", "Cin"):
            return self.c_tol
        if name in ("ESRout", "ESRin"):
            return self.esr_tol
        raise KeyError(f"No tolerance mapping for {name!r}")

    def scaled_at_temp(self, temp_c: float, tcr_sign: float = 1.0) -> "CircuitParams":
        tc = self.tempco
        ppm = tcr_sign * tc.tcr_ppm_per_c
        return replace(
            self,
            r1=tc.scale_passive(self.r1, temp_c, ppm),
            r2=tc.scale_passive(self.r2, temp_c, ppm),
            r_load=tc.scale_passive(self.r_load, temp_c, ppm),
            op=replace(self.op, temp_c=float(temp_c)),
        )

    def as_spice_params(self) -> dict[str, float]:
        vin = self.op.vin_nom
        line_k = self.ldo.line_typ_v / self.ldo.line_dv_in
        return {
            "R1": self.r1,
            "R2": self.r2,
            "CFF": self.cff,
            "COUT": self.cout,
            "CIN": self.cin,
            "ESROUT": self.esr_out,
            "ESRIN": self.esr_in,
            "RLOAD": self.r_load,
            "VIN": vin,
            "ILOAD": self.op.iload_op,
            "VADJ": self.vadj,
            "IADJ": self.iadj,
            "VADJERR": self.vadj_err,
            "IADJEXTRA": self.iadj_extra,
            "VINNOM": self.op.vin_nom,
            "LINEK": line_k,
            "PSRRTYP": self.ldo.psrr_typ_db,
            "ILIM": self.ldo.ilim_typ_a,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_circuit() -> CircuitParams:
    return CircuitParams()


def format_eng(x: float, unit: str = "") -> str:
    ax = abs(x)
    if ax == 0:
        return f"0 {unit}".strip()
    exp = int(math.floor(math.log10(ax) / 3) * 3)
    exp = max(min(exp, 12), -15)
    scaled = x / 10**exp
    prefixes = {
        12: "T",
        9: "G",
        6: "M",
        3: "k",
        0: "",
        -3: "m",
        -6: "µ",
        -9: "n",
        -12: "p",
        -15: "f",
    }
    return f"{scaled:.4g} {prefixes.get(exp, f'e{exp}')}{unit}".strip()
