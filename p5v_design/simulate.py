"""Run ngspice and parse OP / DC / AC / TRAN results."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .netlist import NetlistOptions, write_netlist
from .params import CircuitParams
from .timedomain import TranResult


@dataclass
class AcResult:
    f_hz: np.ndarray
    vout: np.ndarray
    vin: np.ndarray

    @property
    def h(self) -> np.ndarray:
        vin = np.where(np.abs(self.vin) < 1e-30, 1e-30, self.vin)
        return self.vout / vin

    @property
    def mag_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(np.abs(self.h), 1e-30))

    @property
    def psrr_db(self) -> np.ndarray:
        return -self.mag_db


@dataclass
class OpResult:
    nodes: dict[str, float]


@dataclass
class DcSweepResult:
    x: np.ndarray
    vout: np.ndarray
    vadj: np.ndarray | None = None


def _console_ngspice(exe: Path) -> Path:
    if exe.name.lower() == "ngspice.exe":
        con = exe.with_name("ngspice_con.exe")
        if con.is_file():
            return con
    return exe


def _no_window_kwargs() -> dict:
    if os.name != "nt":
        return {}
    kw: dict = {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    kw["creationflags"] = flags
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    kw["startupinfo"] = si
    return kw


def find_ngspice(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return _console_ngspice(p)
        raise FileNotFoundError(f"ngspice not found at {p}")
    env = os.environ.get("NGSPICE_EXE")
    if env:
        p = Path(env)
        if p.is_file():
            return _console_ngspice(p)
    for name in ("ngspice_con", "ngspice"):
        which = shutil.which(name)
        if which:
            return _console_ngspice(Path(which))
    for base in (
        Path(os.environ.get("CONDA_PREFIX", "")),
        Path.home() / "miniforge3",
        Path.home() / "mambaforge",
        Path.home() / "anaconda3",
        Path.home() / "miniconda3",
    ):
        if not base or not str(base) or not base.exists():
            continue
        for cand in (
            base / "Library" / "bin" / "ngspice_con.exe",
            base / "Library" / "bin" / "ngspice.exe",
            base / "bin" / "ngspice",
        ):
            if cand.is_file():
                return _console_ngspice(cand)
    raise FileNotFoundError("ngspice not found. Install via conda-forge or set NGSPICE_EXE.")


def run_ngspice(
    cir_path: Path,
    workdir: Path | None = None,
    ngspice: Path | str | None = None,
    timeout_s: float = 180.0,
) -> subprocess.CompletedProcess:
    cir_path = Path(cir_path).resolve()
    workdir = Path(workdir).resolve() if workdir is not None else cir_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    exe = _console_ngspice(find_ngspice(ngspice))
    log_path = workdir / "ngspice.log"
    cmd = [str(exe), "-b", "-o", str(log_path), str(cir_path)]
    proc = subprocess.run(
        cmd, cwd=str(workdir), capture_output=True, text=True,
        timeout=timeout_s, **_no_window_kwargs(),
    )
    if proc.stdout or proc.stderr:
        with log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write("\n--- stdout ---\n")
            fh.write(proc.stdout or "")
            fh.write("\n--- stderr ---\n")
            fh.write(proc.stderr or "")
    return proc


def _find_wrdata(path: Path, glob: str) -> Path:
    candidates = [path]
    if path.suffix == "":
        candidates.append(Path(str(path) + ".data"))
    else:
        candidates.append(path.with_suffix(path.suffix + ".data"))
        candidates.append(path.with_suffix(".data"))
    for c in list(candidates):
        extra = c.parent / (c.name + ".data")
        if extra.exists():
            candidates.append(extra)
    hit = next((c for c in candidates if c.is_file()), None)
    if hit is None:
        hits = list(path.parent.glob(glob))
        if not hits:
            raise FileNotFoundError(f"wrdata not found near {path}")
        hit = hits[0]
    return hit


def parse_op_log(log_path: Path) -> OpResult:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    nodes: dict[str, float] = {}
    import re

    for m in re.finditer(r"(?im)^\s*v\(([^)]+)\)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text):
        nodes[m.group(1).lower()] = float(m.group(2))
        nodes[f"v({m.group(1).lower()})"] = float(m.group(2))
    return OpResult(nodes=nodes)


def parse_dc_sweep(path: Path) -> DcSweepResult:
    data_path = _find_wrdata(Path(path), "dc_*")
    raw = np.loadtxt(data_path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    # ngspice wrdata: x, re(v1), im(v1), re(v2), im(v2), ...
    x = raw[:, 0]
    vout = raw[:, 1]
    vadj = raw[:, 5] if raw.shape[1] >= 6 else (raw[:, 3] if raw.shape[1] >= 4 else None)
    return DcSweepResult(x=x, vout=vout, vadj=vadj)


def parse_ac_complex_wrdata(path: Path) -> AcResult:
    data_path = _find_wrdata(Path(path), "vout_complex*")
    raw = np.loadtxt(data_path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    f = raw[:, 0]
    vout = raw[:, 1] + 1j * raw[:, 3]
    if raw.shape[1] >= 8:
        vin = raw[:, 5] + 1j * raw[:, 7]
    else:
        vin = np.ones_like(f, dtype=complex)
    return AcResult(f_hz=f, vout=vout, vin=vin)


def parse_tran_wrdata(path: Path) -> TranResult:
    data_path = _find_wrdata(Path(path), "vout_tran*")
    raw = np.loadtxt(data_path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    t = raw[:, 0]
    vout = raw[:, 1]
    vin = raw[:, 3] if raw.shape[1] >= 4 else None
    return TranResult(t=t, vout=vout, vin=vin)


def _write_and_run(
    p: CircuitParams,
    opt: NetlistOptions,
    workdir: Path,
    root: Path | str | None,
    mirror: Path | str | None,
    ngspice=None,
    timeout_s: float = 180.0,
) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cir = workdir / "p5v.cir"
    write_netlist(cir, p, opt, mirror=mirror)
    proc = run_ngspice(cir, workdir=workdir, ngspice=ngspice, timeout_s=timeout_s)
    log = workdir / "ngspice.log"
    if proc.returncode not in (0, None) and not log.is_file():
        raise RuntimeError(f"ngspice failed rc={proc.returncode}  {proc.stderr[-400:]}")
    return workdir


def simulate_op(
    p: CircuitParams | None = None,
    *,
    workdir: Path | str,
    root: Path | str | None = None,
    mirror: Path | str | None = None,
    ngspice=None,
) -> OpResult:
    if p is None:
        p = CircuitParams()
    wd = _write_and_run(p, NetlistOptions(analysis="op", temp_c=p.op.temp_c), Path(workdir), root, mirror, ngspice)
    return parse_op_log(wd / "ngspice.log")


def simulate_dc_vin(
    p: CircuitParams | None = None,
    *,
    workdir: Path | str,
    root: Path | str | None = None,
    mirror: Path | str | None = None,
    ngspice=None,
) -> DcSweepResult:
    if p is None:
        p = CircuitParams()
    opt = NetlistOptions(analysis="dc_vin", temp_c=p.op.temp_c)
    wd = _write_and_run(p, opt, Path(workdir), root, mirror, ngspice)
    return parse_dc_sweep(wd / "dc_vin")


def simulate_dc_i(
    p: CircuitParams | None = None,
    *,
    workdir: Path | str,
    root: Path | str | None = None,
    mirror: Path | str | None = None,
    ngspice=None,
) -> DcSweepResult:
    if p is None:
        p = CircuitParams()
    opt = NetlistOptions(analysis="dc_i", temp_c=p.op.temp_c)
    wd = _write_and_run(p, opt, Path(workdir), root, mirror, ngspice)
    return parse_dc_sweep(wd / "dc_i")


def simulate_ac_psrr(
    p: CircuitParams | None = None,
    *,
    workdir: Path | str,
    root: Path | str | None = None,
    mirror: Path | str | None = None,
    ngspice=None,
) -> AcResult:
    if p is None:
        p = CircuitParams()
    opt = NetlistOptions(analysis="ac_psrr", temp_c=p.op.temp_c)
    wd = _write_and_run(p, opt, Path(workdir), root, mirror, ngspice)
    return parse_ac_complex_wrdata(wd / "vout_complex")


def simulate_tran(
    p: CircuitParams | None = None,
    *,
    kind: str = "startup",
    workdir: Path | str,
    root: Path | str | None = None,
    mirror: Path | str | None = None,
    ngspice=None,
) -> TranResult:
    if p is None:
        p = CircuitParams()
    key = {"startup": "tran_startup", "load": "tran_load", "line": "tran_line"}[kind]
    opt = NetlistOptions(analysis=key, temp_c=p.op.temp_c, load_i1=p.op.iload_op)
    wd = _write_and_run(p, opt, Path(workdir), root, mirror, ngspice, timeout_s=300.0)
    return parse_tran_wrdata(wd / "vout_tran")
