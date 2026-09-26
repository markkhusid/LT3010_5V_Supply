"""LTspice batch execution, .meas parse, .raw TRAN/AC parse."""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .timedomain import TranResult


@dataclass
class LtspiceAcSweep:
    f_hz: np.ndarray
    h: np.ndarray
    variables: dict[str, np.ndarray]
    raw_path: Path

    @property
    def mag_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(np.abs(self.h), 1e-30))

    @property
    def psrr_db(self) -> np.ndarray:
        return -self.mag_db


def find_ltspice(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(p)
    env = os.environ.get("LTSPICE_EXE")
    if env and Path(env).is_file():
        return Path(env)
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local) / "Programs" / "ADI" / "LTspice" / "LTspice.exe",
        Path(r"C:\Program Files\ADI\LTspice\LTspice.exe"),
        Path(r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe"),
        Path(shutil.which("LTspice") or ""),
    ]
    for c in candidates:
        if c and c.is_file():
            return c
    raise FileNotFoundError("LTspice.exe not found. Set LTSPICE_EXE.")


def run_ltspice_deck(
    cir_or_asc: Path | str,
    *,
    exe: Path | str | None = None,
    timeout_s: float = 600.0,
    quiet: bool = False,
) -> Path:
    deck = Path(cir_or_asc).resolve()
    if not deck.is_file():
        raise FileNotFoundError(deck)
    lt = find_ltspice(exe)
    log = deck.with_suffix(".log")
    if log.is_file():
        try:
            log.unlink()
        except OSError:
            pass
    cmd = [str(lt), "-b", "-Run", str(deck)]
    if not quiet:
        print(f"[LTspice] start  {deck.name}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(deck.parent), capture_output=True, text=True, timeout=timeout_s,
    )
    dt = time.perf_counter() - t0
    if not quiet:
        print(f"[LTspice] exit={proc.returncode}  elapsed={dt:.2f}s  {deck.name}", flush=True)
    if log.is_file():
        return log
    alt = deck.with_suffix(".LOG")
    if alt.is_file():
        return alt
    tail = (proc.stderr or proc.stdout or "")[-800:]
    raise FileNotFoundError(f"LTspice did not produce a log for {deck}\n{tail}")


def run_ltspice_decks(
    decks: list[Path | str],
    *,
    exe: Path | str | None = None,
    timeout_s: float = 600.0,
    max_workers: int | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    from .parallel import run_parallel

    jobs = [{"name": Path(d).resolve().stem, "deck": Path(d).resolve()} for d in decks]
    lt = find_ltspice(exe)

    def _one(job: dict) -> dict:
        log = run_ltspice_deck(job["deck"], exe=lt, timeout_s=timeout_s, quiet=True)
        size = log.stat().st_size if log.is_file() else 0
        from .parallel import _fmt_bytes

        return {
            "name": job["name"],
            "ok": True,
            "detail": f"{log.name}  {_fmt_bytes(size)}",
            "value": log,
        }

    values, report = run_parallel(
        _one, jobs, max_workers=max_workers,
        desc=f"LTspice batch  {len(jobs)} deck(s)", engine="LTspice",
        name_of=lambda j: str(j["name"]), verbose=verbose,
        cap=4, env_var="LTSPICE_BATCH_JOBS", leave_one_free=True,
    )
    out: dict[str, Path] = {}
    for job, val, oc in zip(jobs, values, report.outcomes):
        if val is not None:
            out[str(job["name"])] = val
        elif oc.error:
            raise RuntimeError(f"LTspice job {job['name']} failed: {oc.error}")
    return out


def parse_meas_log(log_path: Path | str, keys: Iterable[str] | None = None) -> dict[str, float]:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    keys_l = {k.lower() for k in keys} if keys is not None else None
    found: dict[str, float] = {}
    num = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

    def _want(name: str) -> bool:
        return keys_l is None or name in keys_l

    for m in re.finditer(rf"(?im)^\s*([A-Za-z0-9_]+)\s*:.*?=\s*{num}", text):
        name = m.group(1).lower()
        if _want(name) and name not in found:
            found[name] = float(m.group(2))
    return found


def _ltspice_raw_header(raw_path: Path) -> tuple[str, bytes]:
    data = raw_path.read_bytes()
    if data[1:2] == b"\x00" or data[:2] == b"\xff\xfe":
        marker = b"Binary:\x00\n\x00"
        idx = data.find(marker)
        if idx < 0:
            marker = "Binary:\n".encode("utf-16-le")
            idx = data.find(marker)
        if idx < 0:
            raise ValueError(f"No Binary: marker in {raw_path}")
        return data[:idx].decode("utf-16-le", errors="replace"), data[idx + len(marker) :]
    for nl in (b"Binary:\n", b"Binary:\r\n"):
        idx = data.find(nl)
        if idx >= 0:
            return data[:idx].decode("utf-8", errors="replace"), data[idx + len(nl) :]
    raise ValueError(f"No Binary: marker in {raw_path}")


def _ltspice_var_names(hdr: str, nvars: int) -> list[str]:
    names: list[str] = []
    in_vars = False
    for line in hdr.splitlines():
        ls = line.strip()
        if ls.lower().startswith("variables:"):
            in_vars = True
            continue
        if not in_vars:
            continue
        if not ls or ls.lower().startswith("binary"):
            break
        parts = re.split(r"\t+", ls)
        if len(parts) >= 2 and parts[0].strip().isdigit():
            names.append(parts[1].strip().lower())
            continue
        m = re.match(r"^\s*(\d+)\s*(\S+)", ls)
        if m:
            names.append(m.group(2).lower())
    return names[:nvars]


def parse_ltspice_raw_tran(
    raw_path: Path | str,
    *,
    vout_name: str = "v(p5_iso)",
    vin_name: str = "v(p9v_iso)",
) -> TranResult:
    raw_path = Path(raw_path)
    hdr, payload = _ltspice_raw_header(raw_path)
    nvars = npoints = 0
    for line in hdr.splitlines():
        ls = line.strip()
        if ls.lower().startswith("no. variables:"):
            nvars = int(ls.split(":")[-1])
        elif ls.lower().startswith("no. points:"):
            npoints = int(ls.split(":")[-1])
    names = _ltspice_var_names(hdr, nvars)
    rec = 8 + 4 * max(nvars - 1, 0)
    need = rec * npoints
    if len(payload) < need:
        floats = np.frombuffer(payload, dtype="<f8")
        arr = floats[: npoints * nvars].reshape(npoints, nvars)
        t = arr[:, 0]
        cols = {names[i] if i < len(names) else f"v{i}": arr[:, i] for i in range(nvars)}
    else:
        t = np.empty(npoints, dtype=float)
        others = np.empty((npoints, max(nvars - 1, 0)), dtype=np.float32)
        off = 0
        for i in range(npoints):
            t[i] = struct.unpack_from("<d", payload, off)[0]
            off += 8
            if nvars > 1:
                others[i, :] = np.frombuffer(payload, dtype="<f4", count=nvars - 1, offset=off)
                off += 4 * (nvars - 1)
        cols = {names[0] if names else "time": t}
        for i, name in enumerate(names[1:nvars]):
            cols[name] = others[:, i].astype(float)

    def _pick(*keys: str) -> np.ndarray | None:
        for k, v in cols.items():
            kl = k.lower()
            if any(want in kl for want in keys):
                return np.asarray(v, dtype=float)
        return None

    vout = _pick(vout_name.lower(), "p5_iso", "vout", "out")
    vin = _pick(vin_name.lower(), "p9v_iso", "vin")
    if vout is None:
        vout = np.zeros_like(t)
    return TranResult(t=np.asarray(t, dtype=float), vout=vout, vin=vin)


def meas_values(log_path: Path | str, name: str = "vout_op") -> np.ndarray:
    """Return the stepped `.meas` vector (WC/MC) as a 1-D array."""
    tables = parse_meas_tables(log_path)
    if name not in tables:
        # LTspice lowercases names
        key = next((k for k in tables if k.lower() == name.lower()), None)
        if key is None:
            return np.array([], dtype=float)
        name = key
    return tables[name]["value"].to_numpy(dtype=float)


def meas_frame(log_path: Path | str, name: str = "vout_op", col: str = "vout") -> pd.DataFrame:
    v = meas_values(log_path, name)
    if v.size == 0:
        return pd.DataFrame(columns=[col, "step"])
    return pd.DataFrame({col: v, "step": np.arange(1, v.size + 1)})


def parse_meas_tables(log_path: Path | str) -> dict[str, pd.DataFrame]:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"(?im)^Measurement:\s*", text)
    tables: dict[str, pd.DataFrame] = {}
    for part in parts[1:]:
        lines = part.splitlines()
        if not lines:
            continue
        name = lines[0].strip().lower()
        header_idx = None
        for i, line in enumerate(lines[1:8], start=1):
            if re.search(r"\bstep\b", line, re.I):
                header_idx = i
                break
        if header_idx is None:
            continue
        rows: list[tuple[int, float, float]] = []
        for line in lines[header_idx + 1 :]:
            if not line.strip():
                if rows:
                    break
                continue
            if re.match(r"(?i)^(Measurement:|Date:|Total elapsed|Memory)", line):
                break
            cols = re.split(r"[\t ]+", line.strip())
            if len(cols) < 2:
                continue
            try:
                step = int(float(cols[0]))
                val = float(cols[1].strip().strip(","))
            except ValueError:
                if rows:
                    break
                continue
            t = float(cols[2]) if len(cols) >= 3 else float("nan")
            rows.append((step, val, t))
        if rows:
            tables[name] = pd.DataFrame(rows, columns=["step", "value", "time_s"])
    return tables
