"""UVLO-style parallel SPICE runner with tqdm (or ASCII) progress.

Copied from HV_monitor and pointed at ``P5V_SPICE_WORKERS``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def cpu_count() -> int:
    return int(os.cpu_count() or 2)


def default_workers(
    n_jobs: int,
    *,
    cap: int = 8,
    env_var: str = "P5V_SPICE_WORKERS",
    leave_one_free: bool = True,
) -> int:
    n_jobs = max(1, int(n_jobs))
    env = os.environ.get(env_var, "").strip()
    if env.isdigit():
        return max(1, min(int(env), n_jobs))
    cpu = cpu_count()
    budget = max(1, cpu - 1) if leave_one_free else max(1, cpu)
    return max(1, min(n_jobs, cap, budget))


def resolve_workers(
    max_workers: int | None,
    n_jobs: int,
    *,
    cap: int = 8,
    env_var: str = "P5V_SPICE_WORKERS",
    leave_one_free: bool = False,
) -> int:
    if max_workers is not None:
        return max(1, min(int(max_workers), max(1, int(n_jobs))))
    return default_workers(
        n_jobs, cap=cap, env_var=env_var, leave_one_free=leave_one_free
    )


def tqdm_available() -> bool:
    try:
        import tqdm  # noqa: F401

        return True
    except ImportError:
        return False


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        return False
    if ip is None:
        return False
    return ip.__class__.__name__ == "ZMQInteractiveShell"


def _try_tqdm(total: int, desc: str, unit: str = "job"):
    if in_notebook() or not sys.stdout.isatty() or not tqdm_available():
        return None
    try:
        from tqdm import tqdm

        return tqdm(
            total=total, desc=desc, unit=unit, dynamic_ncols=True,
            smoothing=0.15, leave=True, mininterval=0.2, file=sys.stdout,
        )
    except Exception:
        return None


def _bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "-" * width
    frac = min(1.0, max(0.0, done / total))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def _fmt_s(t: float) -> str:
    if t < 60:
        return f"{t:.2f}s"
    m, s = divmod(t, 60.0)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60.0)
    return f"{int(h)}h{int(m):02d}m"


def _fmt_bytes(n: int) -> int | str:
    n = int(n)
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} kB"
    return f"{n / (1024 * 1024):.2f} MB"


def _short_name(name: str, width: int = 22) -> str:
    s = str(name)
    for pfx in ("P5V_ISO_", "P5V_", "p5v_"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    if len(s) > width:
        s = s[: width - 1] + "…"
    return s


def _eta(done: int, total: int, elapsed: float) -> str:
    if done <= 0 or elapsed <= 0:
        return "?"
    rate = done / elapsed
    remain = (total - done) / rate if rate > 0 else float("inf")
    if not (remain < 1e8):
        return "?"
    return _fmt_s(remain)


@dataclass
class JobOutcome:
    name: str
    ok: bool
    elapsed_s: float
    detail: str = ""
    error: str = ""
    value: Any = None
    index: int = 0


@dataclass
class ParallelReport:
    desc: str
    engine: str
    n_jobs: int
    n_workers: int
    n_ok: int
    n_fail: int
    wall_s: float
    cpu_s: float
    outcomes: list[JobOutcome] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return self.cpu_s / self.wall_s if self.wall_s > 1e-9 else float("nan")

    def summary_line(self) -> str:
        return (
            f"[{self.desc}] done  {self.n_ok} ok / {self.n_fail} fail / {self.n_jobs} jobs  "
            f"workers={self.n_workers}  wall={_fmt_s(self.wall_s)}  "
            f"cpu={_fmt_s(self.cpu_s)}  speedup={self.speedup:.2f}× vs serial"
        )


def _print(msg: str) -> None:
    print(msg, flush=True, file=sys.stdout)


def run_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int | None = None,
    desc: str = "SPICE",
    engine: str = "ngspice",
    name_of: Callable[[T], str] | None = None,
    detail_of: Callable[[R], str] | None = None,
    verbose: bool = True,
    cap: int = 8,
    env_var: str = "P5V_SPICE_WORKERS",
    leave_one_free: bool = False,
) -> tuple[list[R | None], ParallelReport]:
    items = list(items)
    n = len(items)
    if n == 0:
        rep = ParallelReport(desc, engine, 0, 1, 0, 0, 0.0, 0.0, [])
        return [], rep

    workers = resolve_workers(
        max_workers, n, cap=cap, env_var=env_var, leave_one_free=leave_one_free
    )
    if name_of is None:
        name_of = lambda it: str(getattr(it, "name", None) or (it.get("name") if isinstance(it, dict) else it))  # noqa: E731

    env_raw = os.environ.get(env_var, "")
    req = "auto" if max_workers is None else str(max_workers)
    names_all = [name_of(it) for it in items]
    _print("")
    _print("-" * 78)
    _print(f"  {desc}")
    _print(
        f"  {engine}  jobs={n}  workers={workers}  requested={req}  "
        f"cpu={cpu_count()}  env {env_var}={env_raw!r}"
    )
    preview = ", ".join(_short_name(s, 28) for s in names_all[:6])
    if n > 6:
        preview += f"  +{n - 6} more"
    _print(f"  queue: {preview}")
    _print("-" * 78)

    results: list[R | None] = [None] * n
    outcomes: list[JobOutcome] = [JobOutcome(name="?", ok=False, elapsed_s=0.0)] * n
    n_ok = n_fail = 0
    t_wall0 = time.perf_counter()
    bar = _try_tqdm(n, desc)
    log_starts = n <= 16
    log_each = bool(verbose) and n <= 16
    if n > 16:
        _print(f"  logging: heartbeat every ~2 s (N={n} — per-job lines suppressed)")
    running: dict[int, tuple[str, float]] = {}
    lock = threading.Lock()
    bar_lock = threading.Lock()
    last_hb_print = 0.0

    def _log(msg: str) -> None:
        if bar is not None:
            with bar_lock:
                bar.write(msg, file=sys.stdout)
        else:
            _print(msg)

    def _running_str() -> str:
        now = time.perf_counter()
        with lock:
            items_r = sorted(running.values(), key=lambda kv: kv[1])
        if not items_r:
            return "idle"
        parts = [f"{_short_name(nm, 16)} {_fmt_s(now - t0)}" for nm, t0 in items_r[:3]]
        extra = f" +{len(items_r) - 3}" if len(items_r) > 3 else ""
        return "run " + ", ".join(parts) + extra

    def _set_postfix() -> None:
        if bar is None:
            return
        with bar_lock:
            bar.set_postfix_str(f"ok={n_ok} fail={n_fail} {_running_str()}", refresh=True)

    def _run(idx: int, item: T) -> JobOutcome:
        name = name_of(item)
        t0 = time.perf_counter()
        with lock:
            running[idx] = (name, t0)
        if log_starts:
            wall = t0 - t_wall0
            _log(
                f"  →  start  {_short_name(name, 28):<28s}  "
                f"t+{_fmt_s(wall):>7s}  {_running_str()}"
            )
        try:
            raw = fn(item)
            elapsed = time.perf_counter() - t0
            if isinstance(raw, dict) and ("ok" in raw or "value" in raw or "detail" in raw):
                ok = bool(raw.get("ok", True))
                detail = str(raw.get("detail") or "")
                err = str(raw.get("error") or "")
                value = raw.get("value", raw)
                name = str(raw.get("name") or name)
                elapsed = float(raw.get("elapsed_s") or elapsed)
            else:
                ok = True
                detail = detail_of(raw) if detail_of is not None else ""
                err = ""
                value = raw
            return JobOutcome(
                name=name, ok=ok, elapsed_s=elapsed, detail=detail,
                error=err, value=value, index=idx,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            err = f"{type(exc).__name__}: {exc}"
            if verbose:
                err = err + "\n" + traceback.format_exc(limit=4)
            return JobOutcome(
                name=name, ok=False, elapsed_s=elapsed, detail="",
                error=err.strip(), value=None, index=idx,
            )
        finally:
            with lock:
                running.pop(idx, None)

    stop_hb = threading.Event()

    def _heartbeat() -> None:
        nonlocal last_hb_print
        while not stop_hb.wait(1.0):
            if bar is not None:
                _set_postfix()
                continue
            with lock:
                nrun = len(running)
            now = time.perf_counter()
            if nrun == 0 or now - last_hb_print < 2.0:
                continue
            last_hb_print = now
            wall = now - t_wall0
            finished = n_ok + n_fail
            _print(
                f"  …  {finished}/{n} done  ok={n_ok} fail={n_fail}  "
                f"wall {_fmt_s(wall)}  ETA {_eta(max(finished, 1), n, wall)}  "
                f"{_running_str()}"
            )

    hb = threading.Thread(target=_heartbeat, name="spice-status", daemon=True)
    hb.start()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_run, i, item): i for i, item in enumerate(items)}
            done = 0
            for fut in as_completed(futs):
                out = fut.result()
                i = out.index
                outcomes[i] = out
                results[i] = out.value if out.ok else None
                done += 1
                if out.ok:
                    n_ok += 1
                else:
                    n_fail += 1
                wall = time.perf_counter() - t_wall0
                mark = "OK  " if out.ok else "FAIL"
                metric = out.detail if out.ok else (
                    out.error.splitlines()[0] if out.error else "error"
                )
                line = (
                    f"  {'✓' if out.ok else '✗'}  {done:>3d}/{n}  {mark}  "
                    f"{_short_name(out.name, 28):<28s}  "
                    f"job {_fmt_s(out.elapsed_s):>7s}  wall {_fmt_s(wall):>7s}  "
                    f"ETA {_eta(done, n, wall):>7s}  {metric}"
                )
                if bar is not None:
                    with bar_lock:
                        bar.update(1)
                    _set_postfix()
                if log_each or not out.ok:
                    _log(line)
                elif bar is None and (done == n or done % max(1, n // 8) == 0):
                    _print(
                        f"  {_bar(done, n)} {done}/{n} ({100.0 * done / n:5.1f}%)  "
                        f"ok={n_ok} fail={n_fail} {_running_str()}"
                    )
    finally:
        stop_hb.set()
        hb.join(timeout=1.5)
        if bar is not None:
            _set_postfix()
            bar.close()

    wall = time.perf_counter() - t_wall0
    cpu_s = float(sum(o.elapsed_s for o in outcomes))
    report = ParallelReport(
        desc=desc, engine=engine, n_jobs=n, n_workers=workers,
        n_ok=n_ok, n_fail=n_fail, wall_s=wall, cpu_s=cpu_s, outcomes=outcomes,
    )
    _print("-" * 78)
    _print("  " + report.summary_line())
    if n_fail:
        _print(f"  FAILED jobs ({n_fail}):")
        for o in outcomes:
            if not o.ok:
                first = o.error.splitlines()[0] if o.error else "unknown"
                _print(f"    - {o.name}: {first}")
    _print("-" * 78)
    _print("")
    return results, report


def iter_named(items: Iterable[tuple[str, T]]) -> list[dict[str, Any]]:
    return [{"name": n, "payload": p} for n, p in items]
