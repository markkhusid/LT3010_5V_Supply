"""End-to-end smoke without Jupyter."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

from p5v_design.params import CircuitParams
from p5v_design import analysis, schematic, stability
from p5v_design.worst_case import analytical_worst_case, rss_bounds
from p5v_design.monte_carlo import sample_metrics
from p5v_design.temperature import sweep_analytical
from p5v_design.sensitivity import full_sensitivity
from p5v_design.thermal import power_table
from p5v_design.supply_budget import rail_budget
from p5v_design.netlist import write_netlist, write_behav_lib, NetlistOptions


def main() -> None:
    p = CircuitParams()
    m = analysis.summary_metrics(p)
    print("Vout", m["vout"], "gain", m["gain"], "bias", m["setpoint_bias"])
    print("headroom typ/wc", m["headroom_typ"], m["headroom_wc"])
    print("Pdiss", m["p_diss"], "eta", m["efficiency"])
    print(stability.stability_table(p))
    FIG = ROOT / "results" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    schematic.draw_schematic(FIG / "p5v_schematic.png", p)
    wc = analytical_worst_case(p)
    rss = rss_bounds(p)
    print("WC Vout", wc.minimum["vout"], wc.maximum["vout"])
    print("RSS Vout", rss["low"], rss["high"])
    mc = sample_metrics(p, n=64, seed=1)
    print(mc.summary.loc["vout"])
    sw = sweep_analytical(p)
    print(sw.frame[["temp_c", "vout"]].head())
    sens = full_sensitivity(p)
    print(sens.oat.head())
    print(power_table(p)[["ref", "P_W", "status"]])
    print(rail_budget(p))
    write_behav_lib(ROOT / "models" / "lt3010" / "lt3010_behav.lib")
    write_netlist(
        ROOT / "results" / "ngspice_nom" / "op" / "p5v.cir",
        p,
        NetlistOptions(analysis="op"),
        mirror=ROOT / "netlists" / "spice_nom" / "p5v_op.cir",
    )
    try:
        from p5v_design.simulate import simulate_op, find_ngspice

        print("ngspice", find_ngspice())
        op = simulate_op(p, workdir=ROOT / "results" / "ngspice_nom" / "op", root=ROOT)
        print("ngspice OP", op.nodes)
        vsp = op.nodes.get("out")
        if vsp is not None:
            print("OP error mV", 1e3 * (vsp - m["vout"]))
        from p5v_design.worst_case import spice_wc_corners
        from p5v_design.monte_carlo import spice_monte_carlo

        print("--- ngspice WC 4 corners ---")
        print(spice_wc_corners(p, workdir=ROOT / "results" / "ngspice_wc"))
        print("--- ngspice MC n=4 ---")
        sp = spice_monte_carlo(p, n=4, seed=1, workdir=ROOT / "results" / "ngspice_mc_smoke")
        print(sp.summary)
    except FileNotFoundError as e:
        print("ngspice skip:", e)
    except Exception as e:
        print("ngspice run failed:", type(e).__name__, e)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
