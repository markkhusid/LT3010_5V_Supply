# LT3010-5 — 5 V supply with shutdown

Analysis of the typical application on page 1 of the LT3010/LT3010-5
datasheet (Rev. J), figure TA01, "5V Supply with Shutdown".

Datasheet: <https://www.analog.com/media/en/technical-documentation/data-sheets/30105j.pdf>

## Circuit on the figure

| Item | Value |
|------|--------|
| Part | LT3010-5 (fixed 5 V, SENSE tied to OUT) |
| VIN | 5.4 V to 80 V |
| VOUT | 5 V |
| Load | 50 mA (Rload = 100 Ω) |
| Input capacitor | 1 µF |
| Output capacitor | 1 µF |
| SHDN | above 2.0 V on, below 0.3 V off |

The figure does not label a single input voltage. Single-point calculations
use 12 V, which is inside the printed 5.4 V to 80 V range. Ceramic ESR of
10 mΩ and the capacitor voltage ratings used in the derating notebook are
assumptions of this study. They are not printed on the figure.

Page 3 of Rev. J supplies the min/typ/max numbers (output voltage, line and
load regulation, dropout, GND pin current, PSRR, current limit).

## What is not included

The datasheet PDF and the Analog Devices / Linear Technology LT3010 SPICE
macromodel are not in this repository. ngspice uses a behavioral model in
`models/lt3010/lt3010_behav.lib`. LTspice runs that need `LT3010.lib` are
left optional.

## Layout

```text
p5v_design/     Python package
models/lt3010/  behavioral subcircuit
notebooks/      01–10, executed
results/figures plots
```

## Setup

```text
pip install -r requirements.txt
# ngspice on PATH (or NGSPICE_EXE)
```

## Notebooks

| Notebook | Content |
|----------|---------|
| 01 | Analytical setpoint and page-3 error budget |
| 02 | ngspice behavioral model vs analytical |
| 03 | Worst case |
| 04 | Uniform Monte Carlo |
| 05 | Analytical and ngspice corners (LTspice optional) |
| 06 | Temperature, −40 °C to +125 °C |
| 07 | Sensitivity |
| 08 | Derating and thermal |
| 09 | Input current and efficiency |
| 10 | Startup, load step, line step |

## License

MIT, for this analysis. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
Analog Devices retains copyright in the LT3010 datasheet and macromodel.
