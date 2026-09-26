# Notice

The circuit analyzed here is figure TA01, "5V Supply with Shutdown", on
page 1 of:

> LT3010/LT3010-5, 50 mA, 3 V to 80 V Low Dropout Micropower Linear Regulator
> Rev. J, Analog Devices (formerly Linear Technology)
> <https://www.analog.com/en/products/lt3010.html>

Published values used here: LT3010-5, VIN 5.4 V to 80 V, VOUT 5 V,
load 50 mA, 1 µF on the input, 1 µF on the output, SENSE tied to OUT.

Electrical limits (output tolerance, line and load regulation, dropout,
quiescent current, PSRR, current limit) are taken from the electrical
characteristics table in the same datasheet.

The datasheet drawing is copyright Analog Devices. This repository does
not include the PDF. The schematic in the notebooks is redrawn from the
published values.

The LT3010 SPICE macromodel is copyright Linear Technology / Analog
Devices and is not included.

The analysis code, notebooks, and plots are original work by Mark Khusid
and are covered by the MIT License. That license does not apply to the
datasheet or the macromodel.
