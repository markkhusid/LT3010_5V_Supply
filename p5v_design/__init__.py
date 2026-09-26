"""LT3010-5 datasheet example: 5 V at 50 mA, 1 µF in and out, VIN 5.4 V to 80 V."""

from .params import (
    CircuitParams,
    DeratingParams,
    LdoParams,
    OperatingPoint,
    TempcoParams,
    ToleranceSpec,
    format_eng,
)

__all__ = [
    "CircuitParams",
    "DeratingParams",
    "LdoParams",
    "OperatingPoint",
    "TempcoParams",
    "ToleranceSpec",
    "format_eng",
]

__version__ = "0.1.0"
