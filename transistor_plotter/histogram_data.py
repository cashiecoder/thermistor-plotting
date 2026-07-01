from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import parse_bias_value
from .models import Curve, DeviceCurves

HISTOGRAM_VGS = 0.1
HISTOGRAM_VDS = 0.5
HISTOGRAM_DIODE_VDS_VALUES = (0.0, 0.5)
HISTOGRAM_TRANSFER_VDS_VALUES = (0.1, 0.5, 0.9)
HISTOGRAM_IV_VGS_VALUES = (-0.3, 0.1, 0.5)


@dataclass(frozen=True)
class HistogramSpec:
    key: str
    title: str
    xlabel: str
    color: str


HISTOGRAM_SPECS = (
    HistogramSpec(
        "diode_ig_vds_0p0",
        f"Ig DIODE @ Vgs={HISTOGRAM_VGS:+.1f} V, Vds=+0.0 V",
        "Ig (uA/mm)",
        "#1f77b4",
    ),
    HistogramSpec(
        "diode_ig_vds_0p5",
        f"Ig DIODE @ Vgs={HISTOGRAM_VGS:+.1f} V, Vds=+0.5 V",
        "Ig (uA/mm)",
        "#ff7f0e",
    ),
    HistogramSpec(
        "trans_id_vds_0p1",
        f"Id TRANS @ Vgs={HISTOGRAM_VGS:+.1f} V, Vds=+0.1 V",
        "Id (mA/mm)",
        "#2ca02c",
    ),
    HistogramSpec(
        "trans_id_vds_0p5",
        f"Id TRANS @ Vgs={HISTOGRAM_VGS:+.1f} V, Vds=+0.5 V",
        "Id (mA/mm)",
        "#d62728",
    ),
    HistogramSpec(
        "trans_id_vds_0p9",
        f"Id TRANS @ Vgs={HISTOGRAM_VGS:+.1f} V, Vds=+0.9 V",
        "Id (mA/mm)",
        "#9467bd",
    ),
    HistogramSpec(
        "iv_id_vgs_n0p3",
        f"Id IV @ Vds={HISTOGRAM_VDS:+.1f} V, Vgs=-0.3 V",
        "Id (mA/mm)",
        "#8c564b",
    ),
    HistogramSpec(
        "iv_id_vgs_0p1",
        f"Id IV @ Vds={HISTOGRAM_VDS:+.1f} V, Vgs=+0.1 V",
        "Id (mA/mm)",
        "#e377c2",
    ),
    HistogramSpec(
        "iv_id_vgs_0p5",
        f"Id IV @ Vds={HISTOGRAM_VDS:+.1f} V, Vgs=+0.5 V",
        "Id (mA/mm)",
        "#17becf",
    ),
)


@dataclass(frozen=True)
class HistogramSample:
    values: dict[str, float | None]


def extract_histogram_sample(device: DeviceCurves) -> HistogramSample:
    values: dict[str, float | None] = {}

    for vds, key in zip(HISTOGRAM_DIODE_VDS_VALUES, ("diode_ig_vds_0p0", "diode_ig_vds_0p5"), strict=True):
        diode_curve = _curve_for_bias(device.diode_ig_vgs.curves, "VDS", vds)
        values[key] = _interpolated_value(diode_curve, HISTOGRAM_VGS)

    for vds, key in zip(
        HISTOGRAM_TRANSFER_VDS_VALUES,
        ("trans_id_vds_0p1", "trans_id_vds_0p5", "trans_id_vds_0p9"),
        strict=True,
    ):
        transfer_curve = _curve_for_bias(device.trans_id_vgs.curves, "VDS", vds)
        values[key] = _interpolated_value(transfer_curve, HISTOGRAM_VGS)

    for vgs, key in zip(
        HISTOGRAM_IV_VGS_VALUES,
        ("iv_id_vgs_n0p3", "iv_id_vgs_0p1", "iv_id_vgs_0p5"),
        strict=True,
    ):
        output_curve = _curve_for_bias(device.iv_id_vds.curves, "VGS", vgs)
        values[key] = _interpolated_value(output_curve, HISTOGRAM_VDS)

    return HistogramSample(values=values)


def _curve_for_bias(curves: tuple[Curve, ...], bias_name: str, target: float) -> Curve | None:
    best_curve: Curve | None = None
    best_distance = float("inf")
    for curve in curves:
        value = parse_bias_value(curve.label, bias_name)
        if value is None:
            continue
        distance = abs(value - target)
        if distance < best_distance:
            best_distance = distance
            best_curve = curve
    return best_curve


def _interpolated_value(curve: Curve | None, target_x: float) -> float | None:
    if curve is None or len(curve.x) == 0 or len(curve.y) == 0:
        return None
    order = np.argsort(curve.x)
    x = np.asarray(curve.x, dtype=float)[order]
    y = np.asarray(curve.y, dtype=float)[order]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) == 0 or target_x < x[0] or target_x > x[-1]:
        return None
    return float(np.interp(target_x, x, y))
