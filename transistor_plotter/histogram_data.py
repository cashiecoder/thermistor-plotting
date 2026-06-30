from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import parse_bias_value
from .models import Curve, DeviceCurves

HISTOGRAM_VGS = 0.1
HISTOGRAM_VDS = 0.5
HISTOGRAM_TRANSFER_VDS = 0.9
HISTOGRAM_DIODE_VDS = 0.5


@dataclass(frozen=True)
class HistogramSample:
    gate_ig: float | None
    transfer_id: float | None
    output_id: float | None


def extract_histogram_sample(device: DeviceCurves) -> HistogramSample:
    diode_curve = _curve_for_bias(device.diode_ig_vgs.curves, "VDS", HISTOGRAM_DIODE_VDS)
    transfer_curve = _curve_for_bias(device.trans_id_vgs.curves, "VDS", HISTOGRAM_TRANSFER_VDS)
    output_curve = _curve_for_bias(device.iv_id_vds.curves, "VGS", HISTOGRAM_VGS)

    return HistogramSample(
        gate_ig=_interpolated_value(diode_curve, HISTOGRAM_VGS),
        transfer_id=_interpolated_value(transfer_curve, HISTOGRAM_VGS),
        output_id=_interpolated_value(output_curve, HISTOGRAM_VDS),
    )


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
