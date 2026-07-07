from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import parse_bias_value
from .ideal_mosfet import validate_reference_params
from .models import Curve, DeviceCurves


class OperatingPointFitError(ValueError):
    pass


@dataclass(frozen=True)
class OperatingPointFit:
    vth: float
    k: float
    rms_error: float
    points_used: int
    transfer_vds: float
    output_vgs: float
    fit_vgs: np.ndarray
    fit_id: np.ndarray
    fit_transfer_id: np.ndarray
    output_vds: np.ndarray
    output_id: np.ndarray

    @property
    def vov(self) -> float:
        return self.output_vgs - self.vth


def fit_operating_point_model(
    device: DeviceCurves,
    *,
    transfer_vds: float = 0.5,
    output_vgs: float = 0.1,
) -> OperatingPointFit:
    transfer_curve, actual_transfer_vds = _curve_with_bias(device.trans_id_vgs.curves, "VDS", transfer_vds)
    output_curve, actual_output_vgs = _curve_with_bias(device.iv_id_vds.curves, "VGS", output_vgs)

    vgs = np.asarray(transfer_curve.x, dtype=float)
    ids = np.asarray(transfer_curve.y, dtype=float)
    finite = np.isfinite(vgs) & np.isfinite(ids)
    if np.count_nonzero(finite) < 4:
        raise OperatingPointFitError("Need at least four finite transfer points for operating-point fitting.")
    vgs = vgs[finite]
    ids = ids[finite]

    vth, k, fit_mask = _fit_heaviside_saturation(vgs, ids, actual_transfer_vds, actual_output_vgs)
    fit_vgs = vgs[fit_mask]
    fit_id = ids[fit_mask]
    fit_transfer_id = _heaviside_saturation_id(fit_vgs, vth=vth, k=k)
    rms_error = _rms(fit_transfer_id - fit_id)

    output_x = np.asarray(output_curve.x, dtype=float)
    finite_output = output_x[np.isfinite(output_x)]
    vov = actual_output_vgs - vth
    if finite_output.size == 0 or vov <= 0.0:
        output_vds = np.array([], dtype=float)
        output_id = np.array([], dtype=float)
    else:
        max_vds = min(float(np.max(finite_output)), float(vov))
        output_vds = np.linspace(0.0, max_vds, 100)
        output_id = k * ((actual_output_vgs - vth) * output_vds - 0.5 * output_vds**2)

    return OperatingPointFit(
        vth=float(vth),
        k=float(k),
        rms_error=float(rms_error),
        points_used=int(np.count_nonzero(fit_mask)),
        transfer_vds=float(actual_transfer_vds),
        output_vgs=float(actual_output_vgs),
        fit_vgs=fit_vgs,
        fit_id=fit_id,
        fit_transfer_id=fit_transfer_id,
        output_vds=output_vds,
        output_id=output_id,
    )


def _fit_heaviside_saturation(
    vgs: np.ndarray,
    ids: np.ndarray,
    vds: float,
    operating_vgs: float,
) -> tuple[float, float, np.ndarray]:
    low = max(float(np.min(vgs) - 0.2), float((2.0 * operating_vgs) - vds))
    high = min(float(np.max(vgs) + 0.2), float(operating_vgs))
    if low >= high:
        raise OperatingPointFitError("Operating point cannot be placed inside the saturation fit region.")
    best: tuple[float, float, np.ndarray, float] | None = None
    for grid_size, span in ((900, high - low), (700, 0.08), (700, 0.015)):
        if best is None:
            candidates = np.linspace(low, high, grid_size)
        else:
            center = best[0]
            candidates = np.linspace(center - span, center + span, grid_size)
        for vth in candidates:
            if not ((vth < operating_vgs) and (operating_vgs < ((vds + vth) / 2.0))):
                continue
            fit_mask = _fit_region_mask(vgs, vds, vth)
            if np.count_nonzero(fit_mask) < 4:
                continue
            if np.count_nonzero(fit_mask & (vgs >= vth)) < 4:
                continue
            basis = 0.5 * np.square(vgs[fit_mask] - vth)
            basis = np.where(vgs[fit_mask] >= vth, basis, 0.0)
            denom = float(np.dot(basis, basis))
            if denom <= 0.0:
                continue
            k = float(np.dot(basis, ids[fit_mask]) / denom)
            if not np.isfinite(k) or k <= 0.0:
                continue
            predicted = k * basis
            sse = float(np.sum(np.square(predicted - ids[fit_mask])))
            if best is None or sse < best[3]:
                best = (float(vth), k, fit_mask, sse)
    if best is None:
        raise OperatingPointFitError("Could not find a valid operating-point saturation fit.")
    vth, k, fit_mask, _ = best
    validate_reference_params(vth, k)
    return vth, k, fit_mask


def _fit_region_mask(vgs: np.ndarray, vds: float, vth: float) -> np.ndarray:
    return np.isfinite(vgs) & (vgs < ((vds + vth) / 2.0))


def _heaviside_saturation_id(vgs: np.ndarray, *, vth: float, k: float) -> np.ndarray:
    vov = vgs - vth
    return 0.5 * k * np.square(vov) * np.where(vov >= 0.0, 1.0, 0.0)


def _curve_with_bias(curves: tuple[Curve, ...], bias_name: str, target: float) -> tuple[Curve, float]:
    candidates: list[tuple[float, Curve]] = []
    for curve in curves:
        value = parse_bias_value(curve.label, bias_name)
        if value is not None:
            candidates.append((value, curve))
    if not candidates:
        raise OperatingPointFitError(f"Could not find curves labeled with {bias_name}.")
    value, curve = min(candidates, key=lambda item: abs(item[0] - target))
    if abs(value - target) > 1e-6:
        raise OperatingPointFitError(f"Could not find {bias_name}={target:g} V curve.")
    return curve, value


def _rms(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))
