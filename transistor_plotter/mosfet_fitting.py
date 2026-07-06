from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import parse_bias_value
from .ideal_mosfet import validate_reference_params
from .models import DeviceCurves


class FitError(ValueError):
    pass


@dataclass(frozen=True)
class MosfetFitResult:
    method: str
    vth: float
    k: float
    points_used: int
    rms_error: float


@dataclass(frozen=True)
class MosfetReferenceFit:
    saturation: MosfetFitResult
    triode: MosfetFitResult | None = None


DEFAULT_ID_THRESHOLD = 10.0
DEFAULT_ID_MAX_THRESHOLD = 1000.0


def fit_reference_from_transfer_regions(
    device: DeviceCurves,
    id_threshold: float = DEFAULT_ID_THRESHOLD,
) -> MosfetReferenceFit:
    return MosfetReferenceFit(
        triode=fit_triode_eq_5_16(device, id_threshold=id_threshold),
        saturation=fit_saturation_largest_vds(device, id_threshold=id_threshold),
    )


def fit_saturation_largest_vds(device: DeviceCurves, id_threshold: float = DEFAULT_ID_THRESHOLD) -> MosfetFitResult:
    curve, vds = _largest_vds_transfer_curve(device)
    vgs = np.asarray(curve.x, dtype=float)
    ids = np.asarray(curve.y, dtype=float)
    finite = _id_window_mask(vgs, ids, id_threshold, None)
    if np.count_nonzero(finite) < 2:
        raise FitError(f"Need at least two finite Id points above {id_threshold:g} mA/mm on the largest-VDS transfer curve.")

    k, vth = _fit_saturation_once(vgs[finite], ids[finite])
    sat_mask = _saturation_mask(vgs, ids, vds, vth, id_threshold, None)
    if np.count_nonzero(sat_mask) < 2:
        raise FitError("Could not identify enough above-threshold saturation-region points on the largest-VDS curve.")

    for _ in range(6):
        k, vth = _fit_saturation_once(vgs[sat_mask], ids[sat_mask])
        next_mask = _saturation_mask(vgs, ids, vds, vth, id_threshold, None)
        if np.count_nonzero(next_mask) < 2:
            break
        if np.array_equal(next_mask, sat_mask):
            sat_mask = next_mask
            break
        sat_mask = next_mask

    k, vth = _fit_saturation_once(vgs[sat_mask], ids[sat_mask])
    predicted = _saturation_id(vgs[sat_mask], vth=vth, k=k)
    rms_error = _rms(predicted - ids[sat_mask])
    return MosfetFitResult(
        method=f"Eq. 5.20 saturation fit, green largest VDS = {vds:.3g} V, Id>{id_threshold:g}",
        vth=float(vth),
        k=float(k),
        points_used=int(np.count_nonzero(sat_mask)),
        rms_error=rms_error,
    )


def fit_saturation_largest_vds_in_id_window(
    device: DeviceCurves,
    *,
    id_min: float = DEFAULT_ID_THRESHOLD,
    id_max: float | None = DEFAULT_ID_MAX_THRESHOLD,
) -> MosfetFitResult:
    if id_max is not None and id_max <= id_min:
        raise FitError("Fit Id max must be greater than Fit Id min.")

    curve, vds = _largest_vds_transfer_curve(device)
    vgs = np.asarray(curve.x, dtype=float)
    ids = np.asarray(curve.y, dtype=float)
    finite = _id_window_mask(vgs, ids, id_min, id_max)
    if np.count_nonzero(finite) < 2:
        raise FitError(_id_window_error("largest-VDS transfer curve", id_min, id_max))

    k, vth = _fit_saturation_once(vgs[finite], ids[finite])
    sat_mask = _saturation_mask(vgs, ids, vds, vth, id_min, id_max)
    if np.count_nonzero(sat_mask) < 2:
        raise FitError("Could not identify enough saturation-region points inside the selected Id window.")

    for _ in range(6):
        k, vth = _fit_saturation_once(vgs[sat_mask], ids[sat_mask])
        next_mask = _saturation_mask(vgs, ids, vds, vth, id_min, id_max)
        if np.count_nonzero(next_mask) < 2:
            break
        if np.array_equal(next_mask, sat_mask):
            sat_mask = next_mask
            break
        sat_mask = next_mask

    k, vth = _fit_saturation_once(vgs[sat_mask], ids[sat_mask])
    predicted = _saturation_id(vgs[sat_mask], vth=vth, k=k)
    rms_error = _rms(predicted - ids[sat_mask])
    return MosfetFitResult(
        method=f"Eq. 5.20 saturation fit, green largest VDS = {vds:.3g} V, Id {_format_id_window(id_min, id_max)}",
        vth=float(vth),
        k=float(k),
        points_used=int(np.count_nonzero(sat_mask)),
        rms_error=rms_error,
    )


def fit_triode_largest_vds_in_id_window(
    device: DeviceCurves,
    *,
    id_min: float = DEFAULT_ID_THRESHOLD,
    id_max: float | None = DEFAULT_ID_MAX_THRESHOLD,
) -> MosfetFitResult:
    curve, scalar_vds = _largest_vds_transfer_curve(device)
    return _fit_triode_curve_in_id_window(
        curve,
        scalar_vds,
        curve_name="green largest VDS",
        id_min=id_min,
        id_max=id_max,
    )


def fit_triode_smallest_vds_in_id_window(
    device: DeviceCurves,
    *,
    id_min: float = DEFAULT_ID_THRESHOLD,
    id_max: float | None = DEFAULT_ID_MAX_THRESHOLD,
) -> MosfetFitResult:
    curve, scalar_vds = _smallest_vds_transfer_curve(device)
    return _fit_triode_curve_in_id_window(
        curve,
        scalar_vds,
        curve_name="blue smallest VDS",
        id_min=id_min,
        id_max=id_max,
    )


def _fit_triode_curve_in_id_window(
    curve,
    scalar_vds: float,
    *,
    curve_name: str,
    id_min: float,
    id_max: float | None,
) -> MosfetFitResult:
    if id_max is not None and id_max <= id_min:
        raise FitError("Fit Id max must be greater than Fit Id min.")

    vgs = np.asarray(curve.x, dtype=float)
    ids = np.asarray(curve.y, dtype=float)
    vds = np.full_like(vgs, scalar_vds, dtype=float)
    finite_window = _id_window_mask(vgs, ids, id_min, id_max)
    if np.count_nonzero(finite_window) < 2:
        raise FitError(_id_window_error(f"{curve_name} transfer curve", id_min, id_max))
    _, vth = _fit_eq_5_16_once(vgs[finite_window], vds[finite_window], ids[finite_window])
    finite = finite_window & (scalar_vds >= 0.0) & (vgs > vth)
    if np.count_nonzero(finite) < 2:
        raise FitError(_id_window_error(f"{curve_name} transfer curve", id_min, id_max))

    triode_mask = finite & (vds < (vgs - vth))
    if np.count_nonzero(triode_mask) < 2:
        raise FitError(
            f"Could not identify enough triode-region points inside the selected Id window on the {curve_name} curve "
            f"(requires VDS < VGS - Vt; curve VDS={scalar_vds:.3g} V, initial Vt={vth:.4g} V)."
        )

    for _ in range(6):
        k, vth = _fit_eq_5_16_once(vgs[triode_mask], vds[triode_mask], ids[triode_mask])
        next_mask = finite & (vds < (vgs - vth))
        if np.count_nonzero(next_mask) < 2:
            break
        if np.array_equal(next_mask, triode_mask):
            triode_mask = next_mask
            break
        triode_mask = next_mask

    k, vth = _fit_eq_5_16_once(vgs[triode_mask], vds[triode_mask], ids[triode_mask])
    predicted = _triode_id(vgs[triode_mask], vds[triode_mask], vth=vth, k=k)
    return MosfetFitResult(
        method=f"Eq. 5.16 triode fit, {curve_name} = {scalar_vds:.3g} V, Id {_format_id_window(id_min, id_max)}",
        vth=float(vth),
        k=float(k),
        points_used=int(np.count_nonzero(triode_mask)),
        rms_error=_rms(predicted - ids[triode_mask]),
    )


def fit_triode_eq_5_16(device: DeviceCurves, id_threshold: float = DEFAULT_ID_THRESHOLD) -> MosfetFitResult:
    initial = fit_saturation_largest_vds(device, id_threshold=id_threshold)
    curve, scalar_vds = _smallest_vds_transfer_curve(device)
    vgs = np.asarray(curve.x, dtype=float)
    ids = np.asarray(curve.y, dtype=float)
    vds = np.full_like(vgs, scalar_vds, dtype=float)
    finite = np.isfinite(vgs) & np.isfinite(ids) & (scalar_vds >= 0.0) & (ids > id_threshold) & (vgs > initial.vth)
    if np.count_nonzero(finite) < 2:
        raise FitError(f"Need at least two finite blue-curve Id points above {id_threshold:g} mA/mm for Eq. 5.16 fitting.")

    vth = initial.vth
    triode_mask = finite & (vds < (vgs - vth))
    if np.count_nonzero(triode_mask) < 2:
        raise FitError("Could not identify enough above-threshold triode-region points on the smallest-VDS blue curve.")

    for _ in range(4):
        k, vth = _fit_eq_5_16_once(vgs[triode_mask], vds[triode_mask], ids[triode_mask])
        next_mask = finite & (vds < (vgs - vth))
        if np.count_nonzero(next_mask) < 2:
            break
        if np.array_equal(next_mask, triode_mask):
            triode_mask = next_mask
            break
        triode_mask = next_mask

    k, vth = _fit_eq_5_16_once(vgs[triode_mask], vds[triode_mask], ids[triode_mask])
    predicted = _triode_id(vgs[triode_mask], vds[triode_mask], vth=vth, k=k)
    return MosfetFitResult(
        method=f"Eq. 5.16 triode fit, blue smallest VDS = {scalar_vds:.3g} V, Id>{id_threshold:g}",
        vth=float(vth),
        k=float(k),
        points_used=int(np.count_nonzero(triode_mask)),
        rms_error=_rms(predicted - ids[triode_mask]),
    )


def _fit_eq_5_16_once(vgs: np.ndarray, vds: np.ndarray, ids: np.ndarray) -> tuple[float, float]:
    x1 = vgs * vds - 0.5 * vds**2
    x2 = -vds
    design = np.column_stack((x1, x2))
    k, k_vth = np.linalg.lstsq(design, ids, rcond=None)[0]
    if not np.isfinite(k) or k <= 0.0:
        raise FitError("Eq. 5.16 fit failed: fitted k is not positive finite.")
    vth = k_vth / k
    validate_reference_params(vth, k)
    return float(k), float(vth)


def _fit_saturation_once(vgs: np.ndarray, ids: np.ndarray) -> tuple[float, float]:
    fit_sqrt_id = np.sqrt(ids)
    design = np.column_stack((vgs, np.ones_like(vgs)))
    slope, intercept = np.linalg.lstsq(design, fit_sqrt_id, rcond=None)[0]
    if not np.isfinite(slope) or slope <= 0.0:
        raise FitError("Saturation fit failed: sqrt(Id) vs Vgs slope is not positive finite.")
    vth = -intercept / slope
    k = 2.0 * slope**2
    validate_reference_params(vth, k)
    return float(k), float(vth)


def _largest_vds_transfer_curve(device: DeviceCurves):
    candidates = []
    for curve in device.trans_id_vgs.curves:
        vds = parse_bias_value(curve.label, "VDS")
        if vds is not None:
            candidates.append((vds, curve))
    if not candidates:
        raise FitError("Could not parse any VDS values from transfer curve labels.")
    vds, curve = max(candidates, key=lambda item: item[0])
    return curve, vds


def _smallest_vds_transfer_curve(device: DeviceCurves):
    candidates = []
    for curve in device.trans_id_vgs.curves:
        vds = parse_bias_value(curve.label, "VDS")
        if vds is not None:
            candidates.append((vds, curve))
    if not candidates:
        raise FitError("Could not parse any VDS values from transfer curve labels.")
    vds, curve = min(candidates, key=lambda item: item[0])
    return curve, vds


def _transfer_points(device: DeviceCurves) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vgs_values: list[np.ndarray] = []
    vds_values: list[np.ndarray] = []
    id_values: list[np.ndarray] = []
    for curve in device.trans_id_vgs.curves:
        vds = parse_bias_value(curve.label, "VDS")
        if vds is None:
            continue
        x = np.asarray(curve.x, dtype=float)
        y = np.asarray(curve.y, dtype=float)
        vgs_values.append(x)
        vds_values.append(np.full_like(x, vds, dtype=float))
        id_values.append(y)
    if not vgs_values:
        raise FitError("Could not parse any VDS values from transfer curve labels.")
    return np.concatenate(vgs_values), np.concatenate(vds_values), np.concatenate(id_values)


def _triode_id(vgs: np.ndarray, vds: np.ndarray, *, vth: float, k: float) -> np.ndarray:
    return k * ((vgs - vth) * vds - 0.5 * vds**2)


def _saturation_id(vgs: np.ndarray, *, vth: float, k: float) -> np.ndarray:
    return 0.5 * k * (vgs - vth) ** 2


def _saturation_mask(
    vgs: np.ndarray,
    ids: np.ndarray,
    vds: float,
    vth: float,
    id_min: float,
    id_max: float | None,
) -> np.ndarray:
    return _id_window_mask(vgs, ids, id_min, id_max) & (vgs > vth) & (vds >= (vgs - vth))


def _id_window_mask(vgs: np.ndarray, ids: np.ndarray, id_min: float, id_max: float | None) -> np.ndarray:
    mask = np.isfinite(vgs) & np.isfinite(ids) & (ids > id_min)
    if id_max is not None:
        mask &= ids <= id_max
    return mask


def _format_id_window(id_min: float, id_max: float | None) -> str:
    if id_max is None:
        return f"> {id_min:g}"
    return f"{id_min:g}-{id_max:g}"


def _id_window_error(curve_name: str, id_min: float, id_max: float | None) -> str:
    return f"Need at least two finite Id points in the selected Id window ({_format_id_window(id_min, id_max)} mA/mm) on the {curve_name}."


def _rms(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))
