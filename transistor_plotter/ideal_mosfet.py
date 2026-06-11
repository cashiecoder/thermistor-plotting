from __future__ import annotations

import numpy as np


def validate_reference_params(vth: float, k: float) -> None:
    if not np.isfinite(vth):
        raise ValueError("reference_vth must be a finite voltage in volts.")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("reference_k must be a positive finite value in mA/mm/V^2.")


def ideal_gate_leakage(vgs: np.ndarray | float) -> np.ndarray:
    vgs_arr = np.asarray(vgs, dtype=float)
    return np.zeros_like(vgs_arr, dtype=float)


def ideal_id(
    vgs: np.ndarray | float,
    vds: np.ndarray | float,
    *,
    vth: float,
    k: float,
) -> np.ndarray:
    validate_reference_params(vth, k)

    vgs_arr = np.asarray(vgs, dtype=float)
    vds_arr = np.asarray(vds, dtype=float)
    vgs_b, vds_b = np.broadcast_arrays(vgs_arr, vds_arr)

    vov = vgs_b - vth
    off = vov <= 0.0
    triode = (~off) & (vds_b < vov)

    id_triode = k * (vov * vds_b - 0.5 * vds_b**2)
    id_sat = 0.5 * k * vov**2

    return np.where(off, 0.0, np.where(triode, id_triode, id_sat))


def ideal_gm(
    vgs: np.ndarray | float,
    vds: np.ndarray | float,
    *,
    vth: float,
    k: float,
) -> np.ndarray:
    validate_reference_params(vth, k)

    vgs_arr = np.asarray(vgs, dtype=float)
    vds_arr = np.asarray(vds, dtype=float)
    vgs_b, vds_b = np.broadcast_arrays(vgs_arr, vds_arr)

    vov = vgs_b - vth
    off = vov <= 0.0
    triode = (~off) & (vds_b < vov)

    gm_triode = k * vds_b
    gm_sat = k * vov

    return np.where(off, 0.0, np.where(triode, gm_triode, gm_sat))
