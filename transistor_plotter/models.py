from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SensorFiles:
    group: str
    device_id: str
    diode: Path
    iv: Path
    trans: Path

    @property
    def key(self) -> str:
        return f"{self.group}/{self.device_id}"

    @property
    def label(self) -> str:
        return f"{self.device_id} ({self.group})"


@dataclass(frozen=True)
class Curve:
    label: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class CurveSet:
    title: str
    xlabel: str
    ylabel: str
    curves: tuple[Curve, ...]


@dataclass(frozen=True)
class DeviceCurves:
    sensor: SensorFiles
    diode_ig_vgs: CurveSet
    iv_id_vds: CurveSet
    trans_id_vgs: CurveSet
    trans_gm_vgs: CurveSet

    @property
    def panels(self) -> tuple[CurveSet, CurveSet, CurveSet, CurveSet]:
        return (
            self.diode_ig_vgs,
            self.iv_id_vds,
            self.trans_id_vgs,
            self.trans_gm_vgs,
        )
