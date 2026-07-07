from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from transistor_plotter.models import Curve, CurveSet, DeviceCurves, SensorFiles
from transistor_plotter.operating_point import fit_operating_point_model


class OperatingPointTests(unittest.TestCase):
    def test_heaviside_saturation_fit_recovers_synthetic_parameters(self) -> None:
        device = _synthetic_operating_point_device(vth=-0.05, k=120.0)
        result = fit_operating_point_model(device)
        self.assertAlmostEqual(result.vth, -0.05, places=3)
        self.assertAlmostEqual(result.k, 120.0, places=1)
        self.assertGreater(result.points_used, 3)
        self.assertGreater(result.output_vds.size, 0)
        self.assertLessEqual(float(result.output_vds[-1]), result.vov)


def _synthetic_operating_point_device(vth: float, k: float) -> DeviceCurves:
    sensor = SensorFiles("test", "operating", Path("d"), Path("i"), Path("t"))
    empty = CurveSet("", "", "", ())
    vgs = np.linspace(-0.3, 0.35, 80)
    vov = vgs - vth
    ids = 0.5 * k * np.square(vov) * np.where(vov >= 0.0, 1.0, 0.0)
    transfer = CurveSet("Transfer", "Vgs", "Id", (Curve("VDS = 0.500 V", vgs, ids),))
    vds = np.linspace(0.0, 1.0, 80)
    output_vgs = 0.1
    output_ids = k * ((output_vgs - vth) * vds - 0.5 * np.square(vds))
    output = CurveSet("Output", "Vds", "Id", (Curve("VGS = 0.100 V", vds, output_ids),))
    return DeviceCurves(sensor, empty, output, transfer, empty)


if __name__ == "__main__":
    unittest.main()
