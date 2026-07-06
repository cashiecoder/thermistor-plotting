from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from transistor_plotter.ideal_mosfet import ideal_id
from transistor_plotter.models import Curve, CurveSet, DeviceCurves, SensorFiles
from transistor_plotter.mosfet_fitting import (
    FitError,
    fit_reference_from_transfer_regions,
    fit_saturation_largest_vds,
    fit_triode_eq_5_16,
)


class MosfetFittingTests(unittest.TestCase):
    def test_saturation_fit_recovers_synthetic_parameters(self) -> None:
        device = _synthetic_device(vth=0.25, k=12.0)
        result = fit_saturation_largest_vds(device, id_threshold=0.0)
        self.assertAlmostEqual(result.vth, 0.25, places=10)
        self.assertAlmostEqual(result.k, 12.0, places=10)
        self.assertGreater(result.points_used, 2)

    def test_triode_fit_recovers_synthetic_parameters(self) -> None:
        device = _synthetic_device(vth=0.25, k=12.0)
        result = fit_triode_eq_5_16(device, id_threshold=0.0)
        self.assertAlmostEqual(result.vth, 0.25, places=10)
        self.assertAlmostEqual(result.k, 12.0, places=10)
        self.assertGreater(result.points_used, 2)
        self.assertAlmostEqual(result.rms_error, 0.0, places=10)

    def test_reference_fit_combines_blue_triode_and_green_saturation_results(self) -> None:
        device = _synthetic_device(vth=0.25, k=12.0)
        result = fit_reference_from_transfer_regions(device, id_threshold=0.0)
        self.assertEqual(result.triode.method.startswith("Eq. 5.16 triode fit, blue"), True)
        self.assertEqual(result.saturation.method.startswith("Eq. 5.20 saturation fit, green"), True)
        self.assertAlmostEqual(result.triode.vth, 0.25, places=10)
        self.assertAlmostEqual(result.saturation.vth, 0.25, places=10)

    def test_missing_vds_labels_raise_clear_error(self) -> None:
        sensor = SensorFiles("test", "missing", Path("d"), Path("i"), Path("t"))
        bad_transfer = CurveSet(
            "Transfer",
            "Vgs",
            "Id",
            (Curve("not a bias label", np.array([0.0, 1.0]), np.array([0.0, 1.0])),),
        )
        empty = CurveSet("", "", "", ())
        device = DeviceCurves(sensor, empty, empty, bad_transfer, empty)
        with self.assertRaisesRegex(FitError, "Could not parse any VDS"):
            fit_triode_eq_5_16(device)


def _synthetic_device(vth: float, k: float) -> DeviceCurves:
    sensor = SensorFiles("test", "synthetic", Path("d"), Path("i"), Path("t"))
    empty = CurveSet("", "", "", ())

    curves: list[Curve] = []
    triode_vgs = np.linspace(0.6, 1.4, 16)
    for vds in (0.05, 0.10, 0.15):
        ids = k * ((triode_vgs - vth) * vds - 0.5 * vds**2)
        curves.append(Curve(f"VDS = {vds:.3f} V", triode_vgs, ids))

    saturation_vgs = np.linspace(0.3, 1.4, 20)
    curves.append(Curve("VDS = 2.000 V", saturation_vgs, ideal_id(saturation_vgs, 2.0, vth=vth, k=k)))

    transfer = CurveSet("Transfer", "Vgs", "Id", tuple(curves))
    return DeviceCurves(sensor, empty, empty, transfer, empty)


if __name__ == "__main__":
    unittest.main()
