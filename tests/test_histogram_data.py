from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from transistor_plotter.histogram_data import extract_histogram_sample
from transistor_plotter.models import Curve, CurveSet, DeviceCurves, SensorFiles


class HistogramDataTests(unittest.TestCase):
    def test_extract_histogram_sample_uses_target_biases(self) -> None:
        sensor = SensorFiles("group", "device", Path("d"), Path("i"), Path("t"))
        x_vgs = np.array([0.0, 0.2])
        x_vds = np.array([0.0, 1.0])
        device = DeviceCurves(
            sensor=sensor,
            diode_ig_vgs=CurveSet(
                "Gate leakage",
                "Vgs",
                "Ig",
                (
                    Curve("VDS = 0.000 V", x_vgs, np.array([1.0, 3.0])),
                    Curve("VDS = 0.500 V", x_vgs, np.array([2.0, 6.0])),
                ),
            ),
            iv_id_vds=CurveSet(
                "Output",
                "Vds",
                "Id",
                (
                    Curve("VGS = -0.300 V", x_vds, np.array([1.0, 5.0])),
                    Curve("VGS = 0.100 V", x_vds, np.array([10.0, 30.0])),
                    Curve("VGS = 0.500 V", x_vds, np.array([20.0, 60.0])),
                ),
            ),
            trans_id_vgs=CurveSet(
                "Transfer",
                "Vgs",
                "Id",
                (
                    Curve("VDS = 0.100 V", x_vgs, np.array([3.0, 5.0])),
                    Curve("VDS = 0.500 V", x_vgs, np.array([4.0, 8.0])),
                    Curve("VDS = 0.900 V", x_vgs, np.array([5.0, 15.0])),
                ),
            ),
            trans_gm_vgs=CurveSet("", "", "", ()),
        )

        sample = extract_histogram_sample(device)
        self.assertAlmostEqual(sample.values["diode_ig_vds_0p0"], 2.0)
        self.assertAlmostEqual(sample.values["diode_ig_vds_0p5"], 4.0)
        self.assertAlmostEqual(sample.values["trans_id_vds_0p1"], 4.0)
        self.assertAlmostEqual(sample.values["trans_id_vds_0p5"], 6.0)
        self.assertAlmostEqual(sample.values["trans_id_vds_0p9"], 10.0)
        self.assertAlmostEqual(sample.values["iv_id_vgs_n0p3"], 3.0)
        self.assertAlmostEqual(sample.values["iv_id_vgs_0p1"], 20.0)
        self.assertAlmostEqual(sample.values["iv_id_vgs_0p5"], 40.0)


if __name__ == "__main__":
    unittest.main()
