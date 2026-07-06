from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from transistor_plotter.models import Curve, CurveSet, DeviceCurves, SensorFiles
from transistor_plotter.plotting import plot_single_device


class PlottingTests(unittest.TestCase):
    def test_gm_reference_draws_one_reference_per_curve(self) -> None:
        sensor = SensorFiles("group", "device", Path("d"), Path("i"), Path("t"))
        x = np.array([0.0, 0.5, 1.0])
        basic_curve_set = CurveSet("Panel", "x", "y", (Curve("VDS = 0.100 V", x, np.array([0.0, 1.0, 2.0])),))
        gm_curves = (
            Curve("VDS = 0.100 V", x, np.array([0.0, 1.0, 2.0])),
            Curve("VDS = 0.500 V", x, np.array([0.0, 2.0, 4.0])),
            Curve("VDS = 0.900 V", x, np.array([0.0, 3.0, 6.0])),
        )
        device = DeviceCurves(
            sensor=sensor,
            diode_ig_vgs=basic_curve_set,
            iv_id_vds=CurveSet("Panel", "x", "y", (Curve("VGS = 0.100 V", x, np.array([0.0, 1.0, 2.0])),)),
            trans_id_vgs=basic_curve_set,
            trans_gm_vgs=CurveSet("Transconductance", "Vgs", "gm", gm_curves),
        )

        figure = Figure()
        plot_single_device(figure, device, show_reference=True, reference_vth=0.0, reference_k=1.0)

        gm_axis = figure.axes[3]
        reference_lines = [line for line in gm_axis.lines if line.get_linestyle() == "--"]
        self.assertEqual(len(reference_lines), 3)
        self.assertEqual([line.get_color() for line in reference_lines], ["#1f77b4", "#ff7f0e", "#2ca02c"])


if __name__ == "__main__":
    unittest.main()
