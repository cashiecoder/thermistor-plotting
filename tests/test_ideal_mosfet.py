from __future__ import annotations

import unittest

import numpy as np

from transistor_plotter.ideal_mosfet import (
    ideal_gate_leakage,
    ideal_gm,
    ideal_id,
    validate_reference_params,
)


class IdealMosfetTests(unittest.TestCase):
    def test_cutoff_gives_zero_id_and_gm(self) -> None:
        vgs = np.array([-0.4, -0.2, 0.0])
        np.testing.assert_allclose(ideal_id(vgs, 0.5, vth=0.0, k=20.0), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(ideal_gm(vgs, 0.5, vth=0.0, k=20.0), [0.0, 0.0, 0.0])

    def test_triode_id(self) -> None:
        vgs = 1.2
        vds = 0.3
        vth = 0.4
        k = 10.0
        expected = k * ((vgs - vth) * vds - 0.5 * vds**2)
        self.assertAlmostEqual(float(ideal_id(vgs, vds, vth=vth, k=k)), expected)

    def test_saturation_id(self) -> None:
        vgs = 1.2
        vds = 1.0
        vth = 0.4
        k = 10.0
        expected = 0.5 * k * (vgs - vth) ** 2
        self.assertAlmostEqual(float(ideal_id(vgs, vds, vth=vth, k=k)), expected)

    def test_triode_gm(self) -> None:
        self.assertAlmostEqual(float(ideal_gm(1.2, 0.3, vth=0.4, k=10.0)), 3.0)

    def test_saturation_gm(self) -> None:
        self.assertAlmostEqual(float(ideal_gm(1.2, 1.0, vth=0.4, k=10.0)), 8.0)

    def test_gate_leakage_zero_shape(self) -> None:
        vgs = np.array([-1.0, 0.0, 1.0])
        leakage = ideal_gate_leakage(vgs)
        np.testing.assert_array_equal(leakage, np.zeros_like(vgs))
        self.assertEqual(leakage.shape, vgs.shape)

    def test_vectorized_inputs(self) -> None:
        vgs = np.array([0.2, 0.8, 1.2])
        vds = np.array([0.5, 0.2, 1.0])
        actual_id = ideal_id(vgs, vds, vth=0.4, k=10.0)
        actual_gm = ideal_gm(vgs, vds, vth=0.4, k=10.0)
        np.testing.assert_allclose(actual_id, [0.0, 0.6, 3.2])
        np.testing.assert_allclose(actual_gm, [0.0, 2.0, 8.0])

    def test_invalid_reference_params_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference_vth"):
            validate_reference_params(float("nan"), 1.0)
        with self.assertRaisesRegex(ValueError, "reference_k"):
            validate_reference_params(0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "reference_k"):
            ideal_id(1.0, 1.0, vth=0.2, k=float("inf"))


if __name__ == "__main__":
    unittest.main()
