"""快速数值回归：保护原模型、单位换算与输出约定，不依赖私人论文。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import Environment, law_problem1, law_problem23, law_problem4, load_environment_xlsx, simulate
from problem_utils import ROOT, load_radius_function, rounded_matrix, strict_regular_times


class ModelTests(unittest.TestCase):
    def test_environment_interpolation_and_stable_mean(self):
        env = load_environment_xlsx(ROOT / "附件1.xlsx")
        close_time = 0.5 * (env.time_s[0] + env.time_s[1])
        self.assertAlmostEqual(float(env.temperature(close_time)), np.mean(env.temperature_c[:2]))
        self.assertAlmostEqual(float(env.temperature(14400)), env.temperature_c[-1])
        self.assertAlmostEqual(float(env.temperature(14401)), 49.99875609756098)
        self.assertAlmostEqual(float(env.water(14401)), 0.04998414634146341)

    def test_radius_units_and_endpoint_hold(self):
        times, radii, radius = load_radius_function()
        np.testing.assert_allclose(radius(times), radii, atol=1e-12)
        self.assertAlmostEqual(float(radius(times[0] - 1)), 0.02)
        self.assertAlmostEqual(float(radius(times[-1] + 1)), 0.01198)
        self.assertAlmostEqual(float(radius(183934.724387)), 0.012)

    def test_const_heat_properties_not_const_diffusivity(self):
        law = law_problem1()
        c = np.array([0.15, 2.55])
        np.testing.assert_array_equal(law.density(c), [820, 820])
        self.assertGreater(law.diffusivity(c, 323.15)[1], law.diffusivity(c, 323.15)[0])

    def test_arrhenius_temperature_uses_kelvin(self):
        for law, prefactor, moisture_factor in [(law_problem23(), 2.4e-3, 0.45), (law_problem4(), 4.2e-4, 0.30)]:
            c = np.array([0.15, 1.0, 2.55])
            expected = prefactor * np.exp(-moisture_factor / c) * np.exp(-3850 / 323.15)
            np.testing.assert_allclose(law.diffusivity(c, 323.15), expected)
            self.assertTrue(np.all(law.diffusivity(c, 333.15) > expected))

    def test_output_rounding_and_event_grid(self):
        self.assertEqual(rounded_matrix(np.array([[1.234567, np.nan]])), [[1.2346, None]])
        for end, expected in [(120, 120), (120.01, 180), (183934.724387, 183960)]:
            times = strict_regular_times(end, 60)
            self.assertEqual(times[-1], expected)
            np.testing.assert_array_equal(np.diff(times), 60)

    def test_equilibrium_remains_uniform(self):
        env = Environment(np.array([0., 100.]), np.array([28., 28.]), np.array([2.55, 2.55]), 28., 2.55, 0.)
        for law in (law_problem1(), law_problem23(), law_problem4()):
            result = simulate(env, law, 100, nodes=21)
            t, c = result.sample(np.array([0., 50., 100.]))
            np.testing.assert_allclose(t, 28, atol=1e-10)
            np.testing.assert_allclose(c, 2.55, atol=1e-10)

    def test_problem1_published_short_time_regression(self):
        env = load_environment_xlsx(ROOT / "附件1.xlsx")
        result = simulate(env, law_problem1(), 1800, nodes=321)
        t, c = result.sample(np.array([1800.]))
        np.testing.assert_allclose(t[0, [0, -1]], [33.5753, 36.7852], atol=6e-5, rtol=0)
        np.testing.assert_allclose(c[0, [0, -1]], [2.55, 1.5102], atol=6e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
