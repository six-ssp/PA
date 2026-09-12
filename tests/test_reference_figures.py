"""参考样式适配只改变展示方式；检查交互公式与指标，避免视觉改动污染数据。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_reference_figures import interaction, metrics
from sensitivity_surface import OFFSETS, SCALES


class ReferenceFigureTests(unittest.TestCase):
    def test_additive_surface_has_no_interaction(self):
        z = 50 - 2 * OFFSETS[:, None] + 90 * (SCALES[None, :] - 1)
        np.testing.assert_allclose(interaction(z), 0, atol=1e-12)
        stats = metrics({3: z, 4: z - 5})
        self.assertAlmostEqual(stats["3"]["plane_r2"], 1)
        self.assertAlmostEqual(stats["3"]["additive_rmse_h"], 0)
        self.assertEqual(stats["4"]["off_axis_cases"], 192)
        self.assertEqual(stats["paired_cases"]["count"], 221)
        self.assertAlmostEqual(stats["paired_cases"]["median_difference_h"], 5)

    def test_bilinear_interaction_is_preserved(self):
        cross = OFFSETS[:, None] * (SCALES[None, :] - 1)
        z = 50 - 2 * OFFSETS[:, None] + 90 * (SCALES[None, :] - 1) + cross
        np.testing.assert_allclose(interaction(z), cross, atol=1e-12)
        np.testing.assert_allclose(interaction(z)[6, :], 0, atol=1e-12)
        np.testing.assert_allclose(interaction(z)[:, 8], 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
