"""PA_FULL_REGRESSION=1 时重新求解问题 2~4；只读检查，不覆盖正式成果。"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import law_problem23, law_problem4, load_environment_xlsx, simulate
from problem_utils import ROOT, check_solution, load_radius_function


@unittest.skipUnless(os.environ.get("PA_FULL_REGRESSION") == "1", "设置 PA_FULL_REGRESSION=1 启用长时回归")
class FullRegressionTests(unittest.TestCase):
    def test_problem2_three_hours(self):
        env = load_environment_xlsx(ROOT / "附件1.xlsx")
        result = simulate(env, law_problem23(), 10800, nodes=321)
        t, c = result.sample(np.array([10800.]))
        np.testing.assert_allclose(t[0, [0, -1]], [49.8498, 49.9665], atol=6e-5, rtol=0)
        np.testing.assert_allclose(c[0, [0, -1]], [1.7662, 1.0081], atol=6e-5, rtol=0)
        check_solution("problem2 regression", result, 10800, env)

    def test_problem3_and_4_events(self):
        env = load_environment_xlsx(ROOT / "附件1.xlsx")
        summary = json.loads((ROOT / "intermediate/summary.json").read_text(encoding="utf-8"))
        _, _, radius = load_radius_function()
        for q, law, radial in [(3, law_problem23(), None), (4, law_problem4(), radius)]:
            with self.subTest(problem=q):
                result = simulate(env, law, 432000, radius=radial, nodes=641, stop_at_dry=True)
                self.assertIsNotNone(result.drying_time_s)
                # 跨平台 SciPy 允许 1 s 漂移，仍远小于 60 s 的正式输出间隔。
                self.assertAlmostEqual(result.drying_time_s, summary[f"problem{q}_drying_time_s"], delta=1.0)
                check_solution(f"problem{q} regression", result, result.drying_time_s, env, drying_time_s=result.drying_time_s)


if __name__ == "__main__":
    unittest.main()
