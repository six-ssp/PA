"""维护入口本身的回归测试：检查失败路径，不实际重绘或覆盖成果。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_analysis
from check_repository import close, radius_from_attachment


class MaintenanceTests(unittest.TestCase):
    def test_numeric_check_rejects_nan_and_drift(self):
        for value in [float("nan"), float("inf"), 1.01]:
            with self.assertRaises(ValueError):
                close(value, 1.0, "deliberately invalid")

    def test_independent_radius_interpolation(self):
        samples = [(0, 2.0), (10, 1.5), (20, 1.2)]
        for time, radius in [(-1, 2), (0, 2), (5, 1.75), (10, 1.5), (21, 1.2)]:
            self.assertAlmostEqual(radius_from_attachment(time, samples), radius)

    def test_analysis_missing_inputs_fail_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run_analysis, "ROOT", Path(directory)), patch.object(run_analysis, "analyze_errors") as solver:
                with self.assertRaises(FileNotFoundError):
                    run_analysis.main()
                solver.assert_not_called()

    def test_analysis_reuse_switch_and_execution_order(self):
        names = ["analyze_errors", "generate_figures", "analyze_sensitivity", "generate_illustrations", "generate_method_figures"]
        for reuse in (False, True):
            calls = []
            replacements = {name: (lambda *args, _name=name, **kwargs: calls.append((_name, kwargs))) for name in names}
            with patch.multiple(run_analysis, **replacements), patch.object(Path, "is_file", return_value=True), patch("builtins.print"):
                run_analysis.main(reuse_sensitivity=reuse)
            self.assertEqual([name for name, _ in calls], names)
            self.assertEqual(calls[2][1], {"plot_only": reuse})


if __name__ == "__main__":
    unittest.main()
