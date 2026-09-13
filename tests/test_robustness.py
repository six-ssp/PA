"""鲁棒性状态分类与输入保护；不在单元测试中运行大范围扫描。"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robustness_analysis import case,coarse_cases,input_error,solve_case,threshold_jobs

class RobustnessTests(unittest.TestCase):
    def test_invalid_inputs_never_reach_solver(self):
        for p,v in [('temperature_c',-273.15),('radius_scale',0),('diffusivity_scale',0),
                    ('moisture',-.1),('boundary_time_scale',0),('radius_scale',float('nan'))]:
            with self.subTest(parameter=p),patch('robustness_analysis.simulate') as solver:
                self.assertEqual(solve_case(case(3,p,v))['status'],'invalid_input')
                solver.assert_not_called()

    def mock_result(self,dry=None,negative=False):
        water=np.tile([2.55,.15 if dry else .8],(5,1))
        if negative:water[0,-1]=-.01
        return SimpleNamespace(drying_time_s=dry,solution=SimpleNamespace(
            y=np.vstack([np.full((5,2),28.),water]),nfev=3,nlu=1))

    def test_no_event_is_deadline_not_solver_failure(self):
        with patch('robustness_analysis.simulate',return_value=self.mock_result()):
            row=solve_case(case(3,nodes=5))
        self.assertEqual(row['status'],'deadline_exceeded')
        self.assertIsNone(row['drying_time_h'])

    def test_solver_exception_is_recorded(self):
        with patch('robustness_analysis.simulate',side_effect=RuntimeError('test failure')):
            row=solve_case(case(3,nodes=5))
        self.assertEqual(row['status'],'solver_failure')
        self.assertIn('test failure',row['message'])

    def test_invalid_state_is_not_clipped_away(self):
        with patch('robustness_analysis.simulate',return_value=self.mock_result(negative=True)):
            row=solve_case(case(3,nodes=5))
        self.assertEqual(row['status'],'nonphysical')
        self.assertLess(row['min_moisture'],0)

    def test_success_checks_event_residual(self):
        with patch('robustness_analysis.simulate',return_value=self.mock_result(7200.)):
            row=solve_case(case(3,nodes=5))
        self.assertEqual(row['status'],'dry')
        self.assertEqual(row['drying_time_h'],2.)

    def test_grid_design_and_stress_path(self):
        cases=coarse_cases()
        self.assertEqual(len(cases),410)
        self.assertEqual(sum(c['role']=='joint' for c in cases),198)
        c=case(4,'adverse',1.)
        self.assertEqual(c['radius_scale'],1.5)
        self.assertEqual(c['diffusivity_scale'],.5)
        self.assertAlmostEqual(c['temperature_c'],case(4)['temperature_c']-20)

    def test_only_observed_valid_crossings_are_refined(self):
        rows=[{**case(3,'radius_scale',v),'status':s} for v,s in
              [(1.,'dry'),(2.,'deadline_exceeded'),(3.,'solver_failure')]]
        self.assertEqual(threshold_jobs(rows),[(3,'radius_scale',1.,2.)])

if __name__=='__main__':unittest.main()
