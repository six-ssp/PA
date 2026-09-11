"""问题 1：常物性圆柱在前 30 min 内的温度场与水分场。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import Environment, SimulationResult, interpolate_fixed_radius, law_problem1, load_environment_xlsx, simulate
from problem_utils import OUTPUT_DISTANCES_CM, ROOT, SHORT_NODES, check_solution, print_environment, rounded_matrix, save_payload


@dataclass
class Problem1Output:
    """问题 1 的数值解及官方表格所需数组。"""

    simulation: SimulationResult
    times_s: np.ndarray
    distances_cm: np.ndarray
    temperature_c: np.ndarray
    moisture: np.ndarray


def solve_problem1(environment: Environment, *, nodes: int = SHORT_NODES, save: bool = True) -> Problem1Output:
    """求解问题 1；默认写出 ``intermediate/result1.json``。"""
    # 问题 1 只考察前 30 min，并采用附录 2 给出的常物性参数。
    end_time_s = 30.0 * 60.0
    simulation = simulate(environment, law_problem1(), end_time_s, nodes=nodes)

    # 官方结果表要求每 1 s、每 0.1 cm 输出一次，故将数值解插值到这些位置。
    times_s = np.arange(0.0, end_time_s + 1.0, 1.0)
    temperature_c, moisture = interpolate_fixed_radius(simulation, times_s, OUTPUT_DISTANCES_CM)

    # 写文件前检查初值、数值有限性和温度/含水率的物理范围。
    check_solution("问题 1", simulation, end_time_s, environment)

    if save:
        # JSON 是 Excel 导出脚本的数据源，场变量按题目要求保留 4 位小数。
        save_payload("result1.json", {
            "time": times_s.astype(int).tolist(),
            "distance": np.round(OUTPUT_DISTANCES_CM, 1).tolist(),
            "temperature": rounded_matrix(temperature_c),
            "moisture": rounded_matrix(moisture),
        })
    return Problem1Output(simulation, times_s, OUTPUT_DISTANCES_CM.copy(), temperature_c, moisture)


def main() -> None:
    # 直接执行本文件时，只读取公共输入并计算问题 1。
    environment = load_environment_xlsx(ROOT / "附件1.xlsx")
    print_environment(environment)
    output = solve_problem1(environment)
    print(f"问题 1 完成：{output.times_s.size} 个时间点，结果已写入 intermediate/result1.json")


if __name__ == "__main__":
    main()
