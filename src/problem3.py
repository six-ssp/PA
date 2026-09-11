"""问题 3：固定半径、变物性条件下求全空间达到干燥阈值的时刻。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import Environment, SimulationResult, interpolate_fixed_radius, law_problem23, load_environment_xlsx, simulate
from problem_utils import FINAL_NODES, MAX_DRYING_TIME_S, OUTPUT_DISTANCES_CM, ROOT, check_solution, print_environment, rounded_matrix, save_payload, strict_regular_times


@dataclass
class Problem3Output:
    """问题 3 的事件解、规则时间输出解和官方表格数组。"""

    event_simulation: SimulationResult
    simulation: SimulationResult
    drying_time_s: float
    times_s: np.ndarray
    distances_cm: np.ndarray
    moisture: np.ndarray


def solve_problem3(
    environment: Environment,
    *,
    nodes: int = FINAL_NODES,
    max_time_s: float = MAX_DRYING_TIME_S,
    event_simulation: SimulationResult | None = None,
    save: bool = True,
) -> Problem3Output:
    """先连续定位干燥事件，再求至其后首个 60 s 整点。"""
    event_result = event_simulation or simulate(
        environment, law_problem23(), max_time_s, nodes=nodes, stop_at_dry=True,
    )
    if event_result.x.size != nodes:
        raise ValueError("传入的问题 3 事件解节点数与 nodes 不一致")
    if event_result.drying_time_s is None:
        raise RuntimeError("问题 3 在给定最长时间内未达到含水率阈值 0.15")

    drying_time_s = float(event_result.drying_time_s)
    times_s = strict_regular_times(drying_time_s, 60.0)
    simulation = simulate(environment, law_problem23(), float(times_s[-1]), nodes=nodes)
    _, moisture = interpolate_fixed_radius(simulation, times_s, OUTPUT_DISTANCES_CM)

    check_solution("问题 3", simulation, float(times_s[-1]), environment, drying_time_s=drying_time_s)
    if not np.allclose(np.diff(times_s), 60.0) or times_s[-1] < drying_time_s:
        raise RuntimeError("问题 3 输出时间列或事件覆盖范围不正确")

    if save:
        save_payload("result3.json", {
            "time": times_s.astype(int).tolist(),
            "distance": np.round(OUTPUT_DISTANCES_CM, 1).tolist(),
            "moisture": rounded_matrix(moisture),
            "drying_time_s": drying_time_s,
        })
    return Problem3Output(
        event_result, simulation, drying_time_s, times_s,
        OUTPUT_DISTANCES_CM.copy(), moisture,
    )


def main() -> None:
    environment = load_environment_xlsx(ROOT / "附件1.xlsx")
    print_environment(environment)
    output = solve_problem3(environment)
    print(
        f"问题 3 完成：t_dry={output.drying_time_s:.6f} s "
        f"({output.drying_time_s / 3600.0:.9f} h)，结果已写入 intermediate/result3.json"
    )


if __name__ == "__main__":
    main()
