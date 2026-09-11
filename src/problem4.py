"""问题 4：在材料坐标中考虑半径收缩的变物性烘干模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import Environment, SimulationResult, interpolate_fixed_radius, law_problem4, load_environment_xlsx, simulate
from problem_utils import FINAL_NODES, MAX_DRYING_TIME_S, OUTPUT_DISTANCES_CM, ROOT, RadiusFunction, check_radius_data, check_solution, load_radius_function, print_environment, rounded_matrix, save_payload, strict_regular_times


@dataclass
class Problem4Output:
    """问题 4 的事件解、规则时间输出解、收缩半径和表格数组。"""

    event_simulation: SimulationResult
    simulation: SimulationResult
    drying_time_s: float
    times_s: np.ndarray
    distances_cm: np.ndarray
    fixed_moisture: np.ndarray
    surface_moisture: np.ndarray
    radius_time_s: np.ndarray
    radius_m: np.ndarray
    radius: RadiusFunction


def solve_problem4(
    environment: Environment,
    *,
    nodes: int = FINAL_NODES,
    max_time_s: float = MAX_DRYING_TIME_S,
    event_simulation: SimulationResult | None = None,
    radius_data: tuple[np.ndarray, np.ndarray, RadiusFunction] | None = None,
    save: bool = True,
) -> Problem4Output:
    """读取 R(t)，连续定位干燥事件，并生成严格 60 s 间隔结果。"""
    radius_time, radius_m, radius_fn = radius_data or load_radius_function()
    check_radius_data(radius_time, radius_m, radius_fn, max_time_s)

    event_result = event_simulation or simulate(
        environment, law_problem4(), max_time_s,
        radius=radius_fn, nodes=nodes, stop_at_dry=True,
    )
    if event_result.x.size != nodes:
        raise ValueError("传入的问题 4 事件解节点数与 nodes 不一致")
    if event_result.drying_time_s is None:
        raise RuntimeError("问题 4 在给定最长时间内未达到含水率阈值 0.15")

    drying_time_s = float(event_result.drying_time_s)
    times_s = strict_regular_times(drying_time_s, 60.0)
    simulation = simulate(
        environment, law_problem4(), float(times_s[-1]),
        radius=radius_fn, nodes=nodes,
    )
    _, fixed_moisture = interpolate_fixed_radius(simulation, times_s, OUTPUT_DISTANCES_CM)
    _, grid_moisture = simulation.sample(times_s)
    surface_moisture = grid_moisture[:, -1]
    radii_m = np.asarray(radius_fn(times_s), dtype=float)

    # 固定物理坐标越过实时表面后没有材料，因此官方表中写空值。
    for row, radius_now in enumerate(radii_m):
        fixed_moisture[row, OUTPUT_DISTANCES_CM / 100.0 > radius_now + 1.0e-12] = np.nan

    check_solution("问题 4", simulation, float(times_s[-1]), environment, drying_time_s=drying_time_s)
    if not np.allclose(np.diff(times_s), 60.0) or times_s[-1] < drying_time_s:
        raise RuntimeError("问题 4 输出时间列或事件覆盖范围不正确")

    if save:
        save_payload("result4.json", {
            "time": times_s.astype(int).tolist(),
            "distance": np.round(OUTPUT_DISTANCES_CM, 1).tolist() + ["药材表面"],
            "moisture": rounded_matrix(np.hstack((fixed_moisture, surface_moisture[:, None]))),
            "radius_cm": np.round(radii_m * 100.0, 6).tolist(),
            "drying_time_s": drying_time_s,
        })
    return Problem4Output(
        event_result, simulation, drying_time_s, times_s,
        OUTPUT_DISTANCES_CM.copy(), fixed_moisture, surface_moisture,
        radius_time, radius_m, radius_fn,
    )


def main() -> None:
    environment = load_environment_xlsx(ROOT / "附件1.xlsx")
    print_environment(environment)
    output = solve_problem4(environment)
    print(
        f"问题 4 完成：t_dry={output.drying_time_s:.6f} s "
        f"({output.drying_time_s / 3600.0:.9f} h)，结果已写入 intermediate/result4.json"
    )


if __name__ == "__main__":
    main()
