"""统一运行四问、收敛与对照实验，并写出全部中间结果。"""

from __future__ import annotations

import numpy as np

from convergence import run_grid_convergence
from model import interpolate_fixed_radius, law_problem23, law_problem4, load_environment_xlsx, simulate
from problem1 import solve_problem1
from problem2 import solve_problem2
from problem3 import solve_problem3
from problem4 import solve_problem4
from problem_utils import (
    CONTROL_MAX_TIME_S,
    FINAL_NODES,
    MAX_DRYING_TIME_S,
    RESULTS,
    ROOT,
    SHORT_NODES,
    check_radius_data,
    check_solution,
    load_radius_function,
    print_environment,
    rounded_matrix,
    save_payload,
    summary_times_with_event,
)


def drying_time_difference(state_hours: float, legacy_hours: float) -> float:
    """新界面状态法减旧系数平均法的烘干时间，单位 s。"""
    return (state_hours - legacy_hours) * 3600.0


def main() -> None:
    environment = load_environment_xlsx(ROOT / "附件1.xlsx")
    print_environment(environment)

    radius_data = load_radius_function()
    radius_time, radius_m, radius_fn = radius_data
    check_radius_data(radius_time, radius_m, radius_fn, MAX_DRYING_TIME_S)

    # 641 节点事件解由收敛计算直接复用，避免重复求解第一遍。
    grid_rows, event3, event4 = run_grid_convergence(
        environment,
        radius_fn,
        RESULTS / "grid_convergence.csv",
        end_time_s=MAX_DRYING_TIME_S,
    )
    if event3.drying_time_s is None or event4.drying_time_s is None:
        raise RuntimeError("最终网格在 5 天内未达到烘干阈值")

    # 四个模块各自完成对应问题的求解、检查和 JSON 输出。
    problem1 = solve_problem1(environment)
    problem2 = solve_problem2(environment)
    problem3 = solve_problem3(environment, event_simulation=event3)
    problem4 = solve_problem4(
        environment,
        event_simulation=event4,
        radius_data=radius_data,
    )
    drying3 = problem3.drying_time_s
    drying4 = problem4.drying_time_s

    # 同为附录 4 物性，只改变半径模型，隔离纯几何收缩效应。
    fixed_appendix4 = simulate(
        environment, law_problem4(), CONTROL_MAX_TIME_S,
        nodes=FINAL_NODES, stop_at_dry=True,
    )
    if fixed_appendix4.drying_time_s is None:
        raise RuntimeError("附录 4 固定半径对照在 10 天内未达到阈值")
    fixed4_time = float(fixed_appendix4.drying_time_s)
    shrink_reduction = (fixed4_time - drying4) / fixed4_time
    check_solution(
        "附录 4 固定半径对照", fixed_appendix4, fixed4_time,
        environment, drying_time_s=fixed4_time,
    )

    # 在同一环境、同一 161 节点网格下隔离新旧界面系数构造方式的影响。
    legacy3 = simulate(
        environment, law_problem23(), MAX_DRYING_TIME_S,
        nodes=161, stop_at_dry=True, face_mode="coefficient_average",
    )
    legacy4 = simulate(
        environment, law_problem4(), MAX_DRYING_TIME_S,
        radius=radius_fn, nodes=161, stop_at_dry=True, face_mode="coefficient_average",
    )
    if legacy3.drying_time_s is None or legacy4.drying_time_s is None:
        raise RuntimeError("旧界面系数对照在 5 天内未达到阈值")
    state161 = next(row for row in grid_rows if row["nodes"] == 161)

    # 论文正文所需的摘要表和可复核诊断信息。
    summary_dist = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    p1_times = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
    p2_times = np.arange(0.5, 3.0 + 0.1, 0.5) * 3600.0
    p3_times = summary_times_with_event(drying3, 6.0 * 3600.0)
    p4_times = summary_times_with_event(drying4, 6.0 * 3600.0)
    p1_t, p1_c = interpolate_fixed_radius(problem1.simulation, p1_times, summary_dist)
    p2_t, p2_c = interpolate_fixed_radius(problem2.simulation, p2_times, summary_dist)
    _, p3_c = interpolate_fixed_radius(problem3.event_simulation, p3_times, summary_dist)
    _, p4_fixed = interpolate_fixed_radius(problem4.event_simulation, p4_times, summary_dist)
    _, p4_grid = problem4.event_simulation.sample(p4_times)
    p4_radii = np.asarray(radius_fn(p4_times), dtype=float)
    for row, radius_now in enumerate(p4_radii):
        p4_fixed[row, summary_dist / 100.0 > radius_now + 1.0e-12] = np.nan
    p4_table = np.hstack((p4_fixed[:, :3], p4_grid[:, -1, None]))

    summary = {
        "environment_stable_start_s": environment.stable_start_s,
        "environment_stable_end_s": float(environment.time_s[-1]),
        "T_const_c": environment.stable_temperature_c,
        "C_const_kg_per_kg": environment.stable_moisture,
        "problem1_times_s": p1_times.astype(int).tolist(),
        "problem1_temperature": rounded_matrix(p1_t),
        "problem1_moisture": rounded_matrix(p1_c),
        "problem2_times_h": (p2_times / 3600.0).tolist(),
        "problem2_temperature": rounded_matrix(p2_t),
        "problem2_moisture": rounded_matrix(p2_c),
        "problem3_times_h": np.round(p3_times / 3600.0, 9).tolist(),
        "problem3_moisture": rounded_matrix(p3_c),
        "problem3_drying_time_s": drying3,
        "problem3_drying_time_h": drying3 / 3600.0,
        "problem3_excel_end_s": int(problem3.times_s[-1]),
        "problem4_times_h": np.round(p4_times / 3600.0, 9).tolist(),
        "problem4_columns": [0.0, 0.5, 1.0, "药材表面"],
        "problem4_moisture": rounded_matrix(p4_table),
        "problem4_drying_time_s": drying4,
        "problem4_drying_time_h": drying4 / 3600.0,
        "problem4_excel_end_s": int(problem4.times_s[-1]),
        "problem4_radius_at_end_cm": float(radius_fn(drying4) * 100.0),
        "appendix4_fixed_radius_drying_time_s": fixed4_time,
        "appendix4_fixed_radius_drying_time_h": fixed4_time / 3600.0,
        "appendix4_shrinking_radius_drying_time_s": drying4,
        "appendix4_shrinking_radius_drying_time_h": drying4 / 3600.0,
        "appendix4_shrinkage_reduction_fraction": shrink_reduction,
        "appendix4_shrinkage_reduction_percent": 100.0 * shrink_reduction,
        "problem3_to_problem4_change_fraction": (drying3 - drying4) / drying3,
        "grid_convergence": grid_rows,
        "face_discretization_comparison_161": {
            "state_face_problem3_time_h": float(state161["problem3_time_h"]),
            "coefficient_average_problem3_time_h": legacy3.drying_time_s / 3600.0,
            "difference_problem3_s": drying_time_difference(
                float(state161["problem3_time_h"]), legacy3.drying_time_s / 3600.0,
            ),
            "state_face_problem4_time_h": float(state161["problem4_time_h"]),
            "coefficient_average_problem4_time_h": legacy4.drying_time_s / 3600.0,
            "difference_problem4_s": drying_time_difference(
                float(state161["problem4_time_h"]), legacy4.drying_time_s / 3600.0,
            ),
        },
        "previous_reported_161_node_results_h": {
            "problem3": 57.0864,
            "problem4": 50.8068,
        },
        "solver": {
            "problem1_problem2_nodes": SHORT_NODES,
            "problem3_problem4_nodes": FINAL_NODES,
            "method": "conservative radial finite volume + scipy BDF",
            "face_properties": "evaluate k(C_face) and D(C_face,T_face)",
            "environment_after_14400_s": "mean over 12000~14400 s",
            "problem4_coordinate": "material coordinate x=r/R(t)",
        },
        "sanity_checks": "passed",
    }
    save_payload("summary.json", summary)

    print("\n最终结果")
    print(f"问题 3（641 节点）: {drying3 / 3600.0:.9f} h ({drying3:.6f} s)")
    print(f"问题 4（641 节点）: {drying4 / 3600.0:.9f} h ({drying4:.6f} s)")
    print(f"附录 4 固定 R=2 cm: {fixed4_time / 3600.0:.9f} h")
    print(f"附录 4 收缩 R(t):   {drying4 / 3600.0:.9f} h")
    print(f"纯几何收缩时间变化: {100.0 * shrink_reduction:.6f}%")
    print("sanity checks: passed")


if __name__ == "__main__":
    main()
