"""运行四问、收敛与对照实验，并写出待导出的 intermediate/*.json。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from convergence import run_grid_convergence
from model import (
    Environment,
    SimulationResult,
    interpolate_fixed_radius,
    law_problem1,
    law_problem23,
    law_problem4,
    load_environment_xlsx,
    load_radius_xlsx,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE = ROOT / "intermediate"
RESULTS = ROOT / "results"
SHORT_NODES = 321
FINAL_NODES = 641
MAX_DRYING_TIME_S = 5.0 * 24.0 * 3600.0
CONTROL_MAX_TIME_S = 10.0 * 24.0 * 3600.0


def rounded_matrix(values: np.ndarray) -> list[list[float | None]]:
    """按题目要求保留四位小数；NaN 表示固定坐标已在药材外部。"""
    rounded = np.round(values.astype(float), 4)
    return [[None if np.isnan(v) else float(v) for v in row] for row in rounded]


def strict_regular_times(end_time: float, step: float) -> np.ndarray:
    """生成 0 至第一个不小于 end_time 的规则整点，不插入事件时刻。"""
    output_end = int(np.ceil(end_time / step)) * step
    return np.arange(0.0, output_end + 0.1 * step, step)


def summary_times_with_event(end_time: float, step: float) -> np.ndarray:
    """正文摘要允许在规则时刻后追加连续事件时刻。"""
    times = np.arange(0.0, np.floor(end_time / step) * step + 0.1 * step, step)
    if end_time - times[-1] > 1.0e-7:
        times = np.append(times, end_time)
    return times


def save_payload(name: str, payload: dict) -> None:
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    with (INTERMEDIATE / name).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def drying_time_difference(state_hours: float, legacy_hours: float) -> float:
    """新界面状态法减旧系数平均法的烘干时间，单位 s。"""
    return (state_hours - legacy_hours) * 3600.0


def check_solution(
    name: str,
    result: SimulationResult,
    end_time_s: float,
    environment: Environment,
    *,
    drying_time_s: float | None = None,
) -> None:
    """检查初值、有限性、物理范围、径向梯度及事件阈值。"""
    sample_times = np.unique(np.concatenate((
        np.array([0.0, end_time_s]),
        np.linspace(0.0, end_time_s, 101),
    )))
    temperature, moisture = result.sample(sample_times)

    if not np.allclose(temperature[0], 28.0, atol=1.0e-10):
        raise RuntimeError(f"{name}: 初始温度不是 28 °C")
    if not np.allclose(moisture[0], 2.55, atol=1.0e-10):
        raise RuntimeError(f"{name}: 初始含水率不是 2.55 kg/kg")
    if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(moisture)):
        raise RuntimeError(f"{name}: 数值解含 NaN 或 Inf")
    if float(np.min(moisture)) < -1.0e-8:
        raise RuntimeError(f"{name}: 出现负含水率")

    physical_min = min(28.0, float(np.min(environment.temperature_c)), environment.stable_temperature_c)
    physical_max = max(28.0, float(np.max(environment.temperature_c)), environment.stable_temperature_c)
    if float(np.min(temperature)) < physical_min - 0.1 or float(np.max(temperature)) > physical_max + 0.1:
        raise RuntimeError(f"{name}: 温度超出初值和环境温度构成的合理范围")

    main_stage = sample_times >= min(4.0 * 3600.0, 0.2 * end_time_s)
    if np.any(moisture[main_stage, 0] + 2.0e-5 < moisture[main_stage, -1]):
        raise RuntimeError(f"{name}: 主要干燥阶段中心含水率低于表面")

    if drying_time_s is not None:
        _, event_moisture = result.sample(np.array([drying_time_s]))
        event_max = float(np.max(event_moisture[0]))
        if abs(event_max - 0.15) > 2.0e-5:
            raise RuntimeError(f"{name}: 事件时刻 max(C)={event_max:.8f}，未贴合 0.15")


def check_radius_data(
    radius_time: np.ndarray,
    radius_m: np.ndarray,
    radius_fn,
    end_time_s: float,
) -> None:
    """检查半径正性、原始节点一致性以及数据末端后的末值保持。"""
    probe_times = np.unique(np.concatenate((radius_time, np.linspace(0.0, end_time_s, 1001))))
    probe_radius = np.asarray(radius_fn(probe_times), dtype=float)
    if not np.all(np.isfinite(probe_radius)) or np.any(probe_radius <= 0.0):
        raise RuntimeError("问题 4: R(t) 出现非正值或非有限值")
    if not np.allclose(np.asarray(radius_fn(radius_time)), radius_m, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("问题 4: R(t) 在附件 2 原始节点处不一致")
    if not np.isclose(float(radius_fn(end_time_s)), float(radius_m[-1]), atol=1.0e-12):
        raise RuntimeError("问题 4: 附件 2 数据结束后未保持末个半径")


def main() -> None:
    env = load_environment_xlsx(ROOT / "附件1.xlsx")
    print(
        f"14400 s 后环境边界（{env.stable_start_s:.0f}~{env.time_s[-1]:.0f} s 平均）: "
        f"T_const={env.stable_temperature_c:.8f} °C, "
        f"C_const={env.stable_moisture:.8f} kg/kg"
    )

    radius_time, radius_m = load_radius_xlsx(ROOT / "附件2.xlsx")
    radius_fn = lambda t: np.interp(np.asarray(t, dtype=float), radius_time, radius_m)
    check_radius_data(radius_time, radius_m, radius_fn, MAX_DRYING_TIME_S)

    grid_rows, event3, event4 = run_grid_convergence(
        env,
        radius_fn,
        RESULTS / "grid_convergence.csv",
        end_time_s=MAX_DRYING_TIME_S,
    )
    drying3 = event3.drying_time_s
    drying4 = event4.drying_time_s
    if drying3 is None or drying4 is None:
        raise RuntimeError("最终网格在 5 天内未达到烘干阈值")

    # 同为附录 4 物性，只改变半径模型，隔离纯几何收缩效应。
    fixed_appendix4 = simulate(
        env, law_problem4(), CONTROL_MAX_TIME_S,
        nodes=FINAL_NODES, stop_at_dry=True,
    )
    if fixed_appendix4.drying_time_s is None:
        raise RuntimeError("附录 4 固定半径对照在 10 天内未达到阈值")
    fixed4_time = fixed_appendix4.drying_time_s
    shrink_reduction = (fixed4_time - drying4) / fixed4_time

    # 在同一环境、同一 161 节点网格下隔离新旧界面系数构造方式的影响。
    legacy3 = simulate(
        env, law_problem23(), MAX_DRYING_TIME_S,
        nodes=161, stop_at_dry=True, face_mode="coefficient_average",
    )
    legacy4 = simulate(
        env, law_problem4(), MAX_DRYING_TIME_S,
        radius=radius_fn, nodes=161, stop_at_dry=True, face_mode="coefficient_average",
    )
    if legacy3.drying_time_s is None or legacy4.drying_time_s is None:
        raise RuntimeError("旧界面系数对照在 5 天内未达到阈值")
    state161 = next(row for row in grid_rows if row["nodes"] == 161)

    # 问题 3、4 第二遍求至事件之后的首个 60 s 整点，Excel 时间列严格等间隔。
    times3 = strict_regular_times(drying3, 60.0)
    times4 = strict_regular_times(drying4, 60.0)
    result3 = simulate(env, law_problem23(), float(times3[-1]), nodes=FINAL_NODES)
    result4 = simulate(
        env, law_problem4(), float(times4[-1]),
        radius=radius_fn, nodes=FINAL_NODES,
    )

    # 问题 1、2 用 321 节点；问题 3、4 的正式结果用 641 节点。
    distances_cm = np.arange(0.0, 2.0 + 0.05, 0.1)
    result1 = simulate(env, law_problem1(), 1800.0, nodes=SHORT_NODES)
    times1 = np.arange(0.0, 1800.0 + 1.0, 1.0)
    temp1, water1 = interpolate_fixed_radius(result1, times1, distances_cm)
    save_payload("result1.json", {
        "time": times1.astype(int).tolist(),
        "distance": np.round(distances_cm, 1).tolist(),
        "temperature": rounded_matrix(temp1),
        "moisture": rounded_matrix(water1),
    })

    result2 = simulate(env, law_problem23(), 3.0 * 3600.0, nodes=SHORT_NODES)
    times2 = np.arange(0.0, 3.0 * 3600.0 + 1.0, 1.0)
    temp2, water2 = interpolate_fixed_radius(result2, times2, distances_cm)
    save_payload("result2.json", {
        "time": times2.astype(int).tolist(),
        "distance": np.round(distances_cm, 1).tolist(),
        "temperature": rounded_matrix(temp2),
        "moisture": rounded_matrix(water2),
    })

    _, water3 = interpolate_fixed_radius(result3, times3, distances_cm)
    save_payload("result3.json", {
        "time": times3.astype(int).tolist(),
        "distance": np.round(distances_cm, 1).tolist(),
        "moisture": rounded_matrix(water3),
        "drying_time_s": drying3,
    })

    _, fixed_water4 = interpolate_fixed_radius(result4, times4, distances_cm)
    _, sampled_water4 = result4.sample(times4)
    radii4 = np.asarray(radius_fn(times4), dtype=float)
    # 固定坐标越过实时表面后留空；最后一列始终取材料坐标 x=1 的实时表面值。
    for i, radius_now in enumerate(radii4):
        fixed_water4[i, distances_cm / 100.0 > radius_now + 1.0e-12] = np.nan
    save_payload("result4.json", {
        "time": times4.astype(int).tolist(),
        "distance": np.round(distances_cm, 1).tolist() + ["药材表面"],
        "moisture": rounded_matrix(np.hstack((fixed_water4, sampled_water4[:, -1, None]))),
        "radius_cm": np.round(radii4 * 100.0, 6).tolist(),
        "drying_time_s": drying4,
    })

    # 所有检查在 Excel 导出前完成，失败即中止，不留下看似有效的新结果。
    check_solution("问题 1", result1, float(times1[-1]), env)
    check_solution("问题 2", result2, float(times2[-1]), env)
    check_solution("问题 3", result3, float(times3[-1]), env, drying_time_s=drying3)
    check_solution("问题 4", result4, float(times4[-1]), env, drying_time_s=drying4)
    check_solution(
        "附录 4 固定半径对照", fixed_appendix4, fixed4_time,
        env, drying_time_s=fixed4_time,
    )
    if not np.allclose(np.diff(times3), 60.0) or not np.allclose(np.diff(times4), 60.0):
        raise RuntimeError("问题 3/4 输出时间列不是严格 60 s 间隔")
    if times3[-1] < drying3 or times4[-1] < drying4:
        raise RuntimeError("问题 3/4 输出未覆盖连续事件时刻之后的首个 60 s 整点")

    # 论文正文所需的摘要表和可复核诊断信息。
    summary_dist = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    p1_times = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
    p2_times = np.arange(0.5, 3.0 + 0.1, 0.5) * 3600.0
    p3_times = summary_times_with_event(drying3, 6.0 * 3600.0)
    p4_times = summary_times_with_event(drying4, 6.0 * 3600.0)
    p1_t, p1_c = interpolate_fixed_radius(result1, p1_times, summary_dist)
    p2_t, p2_c = interpolate_fixed_radius(result2, p2_times, summary_dist)
    _, p3_c = interpolate_fixed_radius(event3, p3_times, summary_dist)
    _, p4_fixed = interpolate_fixed_radius(event4, p4_times, summary_dist)
    _, p4_grid = event4.sample(p4_times)
    p4_radii = np.asarray(radius_fn(p4_times), dtype=float)
    for i, radius_now in enumerate(p4_radii):
        p4_fixed[i, summary_dist / 100.0 > radius_now + 1.0e-12] = np.nan
    p4_table = np.hstack((p4_fixed[:, :3], p4_grid[:, -1, None]))

    summary = {
        "environment_stable_start_s": env.stable_start_s,
        "environment_stable_end_s": float(env.time_s[-1]),
        "T_const_c": env.stable_temperature_c,
        "C_const_kg_per_kg": env.stable_moisture,
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
        "problem3_excel_end_s": int(times3[-1]),
        "problem4_times_h": np.round(p4_times / 3600.0, 9).tolist(),
        "problem4_columns": [0.0, 0.5, 1.0, "药材表面"],
        "problem4_moisture": rounded_matrix(p4_table),
        "problem4_drying_time_s": drying4,
        "problem4_drying_time_h": drying4 / 3600.0,
        "problem4_excel_end_s": int(times4[-1]),
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
                float(state161["problem3_time_h"]), legacy3.drying_time_s / 3600.0
            ),
            "state_face_problem4_time_h": float(state161["problem4_time_h"]),
            "coefficient_average_problem4_time_h": legacy4.drying_time_s / 3600.0,
            "difference_problem4_s": drying_time_difference(
                float(state161["problem4_time_h"]), legacy4.drying_time_s / 3600.0
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
