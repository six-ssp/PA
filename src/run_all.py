"""运行四问数值模型，并把待导出的二维结果写入 intermediate/*.json。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import (
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


def rounded_matrix(values: np.ndarray) -> list[list[float | None]]:
    """按题目要求保留四位小数；NaN 用于问题 4 已位于药材外部的固定位置。"""
    rounded = np.round(values.astype(float), 4)
    return [[None if np.isnan(v) else float(v) for v in row] for row in rounded]


def regular_times(end_time: float, step: float) -> np.ndarray:
    """生成规则时刻；若烘干结束不在网格上，再追加精确结束时刻。"""
    times = np.arange(0.0, np.floor(end_time / step) * step + 0.1, step)
    if end_time - times[-1] > 1.0e-7:
        times = np.append(times, end_time)
    return times


def save_payload(name: str, payload: dict) -> None:
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    with (INTERMEDIATE / name).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    env = load_environment_xlsx(ROOT / "附件1.xlsx")
    distances_cm = np.arange(0.0, 2.0 + 0.05, 0.1)

    # 问题 1：常物性，0~1800 s，每 1 s 输出。
    result1 = simulate(env, law_problem1(), 1800.0)
    times1 = np.arange(0.0, 1800.0 + 1.0, 1.0)
    temp1, water1 = interpolate_fixed_radius(result1, times1, distances_cm)
    save_payload("result1.json", {
        "time": times1.astype(int).tolist(),
        "distance": np.round(distances_cm, 1).tolist(),
        "temperature": rounded_matrix(temp1),
        "moisture": rounded_matrix(water1),
    })

    # 问题 2：附录 3 变物性，0~3 h，每 1 s 输出。
    result2 = simulate(env, law_problem23(), 3.0 * 3600.0)
    times2 = np.arange(0.0, 3.0 * 3600.0 + 1.0, 1.0)
    temp2, water2 = interpolate_fixed_radius(result2, times2, distances_cm)
    save_payload("result2.json", {
        "time": times2.astype(int).tolist(),
        "distance": np.round(distances_cm, 1).tolist(),
        "temperature": rounded_matrix(temp2),
        "moisture": rounded_matrix(water2),
    })

    # 问题 3：附录 3，事件函数确定所有位置均低于 0.15 kg/kg 的时刻。
    result3 = simulate(env, law_problem23(), 5.0 * 24.0 * 3600.0, stop_at_dry=True)
    if result3.drying_time_s is None:
        raise RuntimeError("问题 3 在 5 天内未达到烘干阈值")
    times3 = regular_times(result3.drying_time_s, 60.0)
    _, water3 = interpolate_fixed_radius(result3, times3, distances_cm)
    save_payload("result3.json", {
        "time": np.round(times3, 4).tolist(),
        "distance": np.round(distances_cm, 1).tolist(),
        "moisture": rounded_matrix(water3),
        "drying_time_s": result3.drying_time_s,
    })

    # 问题 4：附件 2 给出的收缩半径做分段线性插值，数据末端后保持末值。
    radius_time, radius_m = load_radius_xlsx(ROOT / "附件2.xlsx")
    radius_fn = lambda t: np.interp(t, radius_time, radius_m)
    result4 = simulate(
        env,
        law_problem4(),
        5.0 * 24.0 * 3600.0,
        radius=radius_fn,
        stop_at_dry=True,
    )
    if result4.drying_time_s is None:
        raise RuntimeError("问题 4 在 5 天内未达到烘干阈值")
    times4 = regular_times(result4.drying_time_s, 60.0)
    _, fixed_water4 = interpolate_fixed_radius(result4, times4, distances_cm)
    sampled_temp4, sampled_water4 = result4.sample(times4)
    radii4 = np.asarray(radius_fn(times4), dtype=float)
    # 固定物理坐标若已经位于收缩后的药材外部，则留空；末列始终是实时表面值。
    for i, radius_now in enumerate(radii4):
        fixed_water4[i, distances_cm / 100.0 > radius_now + 1.0e-12] = np.nan
    surface_water = sampled_water4[:, -1, None]
    save_payload("result4.json", {
        "time": np.round(times4, 4).tolist(),
        "distance": np.round(distances_cm, 1).tolist() + ["药材表面"],
        "moisture": rounded_matrix(np.hstack((fixed_water4, surface_water))),
        "radius_cm": np.round(radii4 * 100.0, 6).tolist(),
        "drying_time_s": result4.drying_time_s,
    })

    # 论文正文所需的摘要表和诊断信息。
    summary_dist = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    p1_times = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
    p2_times = np.arange(0.5, 3.0 + 0.1, 0.5) * 3600.0
    p3_times = regular_times(result3.drying_time_s, 6.0 * 3600.0)
    p4_times = regular_times(result4.drying_time_s, 6.0 * 3600.0)
    p1_t, p1_c = interpolate_fixed_radius(result1, p1_times, summary_dist)
    p2_t, p2_c = interpolate_fixed_radius(result2, p2_times, summary_dist)
    _, p3_c = interpolate_fixed_radius(result3, p3_times, summary_dist)
    _, p4_fixed = interpolate_fixed_radius(result4, p4_times, summary_dist)
    _, p4_grid = result4.sample(p4_times)
    p4_radii = np.asarray(radius_fn(p4_times), dtype=float)
    for i, radius_now in enumerate(p4_radii):
        p4_fixed[i, summary_dist / 100.0 > radius_now + 1.0e-12] = np.nan
    # 最小半径约 1.2 cm，因此正文表 6 可稳定列出 0、0.5、1.0 cm 和实时表面。
    p4_table = np.hstack((p4_fixed[:, :3], p4_grid[:, -1, None]))
    save_payload("summary.json", {
        "problem1_times_s": p1_times.astype(int).tolist(),
        "problem1_temperature": rounded_matrix(p1_t),
        "problem1_moisture": rounded_matrix(p1_c),
        "problem2_times_h": (p2_times / 3600.0).tolist(),
        "problem2_temperature": rounded_matrix(p2_t),
        "problem2_moisture": rounded_matrix(p2_c),
        "problem3_times_h": np.round(p3_times / 3600.0, 6).tolist(),
        "problem3_moisture": rounded_matrix(p3_c),
        "problem3_drying_time_h": result3.drying_time_s / 3600.0,
        "problem4_times_h": np.round(p4_times / 3600.0, 6).tolist(),
        "problem4_columns": [0.0, 0.5, 1.0, "药材表面"],
        "problem4_moisture": rounded_matrix(p4_table),
        "problem4_drying_time_h": result4.drying_time_s / 3600.0,
        "problem4_radius_at_end_cm": float(radius_fn(result4.drying_time_s) * 100.0),
        "solver": {
            "nodes": 161,
            "method": "radial finite volume + BDF",
            "environment_after_14400_s": "hold last measured value",
        },
    })

    print(f"问题 3 烘干时间: {result3.drying_time_s / 3600.0:.6f} h")
    print(f"问题 4 烘干时间: {result4.drying_time_s / 3600.0:.6f} h")


if __name__ == "__main__":
    main()
