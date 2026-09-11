"""量化空间、时间离散误差及稳定环境数据敏感性。"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from model import law_problem23, law_problem4, load_environment_xlsx, load_radius_xlsx, simulate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAX_TIME_S = 5.0 * 24.0 * 3600.0
ANALYSIS_NODES = 321


def event_time(environment, law, *, radius=None, max_step_s=60.0, rtol=2e-6, atol=2e-8) -> float:
    """返回指定数值设置下的连续烘干事件时刻。"""
    result = simulate(
        environment,
        law,
        MAX_TIME_S,
        radius=radius,
        nodes=ANALYSIS_NODES,
        stop_at_dry=True,
        max_step_s=max_step_s,
        rtol=rtol,
        atol=atol,
    )
    if result.drying_time_s is None:
        raise RuntimeError("误差分析工况在 5 天内未达到烘干阈值")
    return result.drying_time_s


def richardson(values: list[float]) -> dict[str, float]:
    """由连续三档二倍加密网格估计收敛阶、极限值及最细网格剩余误差。"""
    coarse, medium, fine = values[-3:]
    ratio = abs((coarse - medium) / (medium - fine))
    order = float(np.log2(ratio))
    extrapolated = fine + (fine - medium) / (2.0**order - 1.0)
    return {
        "observed_order": order,
        "extrapolated_time_h": extrapolated,
        "estimated_641_error_s": abs(fine - extrapolated) * 3600.0,
    }


def main() -> None:
    environment = load_environment_xlsx(ROOT / "附件1.xlsx")
    radius_time, radius_m = load_radius_xlsx(ROOT / "附件2.xlsx")
    radius_fn = lambda t: np.interp(np.asarray(t, dtype=float), radius_time, radius_m)

    with (RESULTS / "grid_convergence.csv").open(encoding="utf-8-sig") as handle:
        grid = list(csv.DictReader(handle))
    nodes = [int(row["nodes"]) for row in grid]
    q3_grid = [float(row["problem3_time_h"]) for row in grid]
    q4_grid = [float(row["problem4_time_h"]) for row in grid]

    # 同一 321 节点网格下只改变 BDF 最大步长；30 s 作为本组参考。
    temporal: dict[str, dict[str, float]] = {"problem3": {}, "problem4": {}}
    for step in (120.0, 60.0, 30.0):
        temporal["problem3"][f"max_step_{int(step)}s_time_h"] = (
            event_time(environment, law_problem23(), max_step_s=step) / 3600.0
        )
        temporal["problem4"][f"max_step_{int(step)}s_time_h"] = (
            event_time(environment, law_problem4(), radius=radius_fn, max_step_s=step) / 3600.0
        )
    for item in temporal.values():
        reference = item["max_step_30s_time_h"]
        item["max_step_60s_error_vs_30s_s"] = abs(item["max_step_60s_time_h"] - reference) * 3600.0
        item["max_step_120s_error_vs_30s_s"] = abs(item["max_step_120s_time_h"] - reference) * 3600.0

    # 用稳定平台的样本标准差构造观测范围内的一阶敏感性包络；不是概率置信区间。
    stable_mask = environment.time_s >= environment.stable_start_s
    temperature_std = float(np.std(environment.temperature_c[stable_mask], ddof=1))
    moisture_std = float(np.std(environment.moisture[stable_mask], ddof=1))
    favorable = replace(
        environment,
        stable_temperature_c=environment.stable_temperature_c + temperature_std,
        stable_moisture=environment.stable_moisture - moisture_std,
    )
    adverse = replace(
        environment,
        stable_temperature_c=environment.stable_temperature_c - temperature_std,
        stable_moisture=environment.stable_moisture + moisture_std,
    )
    sensitivity = {
        "stable_temperature_std_c": temperature_std,
        "stable_moisture_std_kg_per_kg": moisture_std,
        "note": "mean±one sample standard deviation envelope; deterministic sensitivity, not confidence interval",
        "problem3": {
            "favorable_time_h": event_time(favorable, law_problem23()) / 3600.0,
            "baseline_time_h": q3_grid[nodes.index(ANALYSIS_NODES)],
            "adverse_time_h": event_time(adverse, law_problem23()) / 3600.0,
        },
        "problem4": {
            "favorable_time_h": event_time(favorable, law_problem4(), radius=radius_fn) / 3600.0,
            "baseline_time_h": q4_grid[nodes.index(ANALYSIS_NODES)],
            "adverse_time_h": event_time(adverse, law_problem4(), radius=radius_fn) / 3600.0,
        },
    }
    for item in (sensitivity["problem3"], sensitivity["problem4"]):
        item["half_range_s"] = 0.5 * (item["adverse_time_h"] - item["favorable_time_h"]) * 3600.0

    payload = {
        "analysis_nodes": ANALYSIS_NODES,
        "spatial": {
            "nodes": nodes,
            "problem3_time_h": q3_grid,
            "problem4_time_h": q4_grid,
            "problem3_abs_error_vs_641_s": [abs(v - q3_grid[-1]) * 3600.0 for v in q3_grid],
            "problem4_abs_error_vs_641_s": [abs(v - q4_grid[-1]) * 3600.0 for v in q4_grid],
            "problem3_richardson": richardson(q3_grid),
            "problem4_richardson": richardson(q4_grid),
        },
        "temporal": temporal,
        "environment_sensitivity": sensitivity,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "error_analysis.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("误差分析完成")
    print(
        f"空间 Richardson 估计：Q3 p={payload['spatial']['problem3_richardson']['observed_order']:.3f}, "
        f"641 节点剩余误差≈{payload['spatial']['problem3_richardson']['estimated_641_error_s']:.2f} s"
    )
    print(
        f"空间 Richardson 估计：Q4 p={payload['spatial']['problem4_richardson']['observed_order']:.3f}, "
        f"641 节点剩余误差≈{payload['spatial']['problem4_richardson']['estimated_641_error_s']:.2f} s"
    )
    print(
        f"BDF 60 s vs 30 s：Q3 {temporal['problem3']['max_step_60s_error_vs_30s_s']:.4f} s, "
        f"Q4 {temporal['problem4']['max_step_60s_error_vs_30s_s']:.4f} s"
    )


if __name__ == "__main__":
    main()
