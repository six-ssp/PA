"""四问入口共用的输出格式与数值结果检查工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from model import Environment, SimulationResult, load_radius_xlsx


ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE = ROOT / "intermediate"
RESULTS = ROOT / "results"
SHORT_NODES = 321
FINAL_NODES = 641
MAX_DRYING_TIME_S = 5.0 * 24.0 * 3600.0
CONTROL_MAX_TIME_S = 10.0 * 24.0 * 3600.0
OUTPUT_DISTANCES_CM = np.arange(0.0, 2.0 + 0.05, 0.1)

RadiusFunction = Callable[[np.ndarray | float], np.ndarray | float]


def rounded_matrix(values: np.ndarray) -> list[list[float | None]]:
    """按题目要求保留四位小数；NaN 表示固定坐标已在药材外部。"""
    rounded = np.round(values.astype(float), 4)
    return [[None if np.isnan(value) else float(value) for value in row] for row in rounded]


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
    """以紧凑 UTF-8 JSON 写入 Excel 导出程序使用的中间目录。"""
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    with (INTERMEDIATE / name).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def load_radius_function(
    root: Path = ROOT,
) -> tuple[np.ndarray, np.ndarray, RadiusFunction]:
    """读取附件 2，并构造区间外自动保持端点值的线性插值函数。"""
    radius_time, radius_m = load_radius_xlsx(root / "附件2.xlsx")

    def radius_fn(query_time: np.ndarray | float) -> np.ndarray | float:
        return np.interp(np.asarray(query_time, dtype=float), radius_time, radius_m)

    return radius_time, radius_m, radius_fn


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
    radius_fn: RadiusFunction,
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


def print_environment(environment: Environment) -> None:
    """打印附件 1 稳定段自动计算出的环境边界。"""
    print(
        f"14400 s 后环境边界（{environment.stable_start_s:.0f}~"
        f"{environment.time_s[-1]:.0f} s 平均）: "
        f"T_const={environment.stable_temperature_c:.8f} °C, "
        f"C_const={environment.stable_moisture:.8f} kg/kg"
    )
