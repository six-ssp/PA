"""问题 3、4 的径向网格无关性分析。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from model import Environment, SimulationResult, law_problem23, law_problem4, simulate


GRID_NODES = (81, 161, 321, 641)


def run_grid_convergence(
    environment: Environment,
    shrinking_radius: Callable[[float | np.ndarray], np.ndarray],
    output_csv: str | Path,
    *,
    end_time_s: float = 5.0 * 24.0 * 3600.0,
) -> tuple[list[dict[str, float | int]], SimulationResult, SimulationResult]:
    """逐网格计算烘干事件时间，并返回 641 节点的两份事件解。"""
    rows: list[dict[str, float | int]] = []
    final_problem3: SimulationResult | None = None
    final_problem4: SimulationResult | None = None

    print("\n网格无关性分析")
    print(f"{'nodes':>7} {'problem3_time_h':>18} {'problem4_time_h':>18}")
    for nodes in GRID_NODES:
        problem3 = simulate(
            environment,
            law_problem23(),
            end_time_s,
            nodes=nodes,
            stop_at_dry=True,
        )
        problem4 = simulate(
            environment,
            law_problem4(),
            end_time_s,
            radius=shrinking_radius,
            nodes=nodes,
            stop_at_dry=True,
        )
        if problem3.drying_time_s is None or problem4.drying_time_s is None:
            raise RuntimeError(f"{nodes} 节点网格在 5 天内未达到烘干阈值")

        row: dict[str, float | int] = {
            "nodes": nodes,
            "problem3_time_h": problem3.drying_time_s / 3600.0,
            "problem4_time_h": problem4.drying_time_s / 3600.0,
        }
        rows.append(row)
        print(f"{nodes:7d} {row['problem3_time_h']:18.6f} {row['problem4_time_h']:18.6f}")
        if nodes == GRID_NODES[-1]:
            final_problem3 = problem3
            final_problem4 = problem4

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["nodes", "problem3_time_h", "problem4_time_h"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "nodes": row["nodes"],
                "problem3_time_h": f"{float(row['problem3_time_h']):.9f}",
                "problem4_time_h": f"{float(row['problem4_time_h']):.9f}",
            })

    assert final_problem3 is not None and final_problem4 is not None
    return rows, final_problem3, final_problem4
