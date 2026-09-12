"""问题3、4的参数敏感度分析，并生成论文用图表。

正式四问参数保持不变。本脚本只在基准值附近改变一个因素，比较连续烘干
结束时间，属于确定性单因素敏感度分析，不代表统计置信区间。
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from model import Environment, MaterialLaw, law_problem23, law_problem4, load_environment_xlsx, simulate
from problem_utils import MAX_DRYING_TIME_S, ROOT, load_radius_function


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ANALYSIS_NODES = 321

# 温度每 0.5 °C 取一点；比例参数每 2.5% 取一点，兼顾曲线平滑和计算时间。
TEMPERATURE_OFFSETS_C = np.linspace(-3.0, 3.0, 13)
SCALE_FACTORS = np.round(np.linspace(0.8, 1.2, 17), 3)

PARAMETER_LABELS = {
    "stable_temperature": "稳定温度",
    "boundary_time_scale": "边界时间尺度",
    "radius_scale": "径向尺度",
    "stable_moisture_scale": "环境含水率",
    "diffusivity_scale": "扩散系数",
    "axial_length_scale": "轴向长度",
}

COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "cyan": "#36B5C5",
    "teal": "#1B998B",
    "gold": "#E6A23C",
    "orange": "#E76F51",
    "red": "#C44536",
    "purple": "#7E57C2",
    "ink": "#25313C",
    "muted": "#697786",
    "grid": "#DCE3EA",
    "paper": "#F7F9FC",
}


def configure_style() -> None:
    """设置与仓库现有论文图一致的字体和配色。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": COLORS["paper"],
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["ink"],
        "axes.titlecolor": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "text.color": COLORS["ink"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.72,
        "axes.titleweight": "bold",
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 8.2,
        "lines.linewidth": 2.0,
        "savefig.facecolor": COLORS["paper"],
        "savefig.bbox": "tight",
    })


def scaled_time_environment(environment: Environment, factor: float) -> Environment:
    """伸缩附件1的时间轴，温度和含水率观测值本身不变。"""
    return replace(
        environment,
        time_s=environment.time_s * factor,
        stable_start_s=environment.stable_start_s * factor,
    )


def scaled_diffusivity(material: MaterialLaw, factor: float) -> MaterialLaw:
    """只缩放水分扩散系数的整体量级，保留其温度、含水率函数形状。"""
    return MaterialLaw(
        density=material.density,
        heat_capacity=material.heat_capacity,
        conductivity=material.conductivity,
        diffusivity=lambda concentration, temperature_k: factor
        * material.diffusivity(concentration, temperature_k),
    )


def drying_time_h(
    environment: Environment,
    material: MaterialLaw,
    *,
    radius=None,
) -> float:
    """返回连续事件检测得到的烘干结束时间（小时）。"""
    result = simulate(
        environment,
        material,
        MAX_DRYING_TIME_S,
        radius=radius,
        nodes=ANALYSIS_NODES,
        stop_at_dry=True,
    )
    if result.drying_time_s is None:
        raise RuntimeError("敏感度工况在5天内未达到含水率阈值0.15")
    return float(result.drying_time_s / 3600.0)


def add_row(
    rows: list[dict],
    problem: str,
    parameter: str,
    value: float,
    value_unit: str,
    time_h: float,
    baseline_h: float,
) -> None:
    """把一个工况写成便于复算和绘图的长表记录。"""
    rows.append({
        "problem": problem,
        "parameter": parameter,
        "parameter_label": PARAMETER_LABELS[parameter],
        "value": float(value),
        "value_unit": value_unit,
        "drying_time_h": float(time_h),
        "change_from_baseline_percent": float(100.0 * (time_h - baseline_h) / baseline_h),
    })


def run_sweeps() -> tuple[list[dict], dict[str, float]]:
    """依次执行温度、时间尺度、半径、湿度和扩散系数单因素扫描。"""
    environment = load_environment_xlsx(ROOT / "附件1.xlsx")
    _, _, radius_base = load_radius_function()
    materials = {"problem3": law_problem23(), "problem4": law_problem4()}
    radii = {
        "problem3": lambda time: np.full_like(np.asarray(time, dtype=float), 0.02),
        "problem4": radius_base,
    }

    baseline = {
        problem: drying_time_h(environment, materials[problem], radius=radii[problem])
        for problem in materials
    }
    rows: list[dict] = []

    for problem in materials:
        material = materials[problem]
        radius_fn = radii[problem]

        # 稳定阶段温度只在14400 s后的外推边界中改变，实测段保持原值。
        for offset in TEMPERATURE_OFFSETS_C:
            modified = replace(
                environment,
                stable_temperature_c=environment.stable_temperature_c + float(offset),
            )
            time_h = baseline[problem] if offset == 0.0 else drying_time_h(
                modified, material, radius=radius_fn,
            )
            add_row(rows, problem, "stable_temperature", offset, "degC_offset", time_h, baseline[problem])

        for factor in SCALE_FACTORS:
            # factor=1时复用基准结果，避免重复求解带来无意义的计算开销。
            time_environment = scaled_time_environment(environment, float(factor))
            time_h = baseline[problem] if factor == 1.0 else drying_time_h(
                time_environment, material, radius=radius_fn,
            )
            add_row(rows, problem, "boundary_time_scale", factor, "ratio", time_h, baseline[problem])

            scaled_radius = lambda time, f=float(factor), r=radius_fn: f * np.asarray(r(time), dtype=float)
            time_h = baseline[problem] if factor == 1.0 else drying_time_h(
                environment, material, radius=scaled_radius,
            )
            add_row(rows, problem, "radius_scale", factor, "ratio", time_h, baseline[problem])

            moisture_environment = replace(
                environment,
                stable_moisture=environment.stable_moisture * float(factor),
            )
            time_h = baseline[problem] if factor == 1.0 else drying_time_h(
                moisture_environment, material, radius=radius_fn,
            )
            add_row(rows, problem, "stable_moisture_scale", factor, "ratio", time_h, baseline[problem])

            material_scaled = scaled_diffusivity(material, float(factor))
            time_h = baseline[problem] if factor == 1.0 else drying_time_h(
                environment, material_scaled, radius=radius_fn,
            )
            add_row(rows, problem, "diffusivity_scale", factor, "ratio", time_h, baseline[problem])

            # 一维长圆柱忽略端部效应，轴向长度不出现在控制方程中。
            add_row(
                rows, problem, "axial_length_scale", factor, "ratio",
                baseline[problem], baseline[problem],
            )

    return rows, baseline


def subset(rows: list[dict], problem: str, parameter: str) -> list[dict]:
    """从长表中取出指定问题和参数，并按参数值排序。"""
    return sorted(
        (row for row in rows if row["problem"] == problem and row["parameter"] == parameter),
        key=lambda row: row["value"],
    )


def local_sensitivity(rows: list[dict], baseline: dict[str, float]) -> dict:
    """计算基准附近的中心差分灵敏度和标准扰动影响。"""
    output: dict[str, dict] = {}
    for problem, baseline_h in baseline.items():
        output[problem] = {}
        for parameter in PARAMETER_LABELS:
            records = subset(rows, problem, parameter)
            by_value = {round(row["value"], 8): row["drying_time_h"] for row in records}
            if parameter == "stable_temperature":
                low_value, high_value = -1.0, 1.0
                low_standard, high_standard = -2.0, 2.0
                derivative = (by_value[high_value] - by_value[low_value]) / 2.0
                # 用稳定边界的绝对温度定义无量纲弹性，避免摄氏度零点任意性。
                temperature_k = load_environment_xlsx(ROOT / "附件1.xlsx").stable_temperature_c + 273.15
                elasticity = derivative * temperature_k / baseline_h
            else:
                low_value, high_value = 0.9, 1.1
                low_standard, high_standard = low_value, high_value
                derivative = (by_value[high_value] - by_value[low_value]) / 0.2
                elasticity = derivative / baseline_h
            output[problem][parameter] = {
                "label": PARAMETER_LABELS[parameter],
                "local_derivative_h_per_unit": float(derivative),
                "dimensionless_elasticity": float(elasticity),
                "standard_low_change_percent": float(
                    100.0 * (by_value[low_standard] - baseline_h) / baseline_h
                ),
                "standard_high_change_percent": float(
                    100.0 * (by_value[high_standard] - baseline_h) / baseline_h
                ),
                "standard_perturbation": "±2 degC" if parameter == "stable_temperature" else "±10%",
            }
    return output


def save_outputs(rows: list[dict], baseline: dict[str, float], sensitivity: dict) -> None:
    """保存机器可读的完整扫描表和摘要。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "sensitivity_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "deterministic one-at-a-time sensitivity around the baseline",
        "analysis_nodes": ANALYSIS_NODES,
        "sweep_row_count": len(rows),
        "temperature_offset_points": TEMPERATURE_OFFSETS_C.tolist(),
        "scale_factor_points": SCALE_FACTORS.tolist(),
        "baseline_drying_time_h": baseline,
        "temperature_standard_perturbation": "stable-stage boundary temperature ±2 degC",
        "other_standard_perturbation": "parameter scale ±10%",
        "axial_length_note": "zero sensitivity because axial end effects are excluded from the 1D radial model",
        "local_sensitivity": sensitivity,
    }
    with (RESULTS / "sensitivity_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_saved_outputs() -> tuple[list[dict], dict[str, dict]]:
    """读取已有扫描结果，用于只调整图形而不重复数值求解。"""
    csv_path = RESULTS / "sensitivity_sweep.csv"
    summary_path = RESULTS / "sensitivity_summary.json"
    if not csv_path.exists() or not summary_path.exists():
        raise FileNotFoundError("缺少敏感度结果，请先不带 --plot-only 运行完整分析")
    with csv_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["value"] = float(row["value"])
        row["drying_time_h"] = float(row["drying_time_h"])
        row["change_from_baseline_percent"] = float(row["change_from_baseline_percent"])
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if len(rows) != int(summary["sweep_row_count"]):
        raise RuntimeError("敏感度 CSV 与摘要记录数不一致，请重新运行完整分析")
    return rows, summary["local_sensitivity"]


def clean_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(COLORS["grid"])
    axis.grid(grid, linestyle="-")
    axis.set_axisbelow(True)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0, 1.03, label, transform=axis.transAxes,
        fontsize=11.5, fontweight="bold", color=COLORS["navy"], va="bottom",
    )


def plot_sensitivity(rows: list[dict], sensitivity: dict) -> None:
    """绘制六个互不遮挡的参数响应与局部敏感度面板。"""
    configure_style()
    figure, axes = plt.subplots(3, 2, figsize=(14.2, 13.0), constrained_layout=True)

    for problem, color, marker, label in (
        ("problem3", COLORS["blue"], "o", "问题 3"),
        ("problem4", COLORS["orange"], "s", "问题 4"),
    ):
        temperature = subset(rows, problem, "stable_temperature")
        axes[0, 0].plot(
            [row["value"] for row in temperature],
            [row["drying_time_h"] for row in temperature],
            color=color, marker=marker, markersize=4.2, markevery=2, label=label,
        )
        radius = subset(rows, problem, "radius_scale")
        axes[0, 1].plot(
            [100.0 * (row["value"] - 1.0) for row in radius],
            [row["drying_time_h"] for row in radius],
            color=color, marker=marker, markersize=4.2, markevery=2, label=label,
        )

    axes[0, 0].axvline(0.0, color=COLORS["muted"], linestyle="--", linewidth=1.1)
    axes[0, 0].set(
        title="稳定阶段温度对结束时间的影响",
        xlabel="相对基准温度偏移 / °C",
        ylabel="烘干时间 / h",
    )
    axes[0, 0].legend()
    clean_axis(axes[0, 0])

    axes[0, 1].axvline(0.0, color=COLORS["muted"], linestyle="--", linewidth=1.1)
    axes[0, 1].set(
        title="径向尺度对结束时间的影响",
        xlabel="半径尺度变化 / %",
        ylabel="烘干时间 / h",
    )
    axes[0, 1].legend()
    clean_axis(axes[0, 1])

    # 扩散系数与低敏感边界量分开成图，避免数量级差异压扁曲线。
    for problem, linestyle, marker in (("problem3", "-", "o"), ("problem4", "--", "s")):
        records = subset(rows, problem, "diffusivity_scale")
        axes[1, 0].plot(
            [100.0 * (row["value"] - 1.0) for row in records],
            [row["change_from_baseline_percent"] for row in records],
            color=COLORS["red"] if problem == "problem3" else COLORS["purple"],
            linestyle=linestyle, marker=marker, markersize=4, markevery=2,
            label=f"扩散系数 · {'问题3' if problem == 'problem3' else '问题4'}",
        )
    axes[1, 0].axhline(0.0, color=COLORS["muted"], linewidth=1.0)
    axes[1, 0].axvline(0.0, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    axes[1, 0].set(
        title="扩散系数对结束时间的影响",
        xlabel="参数尺度变化 / %",
        ylabel="结束时间变化 / %",
    )
    axes[1, 0].legend(loc="upper right", ncol=2)
    clean_axis(axes[1, 0])

    for parameter, color, label in (
        ("boundary_time_scale", COLORS["purple"], "时间尺度"),
        ("stable_moisture_scale", COLORS["teal"], "环境含水率"),
    ):
        for problem, linestyle, marker in (("problem3", "-", "o"), ("problem4", "--", "s")):
            records = subset(rows, problem, parameter)
            axes[1, 1].plot(
                [100.0 * (row["value"] - 1.0) for row in records],
                [row["change_from_baseline_percent"] for row in records],
                color=color, linestyle=linestyle, marker=marker,
                linewidth=1.8, markersize=3.5, markevery=2,
                label=f"{label} · {'问题3' if problem == 'problem3' else '问题4'}",
            )
    axes[1, 1].axhline(0.0, color=COLORS["muted"], linewidth=1.0)
    axes[1, 1].axvline(0.0, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        title="边界历程与环境含水率的放大比较",
        xlabel="参数尺度变化 / %",
        ylabel="结束时间变化 / %",
        ylim=(-0.65, 0.75),
    )
    axes[1, 1].legend(loc="upper left", ncol=2)
    clean_axis(axes[1, 1])

    # 标准扰动区间直接展示非线性和正负扰动的不对称性。
    sensitivity_order = [
        "stable_temperature", "radius_scale", "diffusivity_scale",
        "stable_moisture_scale", "boundary_time_scale", "axial_length_scale",
    ]
    y = np.arange(len(sensitivity_order))
    for problem, offset, color, label in (
        ("problem3", -0.13, COLORS["blue"], "问题 3"),
        ("problem4", 0.13, COLORS["orange"], "问题 4"),
    ):
        low = np.array([
            sensitivity[problem][parameter]["standard_low_change_percent"]
            for parameter in sensitivity_order
        ])
        high = np.array([
            sensitivity[problem][parameter]["standard_high_change_percent"]
            for parameter in sensitivity_order
        ])
        axes[2, 0].hlines(y + offset, np.minimum(low, high), np.maximum(low, high), color=color, linewidth=2.7)
        axes[2, 0].scatter(low, y + offset, color="white", edgecolor=color, s=38, linewidth=1.5, zorder=3)
        axes[2, 0].scatter(high, y + offset, color=color, edgecolor="white", s=42, linewidth=0.8, zorder=3, label=label)
    axes[2, 0].axvline(0.0, color=COLORS["muted"], linewidth=1.0)
    axes[2, 0].set_yticks(y, [PARAMETER_LABELS[parameter] for parameter in sensitivity_order])
    axes[2, 0].invert_yaxis()
    axes[2, 0].set(
        title="标准扰动下的结束时间变化区间",
        xlabel="相对基准变化 / %",
    )
    axes[2, 0].text(
        0.01, -0.16, "温度：±2 °C；其余参数：±10%（空心点为低输入，实心点为高输入）",
        transform=axes[2, 0].transAxes, fontsize=8, color=COLORS["muted"], va="top",
    )
    axes[2, 0].legend(loc="lower right", ncol=2)
    clean_axis(axes[2, 0])

    parameter_order = sensitivity_order
    matrix = np.array([
        [sensitivity[problem][parameter]["dimensionless_elasticity"] for problem in ("problem3", "problem4")]
        for parameter in parameter_order
    ])
    limit = max(1.0, float(np.max(np.abs(matrix))))
    image = axes[2, 1].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[2, 1].set_xticks([0, 1], ["问题 3", "问题 4"])
    axes[2, 1].set_yticks(np.arange(len(parameter_order)), [PARAMETER_LABELS[p] for p in parameter_order])
    axes[2, 1].set_title("基准点无量纲局部敏感度")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axes[2, 1].text(
                column_index, row_index, f"{value:+.2f}",
                ha="center", va="center",
                color="white" if abs(value) > 0.55 * limit else COLORS["ink"],
                fontweight="bold", fontsize=8.5,
            )
    colorbar = figure.colorbar(image, ax=axes[2, 1], shrink=0.88, pad=0.025)
    colorbar.set_label("弹性：输入变化 1% 引起的时间变化 %")

    for letter, axis in zip("ABCDEF", axes.flat):
        panel_label(axis, letter)
    figure.suptitle("烘干结束时间的单因素敏感度分析", fontsize=15.5, fontweight="bold", color=COLORS["navy"])

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "06_parameter_sensitivity.png", dpi=220)
    svg_path = FIGURES / "06_parameter_sensitivity.svg"
    figure.savefig(svg_path)
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n"
    with svg_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(svg_text)
    plt.close(figure)


def main(*, plot_only: bool = False) -> None:
    if plot_only:
        rows, sensitivity = load_saved_outputs()
        plot_sensitivity(rows, sensitivity)
        print(f"已从 {len(rows)} 条现有记录重新生成敏感度图")
        return

    rows, baseline = run_sweeps()
    sensitivity = local_sensitivity(rows, baseline)
    save_outputs(rows, baseline, sensitivity)
    plot_sensitivity(rows, sensitivity)

    print("敏感度分析完成")
    print(f"321节点基准：问题3={baseline['problem3']:.6f} h，问题4={baseline['problem4']:.6f} h")
    for problem in ("problem3", "problem4"):
        ranked = sorted(
            sensitivity[problem].items(),
            key=lambda item: abs(item[1]["dimensionless_elasticity"]),
            reverse=True,
        )
        summary = "，".join(f"{value['label']} {value['dimensionless_elasticity']:+.3f}" for _, value in ranked)
        print(f"{problem} 局部弹性排序：{summary}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成问题3、4参数敏感度结果与论文图")
    parser.add_argument("--plot-only", action="store_true", help="读取已有结果，只重画图6")
    arguments = parser.parse_args()
    main(plot_only=arguments.plot_only)
