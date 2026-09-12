"""温度 × 径向尺度联合敏感度：真实 PDE 求解、3D 曲面和交互效应图。

沿用 rf-tpe-surface 模板的曲面/色条布局，但不使用其模拟数据或 TPE 算法。
默认 13×17×2=442 个联合工况，另有 18 个网格间检查点。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl_surface"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
from functools import lru_cache
import hashlib
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from scipy.interpolate import PchipInterpolator

from model import law_problem23, law_problem4, load_environment_xlsx, simulate
from problem_utils import load_radius_function

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OFFSETS = np.linspace(-3, 3, 13)
SCALES = np.round(np.linspace(0.8, 1.2, 17), 3)
NODES = 321
SOURCES = ("src/model.py", "src/problem_utils.py", "附件1.xlsx", "附件2.xlsx")


def hashes() -> dict:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def source_matches(metadata: dict) -> bool:
    """只接受原始来源或经过逐工况复算确认的完整来源指纹。"""
    current = hashes()
    return current == metadata["source_sha256"] or current in metadata.get(
        "verified_compatible_source_sha256", [])


@lru_cache(maxsize=1)
def inputs():
    """每个计算进程只读取一次附件。"""
    return load_environment_xlsx(ROOT / "附件1.xlsx"), load_radius_function()[2]


def solve_case(case: tuple) -> dict:
    problem, offset, scale, role = case
    environment, base_radius = inputs()
    changed = replace(environment, stable_temperature_c=environment.stable_temperature_c + offset)
    radius = (lambda t: np.full_like(np.asarray(t, dtype=float), 0.02 * scale)) if problem == 3 else (
        lambda t: scale * np.asarray(base_radius(t)))
    result = simulate(changed, law_problem23() if problem == 3 else law_problem4(),
                      432000, radius=radius, nodes=NODES, stop_at_dry=True)
    if result.drying_time_s is None:
        raise RuntimeError(f"工况 {case} 在 5 天内没有达标")
    _, water = result.sample(np.array([result.drying_time_s]))
    residual = float(abs(np.max(water) - 0.15))
    if not np.all(np.isfinite(water)) or np.min(water) < 0 or residual > 2e-5:
        raise RuntimeError(f"工况 {case} 未通过事件或含水率检查")
    return dict(problem=problem, temperature_offset_c=offset, radius_scale=scale,
                drying_time_h=result.drying_time_s / 3600, event_residual=residual, role=role)


def interpolate(z, offsets, scales):
    """仅为绘图加密，先温度再半径做保形插值，不把插值点冒充新工况。"""
    along_temperature = PchipInterpolator(OFFSETS, z, axis=0)(offsets)
    return PchipInterpolator(SCALES, along_temperature, axis=1)(scales)


def matrices(rows):
    result = {}
    for q in (3, 4):
        grid = [r for r in rows if r["problem"] == q and r["role"] == "grid"]
        lookup = {(r["temperature_offset_c"], r["radius_scale"]): r["drying_time_h"] for r in grid}
        if len(grid) != 221 or len(lookup) != 221:
            raise ValueError(f"问题 {q} 联合网格缺失或重复")
        z = np.array([[lookup[(float(t), float(s))] for s in SCALES] for t in OFFSETS])
        if not np.all(np.isfinite(z)) or np.any(z <= 0):
            raise ValueError("曲面含非法烘干时长")
        result[q] = z
    return result


def validate(rows):
    grids = matrices(rows)
    oat = json.loads((RESULTS / "sensitivity_summary.json").read_text(encoding="utf-8"))
    with (RESULTS / "sensitivity_sweep.csv").open(encoding="utf-8-sig", newline="") as handle:
        old = list(csv.DictReader(handle))
    record = {"nodes": NODES, "grid_shape_per_problem": [13, 17], "grid_cases": 442,
              "validation_cases": 18, "source_sha256": hashes(), "problems": {},
              "temperature_offsets_c": OFFSETS.tolist(), "radius_scales": SCALES.tolist(),
              "stable_temperature_baseline_c": inputs()[0].stable_temperature_c,
              "temperature_scope": "only boundary after 14400 s; measured segment unchanged",
              "radius_scope": "scale fixed R for Q3 and entire attachment R(t) for Q4",
              "interpolation": "PCHIP along temperature then radius; rendering only",
              "probe_error_limit_percent": 0.05,
              "interaction_definition": "t(dT,s)-t(dT,1)-t(0,s)+t(0,1), hours",
              "statistical_claim": "deterministic two-factor sweep, not confidence intervals or global variance decomposition"}
    for q, z in grids.items():
        baseline = float(z[6, 8])
        if abs(baseline - oat["baseline_drying_time_h"][f"problem{q}"]) > 1 / 3600:
            raise ValueError("联合敏感度基准与已有 321 节点基准不一致")
        # 两条穿过基准点的截线必须退化为已有单因素分析。
        for row in old:
            if row["problem"] != f"problem{q}":
                continue
            p, value = row["parameter"], float(row["value"])
            if p == "stable_temperature":
                estimate = z[np.argmin(abs(OFFSETS - value)), 8]
            elif p == "radius_scale":
                estimate = z[6, np.argmin(abs(SCALES - value))]
            else:
                continue
            if abs(estimate - float(row["drying_time_h"])) > 1 / 3600:
                raise ValueError("联合扫描轴截线与已有单因素扫描不一致")
        probes = [r for r in rows if r["problem"] == q and r["role"] == "validation"]
        if len(probes) != 9:
            raise ValueError("网格间验证点不完整")
        errors = [abs(float(interpolate(z, [r["temperature_offset_c"]], [r["radius_scale"]])[0, 0])
                      / r["drying_time_h"] - 1) * 100 for r in probes]
        if max(errors) > 0.05:
            raise ValueError("绘图插值的网格间误差超过 0.05%，应加密真实工况")
        if not np.all(np.diff(z, axis=0) < 0) or not np.all(np.diff(z, axis=1) > 0):
            raise ValueError("真实网格未满足本扫描区间预期单调性，请检查工况")
        interaction = z - z[:, [8]] - z[[6], :] + baseline
        record["problems"][str(q)] = dict(
            baseline_h=baseline, min_h=float(z.min()), max_h=float(z.max()),
            max_interpolation_probe_error_percent=max(errors),
            max_abs_interaction_h=float(abs(interaction).max()),
            event_residual_max=max(r["event_residual"] for r in rows if r["problem"] == q))
    return record


def configure():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.labelsize": 11,
                         "text.color": "#263748", "axes.labelcolor": "#263748",
                         "svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "pa-joint-sensitivity",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        metadata = {"Date": None} if suffix == "svg" else ({"CreationDate": None, "ModDate": None} if suffix == "pdf" else {})
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=260, bbox_inches="tight", pad_inches=0.24, metadata=metadata)
    plt.close(fig)


def draw(rows):
    configure()
    z_by_q = matrices(rows)
    temperature = inputs()[0].stable_temperature_c
    fine_t, fine_s = np.linspace(-3, 3, 121), np.linspace(.8, 1.2, 121)
    x, y = np.meshgrid(temperature + fine_t, fine_s, indexing="ij")
    norm = Normalize(30, 95)
    fig = plt.figure(figsize=(15.2, 6.7))
    for panel, q in enumerate((3, 4)):
        ax = fig.add_axes([0.025 + .45 * panel, .10, .42, .83], projection="3d", computed_zorder=False)
        z = z_by_q[q]
        dense = interpolate(z, fine_t, fine_s)
        ax.plot_surface(x, y, dense, cmap="coolwarm", norm=norm, linewidth=0, antialiased=False,
                        rcount=121, ccount=121, shade=False, alpha=1., zorder=2)
        ax.contour(x, y, dense, levels=np.arange(35, 96, 10), zdir="z", offset=25,
                   cmap="coolwarm", norm=norm, linewidths=.85, alpha=.9, zorder=1)
        # 两条黑色基准截线对应既有单因素扫描；空心点标记未扰动基准。
        ax.plot(np.full(17, temperature), SCALES, z[6, :] + .06, color="#263748", lw=1.05, alpha=.7, zorder=3)
        ax.plot(temperature + OFFSETS, np.ones(13), z[:, 8] + .06, color="#263748", lw=1.05, alpha=.7, zorder=3)
        ax.scatter([temperature], [1.], [z[6, 8] + .15], s=48, facecolor="white", edgecolor="#263748", depthshade=False, zorder=10)
        ax.text2D(.10, .95, f"({chr(97+panel)}) 问题 {q}" + (" · 固定半径" if q == 3 else " · 收缩半径"), transform=ax.transAxes, fontsize=12)
        ax.text2D(.10, .89, f"基准 {z[6, 8]:.2f} h", transform=ax.transAxes, fontsize=10, color="#68788a")
        ax.set_xlabel("稳定阶段温度 / °C", labelpad=13)
        ax.set_ylabel("径向尺度倍数", labelpad=12)
        # 3D 原生 z 标签在紧边界导出时可能被裁切，改用固定画布文本。
        fig.text(.015 + .45 * panel, .50, "烘干时间 / h", rotation=90, va="center", fontsize=11)
        ax.set(xlim=(temperature-3, temperature+3), ylim=(.8, 1.2), zlim=(25, 95),
               xticks=[47, 49, 51, 53], yticks=[.8, .9, 1., 1.1, 1.2], zticks=[30, 50, 70, 90])
        ax.view_init(elev=27, azim=-132)
        ax.set_box_aspect((1.22, 1.12, .78))
        ax.tick_params(labelsize=9, pad=1)
        ax.tick_params(axis="x", pad=5)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor("#dbe2e9")
            axis._axinfo["grid"].update(color=(.75, .8, .85, .45), linewidth=.55)
    cax = fig.add_axes([.932, .26, .014, .46])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), cax=cax, ticks=[30, 45, 60, 75, 90])
    cb.set_label("烘干时间 / h", labelpad=12)
    cb.outline.set_visible(False)
    fig.text(.08, .045, "每问 13 × 17 个真实联合工况   ·   黑线：单因素截线   ·   白点：未扰动基准", fontsize=10, color="#68788a")
    save(fig, "16_joint_sensitivity_surface")

    # 分离非加和交互效应；不是 Q3 与 Q4 的差值，也不是统计误差。
    interactions = {q: z - z[:, [8]] - z[[6], :] + z[6, 8] for q, z in z_by_q.items()}
    limit = np.ceil(max(abs(z).max() for z in interactions.values()) * 2) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    levels = np.linspace(-limit, limit, 21)
    for panel, (q, ax) in enumerate(zip((3, 4), axes)):
        dense = interpolate(interactions[q], fine_t, fine_s)
        contour = ax.contourf(x, y, dense, levels=levels, cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit))
        lines = ax.contour(x, y, dense, levels=[-2, -1, -.5, .5, 1, 2], colors="#485a6b", linewidths=.55, alpha=.85)
        ax.clabel(lines, fmt="%g", fontsize=8)
        ax.axvline(temperature, color="#5d6d7c", ls="--", lw=.8)
        ax.axhline(1., color="#5d6d7c", ls="--", lw=.8)
        ax.scatter([temperature], [1.], s=30, c="white", edgecolors="#263748", zorder=3)
        ax.set_xlabel("稳定阶段温度 / °C")
        ax.set_ylabel("径向尺度倍数")
        ax.set_title(f"({chr(97+panel)}) 问题 {q}", loc="left", fontsize=12, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
    cb = fig.colorbar(contour, ax=axes, shrink=.86, pad=.025, ticks=np.arange(-limit, limit + .1, 1))
    cb.set_label("非加和交互效应 / h")
    cb.outline.set_visible(False)
    save(fig, "17_joint_sensitivity_interaction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="只核对已存曲面数据及其来源，不求解或绘图")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须至少为 1")
    path = RESULTS / "joint_sensitivity_sweep.csv"
    meta_path = RESULTS / "joint_sensitivity_summary.json"
    if args.plot_only or args.check_only:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not source_matches(metadata):
            raise ValueError("模型或附件已变化，请重新求解，不能复用旧曲面")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [{k: (v if k == "role" else int(v) if k == "problem" else float(v)) for k, v in row.items()} for row in csv.DictReader(handle)]
    else:
        source_before = hashes()
        cases = [(q, float(t), float(s), "grid") for q in (3, 4) for t in OFFSETS for s in SCALES]
        cases += [(q, t, s, "validation") for q in (3, 4) for t in (-2.75, .25, 2.75) for s in (.8125, 1.0125, 1.1875)]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(solve_case, cases, chunksize=1):
                rows.append(row)
                if len(rows) % 40 == 0 or len(rows) == len(cases):
                    print(f"已完成 {len(rows)}/{len(cases)} 个真实工况", flush=True)
        if hashes() != source_before:
            raise RuntimeError("计算期间模型或附件发生变化，未发布混合批次结果")
        metadata = validate(rows)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checked = validate(rows)
    # 复用缓存时保留真实生成来源，不能把当前核对来源冒充原始生成来源。
    checked["source_sha256"] = metadata["source_sha256"]
    metadata.update(checked)
    if not args.check_only:
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        draw(rows)
    print(json.dumps(metadata["problems"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
