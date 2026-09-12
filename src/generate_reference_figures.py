"""用现有真实数值结果适配科研模板，生成图 18~22（不改变主模型）。

参考 paired-raincloud、grouped-circular-heatmap、prediction-marginal-grid
与 rf-tpe-surface 的视觉组织。所有数值来自 PDE 或已保存的扫描；随机数
只用于云雨图横向避让，不用于生成观测、响应、噪声或显著性。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl_reference"))

import argparse
import csv
import hashlib
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
from scipy.stats import gaussian_kde

from model import law_problem23, law_problem4, simulate
from sensitivity_surface import OFFSETS, SCALES, configure, hashes, inputs, interpolate, matrices, save, source_matches, validate

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
INK, TEAL, ORANGE = "#263748", "#247f91", "#d77855"
PARAMETERS = ["stable_temperature", "radius_scale", "diffusivity_scale", "stable_moisture_scale", "boundary_time_scale", "axial_length_scale"]
LABELS = ["稳定温度", "径向尺度", "扩散系数", "环境含水率", "边界时间尺度", "轴向长度"]


def read_joint():
    metadata = json.loads((RESULTS / "joint_sensitivity_summary.json").read_text(encoding="utf-8"))
    if not source_matches(metadata):
        raise ValueError("联合扫描来源与当前模型不一致，请先运行 sensitivity_surface.py")
    with (RESULTS / "joint_sensitivity_sweep.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [{k: (v if k == "role" else int(v) if k == "problem" else float(v)) for k, v in r.items()} for r in csv.DictReader(handle)]
    validate(rows)
    return matrices(rows)


def interaction(z):
    """相对于基准的非加和量；它与总烘干时间是不同的纵轴变量。"""
    return z - z[:, [8]] - z[[6], :] + z[6, 8]


def metrics(grids):
    t, s = np.meshgrid(OFFSETS, SCALES, indexing="ij")
    design = np.column_stack([np.ones(t.size), t.ravel(), s.ravel()])
    # 排除两个基准截线：这些点的加和近似按定义就精确，不计入近似误差。
    off_axis = (abs(t) > 1e-10) & (abs(s - 1) > 1e-10)
    result = {}
    for q, z in grids.items():
        plane = (design @ np.linalg.lstsq(design, z.ravel(), rcond=None)[0]).reshape(z.shape)
        residual = z - plane
        err = interaction(z)[off_axis]
        result[str(q)] = dict(
            plane_r2=float(1 - np.sum(residual**2) / np.sum((z - z.mean())**2)),
            plane_rmse_h=float(np.sqrt(np.mean(residual**2))),
            max_plane_deviation_h=float(abs(residual).max()),
            range_h=float(np.ptp(z)), off_axis_cases=int(off_axis.sum()),
            additive_rmse_h=float(np.sqrt(np.mean(err**2))), additive_mae_h=float(np.mean(abs(err))),
            additive_max_error_h=float(abs(err).max()))
    delta = (grids[3] - grids[4]).ravel()
    result["paired_cases"] = dict(count=int(delta.size), median_difference_h=float(np.median(delta)),
                                 min_difference_h=float(delta.min()), max_difference_h=float(delta.max()))
    return result


def axes3d(fig, panel, title, zlabel):
    ax = fig.add_axes([.025 + .45 * panel, .11, .42, .80], projection="3d", computed_zorder=False)
    ax.text2D(.12, .98, title, transform=ax.transAxes, fontsize=12)
    fig.text(.015 + .45 * panel, .51, zlabel, rotation=90, va="center", fontsize=11)
    ax.view_init(elev=29, azim=-133)
    ax.set_box_aspect((1.18, 1.12, .86))
    ax.tick_params(labelsize=9, pad=1)
    ax.tick_params(axis="x", pad=5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#dbe2e9")
        axis._axinfo["grid"].update(color=(.75, .8, .85, .45), linewidth=.55)
    return ax


def draw_interaction(grids):
    """保留真实交互的鞍形，不给总时间曲面添加人为峰谷。"""
    ft, fs = np.linspace(-3, 3, 101), np.linspace(.8, 1.2, 101)
    temperature = inputs()[0].stable_temperature_c
    x, y = np.meshgrid(temperature + ft, fs, indexing="ij")
    norm = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    fig = plt.figure(figsize=(15.2, 6.8))
    for panel, q in enumerate((3, 4)):
        ax = axes3d(fig, panel, f"({chr(97 + panel)}) 问题 {q} · 非加和交互", "交互效应 / h")
        z = interpolate(interaction(grids[q]), ft, fs)
        ax.plot_surface(x, y, z, cmap="RdBu_r", norm=norm, linewidth=0, antialiased=False, shade=False,
                        rcount=101, ccount=101, zorder=2)
        ax.contour(x, y, z, levels=[-2, -1, 0, 1, 2], offset=-3.3, zdir="z", cmap="RdBu_r", norm=norm, linewidths=.8, zorder=1)
        ax.plot(temperature + OFFSETS, np.ones(13), np.zeros(13), color=INK, lw=1.2, zorder=3)
        ax.plot(np.full(17, temperature), SCALES, np.zeros(17), color=INK, lw=1.2, zorder=3)
        ax.set(xlim=(temperature - 3, temperature + 3), ylim=(.8, 1.2), zlim=(-3.3, 3.3),
               xticks=[47, 49, 51, 53], yticks=[.8, .9, 1, 1.1, 1.2], zticks=[-3, -1.5, 0, 1.5, 3])
        ax.set_xlabel("稳定阶段温度 / °C", labelpad=13)
        ax.set_ylabel("径向尺度倍数", labelpad=12)
    cax = fig.add_axes([.938, .27, .013, .45])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.set_label("相对单因素加和近似的偏差 / h", labelpad=11)
    cb.outline.set_visible(False)
    fig.text(.09, .045, "纵轴为交互效应，并非总烘干时间；正值：比加和近似更慢，负值：更快。", fontsize=10, color="#647789")
    save(fig, "18_interaction_response_surface")


def half_cloud(ax, values, position, direction, color, width=.33):
    """半小提琴形状仅汇总有限扫描点分布，不解释为随机样本密度。"""
    grid = np.linspace(values.min(), values.max(), 200)
    density = gaussian_kde(values)(grid)
    ax.fill_betweenx(grid, position, position + direction * width * density / density.max(), color=color, alpha=.32, lw=.9, edgecolor=color)


def draw_raincloud(grids):
    a, b = grids[3].ravel(), grids[4].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6), gridspec_kw={"width_ratios": [1.65, 1]}, layout="constrained")
    ax, delta_ax = axes
    jitter = np.random.default_rng(20260912).uniform(-.045, .045, a.size)
    half_cloud(ax, a, .84, -1, TEAL)
    half_cloud(ax, b, 2.16, 1, ORANGE)
    for i in range(a.size):
        ax.plot([1.07 + jitter[i], 1.93 + jitter[i]], [a[i], b[i]], color="#7c8c9a", lw=.5, alpha=.12, zorder=1)
    for values, x, point_x, color in [(a, .91, 1.07, TEAL), (b, 2.09, 1.93, ORANGE)]:
        ax.scatter(point_x + jitter, values, s=10, color=color, alpha=.62, linewidths=0, zorder=3)
        ax.boxplot(values, positions=[x], widths=.10, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="white", edgecolor=color), medianprops=dict(color=color, linewidth=1.8),
                   whiskerprops=dict(color=color), capprops=dict(color=color), zorder=4)
    ax.set(xlim=(.35, 2.65), xticks=[.98, 2.02], xticklabels=["问题 3", "问题 4"], ylabel="烘干时间 / h", ylim=(27, 96))
    ax.set_title("(a) 相同扰动条件逐点配对", loc="left", fontsize=12, pad=13)
    delta = a - b
    half_cloud(delta_ax, delta, .94, -1, "#77619e", width=.30)
    delta_ax.scatter(1.04 + jitter, delta, color="#77619e", s=15, alpha=.7, linewidths=0)
    delta_ax.boxplot(delta, positions=[1.18], widths=.10, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="white", edgecolor="#77619e"), medianprops=dict(color="#77619e"))
    delta_ax.axhline(np.median(delta), color="#77619e", ls="--", lw=.9)
    delta_ax.set(xlim=(.55, 1.55), xticks=[], ylabel="配对时间差：问题 3 − 问题 4 / h")
    delta_ax.set_title(f"(b) 中位时间差 {np.median(delta):.2f} h", loc="left", fontsize=12, pad=13)
    delta_ax.set_xlabel("221 对工况；两问差异包含物性变化", fontsize=9, labelpad=12)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e7ecf1", linewidth=.65)
        axis.set_axisbelow(True)
    save(fig, "19_paired_scenario_raincloud")


def draw_circular():
    with (RESULTS / "sensitivity_sweep.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # 原始响应跨越约 0.02% 至 40%，用明确标注的对称对数色标保留弱响应。
    norm = SymLogNorm(linthresh=.1, linscale=.75, vmin=-45, vmax=45, base=10)
    cmap = mpl.colormaps["RdBu_r"]
    fig = plt.figure(figsize=(12.5, 7.1))
    for panel, q in enumerate((3, 4)):
        ax = fig.add_axes([.025 + .49 * panel, .18, .45, .74], projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_axis_off()
        ax.set_ylim(0, 3.05)
        for index, (parameter, label) in enumerate(zip(PARAMETERS, LABELS)):
            group = sorted([r for r in rows if r["problem"] == f"problem{q}" and r["parameter"] == parameter], key=lambda r: float(r["value"]))
            n = len(group)
            if n != (13 if index == 0 else 17):
                raise ValueError("环形热图扫描记录不完整")
            theta = np.linspace(np.deg2rad(40), np.deg2rad(320), n)
            bottom = 2.5 - index * .23
            values = np.array([float(r["change_from_baseline_percent"]) for r in group])
            ax.bar(theta, np.full(n, .205), bottom=bottom, width=np.deg2rad(280 / (n - 1) * .92),
                   color=cmap(norm(values)), edgecolor="white", linewidth=.5)
            ax.text(np.deg2rad(27), bottom + .10, str(index + 1), ha="center", va="center", fontsize=10, color=INK)
        for value in [-1, -.5, 0, .5, 1]:
            angle = np.deg2rad(40 + (value + 1) * 140)
            ax.text(angle, 2.96, f"{value:+g}" if value else "0", ha="center", va="center", fontsize=10)
        ax.text(0, 0, f"问题 {q}\n\n" + "\n".join(f"{i+1}  {s}" for i, s in enumerate(LABELS)), ha="center", va="center", fontsize=10, linespacing=1.65)
        fig.text(.10 + .49 * panel, .94, f"({chr(97+panel)}) 六因素扫描响应", fontsize=12)
    cax = fig.add_axes([.27, .12, .46, .022])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-40, -10, -1, -.1, 0, .1, 1, 10, 40])
    cb.set_ticklabels(["−40", "−10", "−1", "−0.1", "0", "0.1", "1", "10", "40"])
    cb.set_label("烘干时间变化 / %（对称对数色标，±0.1% 内线性）", labelpad=7)
    cb.outline.set_visible(False)
    fig.text(.10, .025, "角度表示扫描位置 ξ：温度偏移 = 3ξ °C；其余参数倍数 = 1 + 0.2ξ。缺口不连接。", fontsize=10, color="#647789")
    save(fig, "20_grouped_sensitivity_rings")


def draw_marginal(grids, stats):
    fig = plt.figure(figsize=(12.8, 6.8))
    outer = fig.add_gridspec(1, 2, left=.075, right=.96, bottom=.17, top=.94, wspace=.30)
    delta, scale = np.meshgrid(OFFSETS, SCALES, indexing="ij")
    mask = (abs(delta) > 1e-10) & (abs(scale - 1) > 1e-10)
    for panel, q in enumerate((3, 4)):
        z = grids[q]
        actual = z[mask]
        estimate = (z - interaction(z))[mask]
        sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[panel], height_ratios=[.24, 1], width_ratios=[1, .22], hspace=.05, wspace=.05)
        top, ax, right = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 0]), fig.add_subplot(sub[1, 1])
        points = ax.scatter(actual, estimate, c=delta[mask], cmap="coolwarm", vmin=-3, vmax=3, s=25, alpha=.82, edgecolors="white", linewidths=.25)
        ax.plot([28, 96], [28, 96], color="#68798a", lw=.8, ls="--")
        ax.set(xlim=(28, 96), ylim=(28, 96), xlabel="联合 PDE 求解时间 / h", ylabel="单因素加和近似时间 / h")
        ax.text(.06, .93, f"RMSE = {stats[str(q)]['additive_rmse_h']:.3f} h\nMAE = {stats[str(q)]['additive_mae_h']:.3f} h\n192 个非基准轴工况", va="top", transform=ax.transAxes, fontsize=10)
        ax.grid(color="#edf0f4", lw=.6)
        top.hist(actual, bins=np.arange(28, 97, 4), color=TEAL, alpha=.26, edgecolor=TEAL, linewidth=.6, density=True)
        xx = np.linspace(actual.min(), actual.max(), 200)
        top.plot(xx, gaussian_kde(actual)(xx), color=TEAL, lw=1.2)
        top.set(xlim=(28, 96), xticks=[], yticks=[])
        top.set_title(f"({chr(97+panel)}) 问题 {q}", loc="left", fontsize=12, pad=10)
        right.hist(estimate, bins=np.arange(28, 97, 4), orientation="horizontal", color=ORANGE, alpha=.26, edgecolor=ORANGE, linewidth=.6, density=True)
        yy = np.linspace(estimate.min(), estimate.max(), 200)
        right.plot(gaussian_kde(estimate)(yy), yy, color=ORANGE, lw=1.2)
        right.set(ylim=(28, 96), xticks=[], yticks=[])
        for axis in (top, ax, right):
            axis.spines[["top", "right"]].set_visible(False)
    cax = fig.add_axes([.34, .052, .32, .018])
    cb = fig.colorbar(points, cax=cax, orientation="horizontal", ticks=[-3, -1.5, 0, 1.5, 3])
    cb.set_label("稳定温度偏移 / °C", labelpad=5)
    cb.outline.set_visible(False)
    save(fig, "21_additive_approximation_marginals")


def prepare_fields(reuse=False):
    path = RESULTS / "reference_moisture_fields.npz"
    meta_path = RESULTS / "reference_moisture_fields.json"
    if reuse:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not source_matches(meta) or meta["data_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("时空场缓存来源已变化，请重新求解")
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    environment, radius = inputs()
    source_before = hashes()
    summary = json.loads((ROOT / "intermediate/summary.json").read_text(encoding="utf-8"))
    # 前期变化快：按 sqrt(t) 均匀采样，加密起始段；不改变时间坐标本身。
    data = {"time_h": np.linspace(0, np.sqrt(24), 181)**2, "material_x": np.linspace(0, 1, 161)}
    events = {}
    for q in (3, 4):
        solution = simulate(environment, law_problem23() if q == 3 else law_problem4(), 432000,
                            radius=None if q == 3 else radius, nodes=641, stop_at_dry=True)
        if solution.drying_time_s is None or abs(solution.drying_time_s - summary[f"problem{q}_drying_time_s"]) > 1:
            raise ValueError("时空曲面求解与正式 641 节点事件不一致")
        _, field = solution.sample(data["time_h"] * 3600)
        data[f"problem{q}_moisture"] = field[:, ::4]
        if not np.all(np.isfinite(field)) or np.min(field) < 0 or not np.allclose(field[0], 2.55):
            raise ValueError("时空场初值或有限性检查失败")
        events[str(q)] = solution.drying_time_s
    if hashes() != source_before:
        raise ValueError("求解期间模型来源改变")
    np.savez_compressed(path, **data)
    meta_path.write_text(json.dumps(dict(nodes=641, source_sha256=hashes(), event_times_s=events,
                                        time_range_h=[0, 24], field_unit="kg/kg dry basis", coordinate="x=r/R(t)",
                                        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def draw_fields(data):
    fig = plt.figure(figsize=(15.2, 6.8))
    time, radius = np.meshgrid(data["time_h"], data["material_x"][::2], indexing="ij")
    norm = Normalize(0, 2.55)
    for panel, q in enumerate((3, 4)):
        ax = axes3d(fig, panel, f"({chr(97+panel)}) 问题 {q} · 前 24 h", "干基含水率 / (kg/kg)")
        z = data[f"problem{q}_moisture"][:, ::2]
        ax.plot_surface(time, radius, z, cmap="viridis", norm=norm, linewidth=0, antialiased=False,
                        shade=False, rcount=181, ccount=81, zorder=2)
        ax.contour(time, radius, z, levels=[.15, .5, 1, 1.5, 2], zdir="z", offset=-.12, cmap="viridis", norm=norm, linewidths=.8, zorder=1)
        for r_index in [0, -1]:
            ax.plot(data["time_h"], np.full(181, data["material_x"][r_index]), z[:, r_index], color=INK, lw=1.1, zorder=3)
        ax.set(xlim=(0, 24), ylim=(0, 1), zlim=(-.12, 2.7), xticks=[0, 6, 12, 18, 24], yticks=[0, .25, .5, .75, 1], zticks=[0, .5, 1, 1.5, 2, 2.5])
        ax.set_xlabel("时间 / h", labelpad=12)
        ax.set_ylabel("归一化半径 x = r/R(t)", labelpad=14)
        ax.view_init(elev=28, azim=-133)
    cax = fig.add_axes([.938, .27, .013, .45])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, ticks=[0, .5, 1, 1.5, 2, 2.5])
    cb.set_label("干基含水率 / (kg/kg)", labelpad=11)
    cb.outline.set_visible(False)
    fig.text(.09, .04, "641 节点真实数值解   ·   x = 0 为中心，x = 1 为实时表面   ·   同一时刻横截面并不均匀", fontsize=10, color="#647789")
    save(fig, "22_moisture_spacetime_surface")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="复用经过哈希核对的时空场，不重新求解")
    args = parser.parse_args()
    configure()
    FIGURES.mkdir(exist_ok=True)
    grids = read_joint()
    stats = metrics(grids)
    for function in (draw_interaction, draw_raincloud):
        function(grids)
        print(function.__name__ + " 完成", flush=True)
    draw_circular()
    draw_marginal(grids, stats)
    fields = prepare_fields(args.plot_only)
    draw_fields(fields)
    stats["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                              ("results/joint_sensitivity_sweep.csv", "results/sensitivity_sweep.csv", "results/reference_moisture_fields.npz")}
    stats["interpretation"] = "finite deterministic scenario grids; no independent experimental truth; raincloud KDE is descriptive only"
    (RESULTS / "reference_figure_diagnostics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
