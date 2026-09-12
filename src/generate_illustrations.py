"""生成论文中的模型示意图。

本脚本只负责概念性插图，不参与数值计算。所有图同时导出 PNG 和 SVG：
PNG 便于直接插入论文，SVG 便于后续在矢量软件中修改文字与配色。
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"

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
    "white": "#FFFFFF",
    "herb": "#E9B872",
    "herb_dark": "#C98643",
}


def configure_style() -> None:
    """配置与仓库现有数据图一致的字体和颜色风格。"""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "figure.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "text.color": COLORS["ink"],
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """同时保存高分辨率位图与可编辑矢量图。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{stem}.png"
    svg_path = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.12)

    # 统一 SVG 换行符，避免不同平台生成无意义的 Git 差异。
    svg_text = svg_path.read_text(encoding="utf-8")
    clean_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
    svg_path.write_text(clean_svg, encoding="utf-8", newline="\n")
    plt.close(fig)


def add_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = COLORS["white"],
    edgecolor: str = COLORS["grid"],
    radius: float = 0.018,
    linewidth: float = 1.2,
    zorder: int = 0,
) -> FancyBboxPatch:
    """在归一化画布中添加圆角信息卡片。"""
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(card)
    return card


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["navy"],
    linewidth: float = 1.8,
    mutation_scale: float = 14,
    connectionstyle: str = "arc3",
    zorder: int = 3,
) -> FancyArrowPatch:
    """添加统一样式的箭头。"""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def new_canvas(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    """创建 0~1 归一化坐标的无边框画布。"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def load_summary() -> dict:
    """读取主流程导出的正式摘要，避免插图中的结果数字被手工写死。"""
    summary_path = ROOT / "intermediate" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("缺少 intermediate/summary.json，请先运行 python src/run_all.py")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def physical_model_schematic() -> None:
    """绘制长圆柱近似和一维径向传递假设。"""
    fig, ax = new_canvas((14.2, 7.2))
    fig.suptitle(
        "药材长圆柱近似与径向传热—传质模型",
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.965,
    )
    ax.text(
        0.5,
        0.905,
        "将轴向长度远大于半径的药材视为轴对称圆柱，仅求解中部截面的径向变化",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    add_card(ax, 0.035, 0.12, 0.54, 0.72)
    add_card(ax, 0.605, 0.12, 0.36, 0.72)
    ax.text(0.065, 0.79, "物理对象：细长圆柱", fontsize=13, fontweight="bold")
    ax.text(0.635, 0.79, "计算截面：一维径向", fontsize=13, fontweight="bold")

    # 左侧长圆柱：矩形柱身配合椭圆端面形成简洁的三维效果。
    body_x, body_y, body_w, body_h = 0.145, 0.405, 0.315, 0.205
    ax.add_patch(
        Rectangle(
            (body_x, body_y),
            body_w,
            body_h,
            facecolor=COLORS["herb"],
            edgecolor=COLORS["herb_dark"],
            linewidth=1.8,
            zorder=2,
        )
    )
    ax.add_patch(
        Ellipse(
            (body_x, body_y + body_h / 2),
            0.075,
            body_h,
            facecolor="#D39A57",
            edgecolor=COLORS["herb_dark"],
            linewidth=1.8,
            zorder=4,
        )
    )
    ax.add_patch(
        Ellipse(
            (body_x + body_w, body_y + body_h / 2),
            0.075,
            body_h,
            facecolor="#F2C98F",
            edgecolor=COLORS["herb_dark"],
            linewidth=1.8,
            zorder=4,
        )
    )

    # 取轴向中部截面，虚线用于强调截面位置而不是额外边界。
    section_x = 0.315
    ax.add_patch(
        Ellipse(
            (section_x, body_y + body_h / 2),
            0.065,
            body_h,
            fill=False,
            edgecolor=COLORS["navy"],
            linewidth=1.5,
            linestyle=(0, (3, 2)),
            zorder=5,
        )
    )
    ax.annotate(
        "轴向中部截面",
        xy=(section_x, body_y + body_h + 0.01),
        xytext=(section_x, 0.695),
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=COLORS["navy"],
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["navy"],
            "lw": 1.3,
        },
    )

    # 表面对流换热和外部传质通量采用两种颜色区分。
    for x_pos, color in [
        (0.195, COLORS["orange"]),
        (0.275, COLORS["blue"]),
        (0.375, COLORS["orange"]),
        (0.435, COLORS["blue"]),
    ]:
        add_arrow(ax, (x_pos, 0.665), (x_pos, 0.62), color=color, linewidth=1.6)
        add_arrow(ax, (x_pos, 0.35), (x_pos, 0.395), color=color, linewidth=1.6)

    ax.text(
        0.302,
        0.745,
        r"干燥介质：$T_\infty(t),\ C_\infty(t)$",
        ha="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )
    ax.text(0.185, 0.33, r"热通量 $q_T$", ha="center", fontsize=10.5, color=COLORS["orange"])
    ax.text(0.425, 0.33, r"水分通量 $q_C$", ha="center", fontsize=10.5, color=COLORS["blue"])

    # 圆柱轴线和“长径比大”的假设提示。
    add_arrow(ax, (0.15, 0.245), (0.455, 0.245), color=COLORS["muted"], linewidth=1.4)
    ax.text(0.302, 0.215, r"轴向 $z$：忽略端部效应与轴向梯度", ha="center", fontsize=10.5)

    # 右侧截面使用同心环表示径向网格，并突出中心对称边界。
    center = (0.785, 0.505)
    outer_r = 0.145
    # 画布宽高比约为 2:1；横向半径需要缩短，屏幕上才显示为真正的圆。
    x_radius = outer_r / (14.2 / 7.2)
    for frac, lw, alpha in [(1.0, 2.1, 1.0), (0.67, 1.0, 0.75), (0.34, 1.0, 0.75)]:
        ax.add_patch(
            Ellipse(
                center,
                width=2 * x_radius * frac,
                height=2 * outer_r * frac,
                facecolor=COLORS["herb"] if frac == 1.0 else "none",
                edgecolor=COLORS["navy"] if frac == 1.0 else COLORS["grid"],
                linewidth=lw,
                alpha=alpha,
                zorder=1 if frac == 1.0 else 2,
            )
        )
    ax.plot(center[0], center[1], "o", color=COLORS["navy"], markersize=4.5, zorder=4)

    theta = np.deg2rad(36)
    radial_end = (
        center[0] + x_radius * np.cos(theta),
        center[1] + outer_r * np.sin(theta),
    )
    add_arrow(ax, center, radial_end, color=COLORS["navy"], linewidth=1.9)
    ax.text(0.855, 0.565, r"$r$", fontsize=13, color=COLORS["navy"])
    ax.text(0.837, 0.617, r"$R(t)$", fontsize=12.5, color=COLORS["navy"])

    for angle, color in zip(
        [0, 60, 120, 180, 240, 300],
        [COLORS["blue"], COLORS["orange"]] * 3,
    ):
        theta_i = np.deg2rad(angle)
        direction = np.array([x_radius * np.cos(theta_i), outer_r * np.sin(theta_i)])
        unit_display = direction / outer_r
        start = np.array(center) + direction + unit_display * 0.058
        end = np.array(center) + direction + unit_display * 0.008
        add_arrow(ax, tuple(start), tuple(end), color=color, linewidth=1.55, mutation_scale=12)

    ax.text(
        center[0],
        0.285,
        r"中心：$\partial T/\partial r=\partial C/\partial r=0$",
        ha="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    ax.text(
        center[0],
        0.205,
        r"表面：$-kT_r=h(T_s-T_\infty)$，$-DC_r=h_m(C_s-C_\infty)$",
        ha="center",
        fontsize=10.5,
        color=COLORS["ink"],
    )

    # 底部标签总结建模边界，便于论文读者快速识别假设。
    pill_specs = [
        (0.14, "长圆柱近似", COLORS["orange"]),
        (0.31, "轴对称", COLORS["teal"]),
        (0.44, "径向耦合", COLORS["blue"]),
        (0.75, "有限体积离散", COLORS["purple"]),
    ]
    for x_pos, text, color in pill_specs:
        ax.text(
            x_pos,
            0.075,
            text,
            ha="center",
            va="center",
            fontsize=10.2,
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": color + "16",
                "edgecolor": color + "55",
                "linewidth": 1.0,
            },
        )

    save_figure(fig, "07_physical_model_schematic")


def material_coordinate_mapping() -> None:
    """绘制收缩物理域到固定材料坐标域的映射。"""
    fig = plt.figure(figsize=(14.2, 6.8), facecolor=COLORS["paper"])
    fig.suptitle(
        "收缩物理域到固定材料坐标域的映射",
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        r"用 $x=r/R(t)$ 固定计算区间，使同一材料点在收缩过程中保持相同的 $x$ 坐标",
        ha="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    axes = [
        fig.add_axes((0.045, 0.22, 0.25, 0.58)),
        fig.add_axes((0.375, 0.22, 0.25, 0.58)),
        fig.add_axes((0.705, 0.22, 0.25, 0.58)),
    ]
    node_colors = [COLORS["navy"], COLORS["blue"], COLORS["cyan"], COLORS["gold"], COLORS["red"]]
    node_fracs = np.linspace(0.0, 1.0, len(node_colors))

    # 初始物理域：材料点沿半径均匀标记。
    ax0 = axes[0]
    ax0.set_aspect("equal")
    ax0.set_xlim(-1.25, 1.25)
    ax0.set_ylim(-1.25, 1.25)
    ax0.axis("off")
    ax0.set_title(r"初始物理域  $0\leq r\leq R_0$", fontsize=13, fontweight="bold", pad=10)
    ax0.add_patch(Circle((0, 0), 1, facecolor="#F5D7AD", edgecolor=COLORS["herb_dark"], linewidth=2.1))
    ax0.plot([0, 1], [0, 0], color=COLORS["muted"], linewidth=1.5)
    for frac, color in zip(node_fracs, node_colors):
        ax0.plot(frac, 0, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax0.annotate(
        r"$R_0$",
        xy=(1, 0),
        xytext=(0.57, 0.19),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["navy"], "lw": 1.5},
        color=COLORS["navy"],
        fontsize=12,
    )
    ax0.text(0, -1.16, "材料点随半径编号", ha="center", fontsize=10.5, color=COLORS["muted"])

    # 收缩后的物理域：虚线保留初始轮廓，实线表示当前半径。
    ax1 = axes[1]
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.25, 1.25)
    ax1.set_ylim(-1.25, 1.25)
    ax1.axis("off")
    ax1.set_title(r"收缩物理域  $0\leq r\leq R(t)$", fontsize=13, fontweight="bold", pad=10)
    ax1.add_patch(Circle((0, 0), 1, fill=False, edgecolor=COLORS["grid"], linewidth=1.5, linestyle=(0, (4, 3))))
    shrink = 0.68
    ax1.add_patch(Circle((0, 0), shrink, facecolor="#F5D7AD", edgecolor=COLORS["orange"], linewidth=2.3))
    ax1.plot([0, shrink], [0, 0], color=COLORS["muted"], linewidth=1.5)
    for frac, color in zip(node_fracs, node_colors):
        ax1.plot(shrink * frac, 0, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax1.annotate(
        r"$R(t)$",
        xy=(shrink, 0),
        xytext=(0.34, 0.19),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["orange"], "lw": 1.5},
        color=COLORS["orange"],
        fontsize=12,
    )
    ax1.text(0, -1.16, r"物理位置 $r=xR(t)$ 向内移动", ha="center", fontsize=10.5, color=COLORS["muted"])

    # 固定材料坐标域：相同颜色节点位置不随时间改变。
    ax2 = axes[2]
    ax2.set_xlim(-0.12, 1.12)
    ax2.set_ylim(-0.6, 0.6)
    ax2.axis("off")
    ax2.set_title(r"固定计算域  $0\leq x\leq1$", fontsize=13, fontweight="bold", pad=10)
    ax2.plot([0, 1], [0, 0], color=COLORS["navy"], linewidth=3, solid_capstyle="round")
    for frac, color in zip(node_fracs, node_colors):
        ax2.plot(frac, 0, "o", color=color, markersize=9, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax2.plot([frac, frac], [-0.065, 0.065], color=color, linewidth=1.2, alpha=0.75)
    ax2.text(0, -0.16, "0（中心）", ha="center", fontsize=10.5)
    ax2.text(1, -0.16, "1（表面）", ha="center", fontsize=10.5)
    ax2.text(0.5, 0.22, "节点固定，网格无需随边界重构", ha="center", fontsize=10.5, color=COLORS["teal"])

    # 在图级坐标中连接三个阶段。
    overlay = fig.add_axes((0, 0, 1, 1), frameon=False)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    add_arrow(overlay, (0.300, 0.515), (0.365, 0.515), color=COLORS["orange"], linewidth=2.1)
    overlay.text(0.333, 0.545, "半径收缩", ha="center", fontsize=10.2, color=COLORS["orange"], fontweight="bold")
    add_arrow(overlay, (0.630, 0.515), (0.695, 0.515), color=COLORS["purple"], linewidth=2.1)
    overlay.text(0.663, 0.545, r"$x=r/R(t)$", ha="center", fontsize=11, color=COLORS["purple"], fontweight="bold")

    fig.text(
        0.5,
        0.105,
        r"扩散项尺度：$1/R(t)^2$　　表面对流边界：$kT_x=R(t)h(T_\infty-T_s)$，$DC_x=R(t)h_m(C_\infty-C_s)$",
        ha="center",
        va="center",
        fontsize=11.2,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["grid"],
            "linewidth": 1.1,
        },
    )
    fig.text(
        0.5,
        0.045,
        "要点：坐标映射只改变方程中的尺度系数，不额外引入经验性收缩通量。",
        ha="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    save_figure(fig, "08_material_coordinate_mapping")


def model_framework() -> None:
    """绘制四问共用模型、输入数据和输出结果之间的关系。"""
    fig, ax = new_canvas((14.6, 8.2))
    fig.suptitle(
        "四问建模框架与数据流",
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.965,
    )
    ax.text(
        0.5,
        0.91,
        "共享同一套径向传热—传质方程，通过物性、时长与几何假设逐问递进",
        ha="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    input_specs = [
        (0.045, "环境边界", "附件 1\n温度与湿度时序", COLORS["orange"]),
        (0.37, "材料物性", "附件 2/3/4\n" + r"$k,\ c_p,\ \rho,\ D$", COLORS["teal"]),
        (0.695, "几何信息", "初始尺寸与附件 2\n半径收缩曲线 $R(t)$", COLORS["purple"]),
    ]
    for x_pos, title, body, color in input_specs:
        add_card(ax, x_pos, 0.755, 0.26, 0.12, facecolor=color + "12", edgecolor=color + "70", linewidth=1.4)
        ax.text(x_pos + 0.02, 0.835, title, fontsize=11.5, fontweight="bold", color=color)
        ax.text(x_pos + 0.13, 0.79, body, ha="center", va="center", fontsize=10.2, linespacing=1.35)
        add_arrow(ax, (x_pos + 0.13, 0.745), (0.5, 0.665), color=color, linewidth=1.5, mutation_scale=13)

    add_card(ax, 0.175, 0.425, 0.65, 0.23, facecolor=COLORS["white"], edgecolor=COLORS["navy"], linewidth=1.8)
    ax.text(0.5, 0.615, "共享耦合模型", ha="center", fontsize=14, fontweight="bold", color=COLORS["navy"])
    ax.text(
        0.5,
        0.555,
        r"$\rho c_p\,\partial T/\partial t=\frac{1}{r}\partial_r(kr\,\partial_rT)$"
        r"　　　　$\partial C/\partial t=\frac{1}{r}\partial_r(Dr\,\partial_rC)$",
        ha="center",
        fontsize=12.3,
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.49,
        "中心对称边界 + 表面对流边界　｜　有限体积空间离散 + BDF 时间积分",
        ha="center",
        fontsize=10.8,
        color=COLORS["muted"],
    )
    ax.text(
        0.5,
        0.45,
        "主输出：径向温度场、湿基含水率场、中心/表面过程线与达标时刻",
        ha="center",
        fontsize=10.8,
        color=COLORS["blue"],
        fontweight="bold",
    )

    question_specs = [
        (0.03, "问题 1", "基础场", "常物性 · 固定半径\n30 min 温湿分布", COLORS["navy"]),
        (0.275, "问题 2", "变物性", "$k,c_p,D$ 随状态变化\n3 h 温湿演化", COLORS["blue"]),
        (0.52, "问题 3", "事件求解", "固定半径\n求 " + r"$\max C\leq0.15$" + " 时刻", COLORS["orange"]),
        (0.765, "问题 4", "收缩修正", "材料坐标 $x=r/R(t)$\n重新计算达标时刻", COLORS["purple"]),
    ]
    centers = []
    for x_pos, q_no, title, body, color in question_specs:
        add_card(ax, x_pos, 0.12, 0.205, 0.205, facecolor=COLORS["white"], edgecolor=color + "90", linewidth=1.5)
        ax.text(x_pos + 0.018, 0.288, q_no, fontsize=10.5, color=color, fontweight="bold")
        ax.text(x_pos + 0.1025, 0.243, title, ha="center", fontsize=12.2, fontweight="bold", color=COLORS["ink"])
        ax.text(x_pos + 0.1025, 0.175, body, ha="center", va="center", fontsize=10.0, linespacing=1.45, color=COLORS["muted"])
        center_x = x_pos + 0.1025
        centers.append(center_x)
        add_arrow(ax, (0.5, 0.414), (center_x, 0.335), color=color, linewidth=1.6, mutation_scale=13)

    # 连接四问，强调“保持整体模型，仅逐步放宽假设”的主线。
    for i in range(3):
        add_arrow(
            ax,
            (centers[i] + 0.108, 0.085),
            (centers[i + 1] - 0.108, 0.085),
            color=COLORS["grid"],
            linewidth=1.8,
            mutation_scale=12,
        )
    ax.text(
        0.5,
        0.052,
        "由基础验证到变物性、事件判定和收缩几何：逐层增加复杂度，同时保留可对照的基线结果",
        ha="center",
        fontsize=10.6,
        color=COLORS["muted"],
    )

    save_figure(fig, "09_model_framework")


def drying_process_overview() -> None:
    """用三阶段插画概括升温、脱水和收缩过程。"""
    summary = load_summary()
    initial_moisture = float(summary["problem4_moisture"][0][0])
    stable_temperature = float(summary["T_const_c"])
    drying_time_h = float(summary["problem4_drying_time_h"])
    final_radius_cm = float(summary["problem4_radius_at_end_cm"])

    fig = plt.figure(figsize=(14.4, 7.2), facecolor=COLORS["paper"])
    fig.suptitle(
        "药材干燥过程概览",
        fontsize=21,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        "外部热量由表面向中心传递，内部水分由中心向表面迁移；持续脱水同时伴随径向收缩",
        ha="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    axes = [
        fig.add_axes((0.045, 0.22, 0.25, 0.59)),
        fig.add_axes((0.375, 0.22, 0.25, 0.59)),
        fig.add_axes((0.705, 0.22, 0.25, 0.59)),
    ]

    for stage_ax in axes:
        stage_ax.set_aspect("equal")
        stage_ax.set_xlim(-1.55, 1.55)
        stage_ax.set_ylim(-1.55, 1.55)
        stage_ax.axis("off")

    # 阶段一：初始状态近似均匀。
    ax0 = axes[0]
    ax0.set_title("Ⅰ  初始状态", fontsize=14, fontweight="bold", color=COLORS["navy"], pad=8)
    ax0.add_patch(
        Circle((0, 0), 1.0, facecolor="#DCEFF4", edgecolor=COLORS["blue"], linewidth=2.3)
    )
    for radius in (0.34, 0.67):
        ax0.add_patch(
            Circle((0, 0), radius, fill=False, edgecolor=COLORS["white"], linewidth=1.4, alpha=0.9)
        )
    ax0.text(0, 0.16, r"$T_0=28\ ^\circ\mathrm{C}$", ha="center", fontsize=13, color=COLORS["navy"])
    ax0.text(
        0,
        -0.17,
        rf"$C_0={initial_moisture:.2f}\ \mathrm{{kg/kg}}$",
        ha="center",
        fontsize=13,
        color=COLORS["blue"],
    )
    ax0.text(0, -1.25, r"温度、含水率近似均匀　$R_0=2.0\ \mathrm{cm}$", ha="center", fontsize=10.4, color=COLORS["muted"])

    # 阶段二：用由暖到冷的同心色层显示表面先升温，用双向箭头显示两类通量。
    ax1 = axes[1]
    ax1.set_title("Ⅱ  耦合传递", fontsize=14, fontweight="bold", color=COLORS["orange"], pad=8)
    ax1.add_patch(Circle((0, 0), 1.0, facecolor="#EFA45B", edgecolor=COLORS["orange"], linewidth=2.4))
    ax1.add_patch(Circle((0, 0), 0.74, facecolor="#F3C47D", edgecolor="none"))
    ax1.add_patch(Circle((0, 0), 0.47, facecolor="#A9DCE6", edgecolor="none"))
    ax1.add_patch(Circle((0, 0), 0.20, facecolor="#6DBED1", edgecolor="none"))
    for radius in (0.47, 0.74):
        ax1.add_patch(Circle((0, 0), radius, fill=False, edgecolor=COLORS["white"], linewidth=1.2, alpha=0.75))

    for angle in (45, 135, 225, 315):
        theta = np.deg2rad(angle)
        # 橙色箭头：环境热量指向药材内部。
        ax1.annotate(
            "",
            xy=(0.9 * np.cos(theta), 0.9 * np.sin(theta)),
            xytext=(1.42 * np.cos(theta), 1.42 * np.sin(theta)),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["orange"], "lw": 2.0},
        )
    for angle in (0, 90, 180, 270):
        theta = np.deg2rad(angle)
        # 蓝色箭头：内部水分由中心方向迁移到外界。
        ax1.annotate(
            "",
            xy=(1.42 * np.cos(theta), 1.42 * np.sin(theta)),
            xytext=(0.72 * np.cos(theta), 0.72 * np.sin(theta)),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["blue"], "lw": 2.0},
        )
    ax1.text(0, 0.16, "热量向内", ha="center", fontsize=12.5, color=COLORS["orange"], fontweight="bold")
    ax1.text(0, -0.17, "水分向外", ha="center", fontsize=12.5, color=COLORS["blue"], fontweight="bold")
    ax1.text(0, -1.25, "表面响应更快，中心形成温湿梯度", ha="center", fontsize=10.4, color=COLORS["muted"])

    # 阶段三：虚线保留初始轮廓，当前半径按 1.2/2.0 的真实比例绘制。
    ax2 = axes[2]
    ax2.set_title("Ⅲ  达标状态", fontsize=14, fontweight="bold", color=COLORS["teal"], pad=8)
    ax2.add_patch(
        Circle((0, 0), 1.0, fill=False, edgecolor=COLORS["grid"], linewidth=1.6, linestyle=(0, (4, 3)))
    )
    shrink_ratio = final_radius_cm / 2.0
    ax2.add_patch(
        Circle((0, 0), shrink_ratio, facecolor="#F1C98D", edgecolor=COLORS["teal"], linewidth=2.5)
    )
    ax2.add_patch(Circle((0, 0), shrink_ratio * 0.62, fill=False, edgecolor=COLORS["white"], linewidth=1.3))
    ax2.text(0, 0.12, "达标", ha="center", va="center", fontsize=16, color=COLORS["teal"], fontweight="bold")
    ax2.text(0, -0.22, r"$\max C\leq0.15$", ha="center", fontsize=13.5, color=COLORS["navy"], fontweight="bold")
    ax2.text(
        0,
        -1.25,
        rf"$R={final_radius_cm:.1f}\ \mathrm{{cm}}$　$t={drying_time_h:.3f}\ \mathrm{{h}}$",
        ha="center",
        fontsize=10.6,
        color=COLORS["muted"],
    )

    # 图级箭头串联三个阶段。
    overlay = fig.add_axes((0, 0, 1, 1), frameon=False)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    add_arrow(overlay, (0.302, 0.515), (0.365, 0.515), color=COLORS["gold"], linewidth=2.4, mutation_scale=17)
    overlay.text(0.333, 0.55, "升温", ha="center", fontsize=11, color=COLORS["orange"], fontweight="bold")
    add_arrow(overlay, (0.632, 0.515), (0.695, 0.515), color=COLORS["teal"], linewidth=2.4, mutation_scale=17)
    overlay.text(0.663, 0.55, "脱水 · 收缩", ha="center", fontsize=11, color=COLORS["teal"], fontweight="bold")

    fig.text(
        0.5,
        0.075,
        rf"稳定烘房温度约 {stable_temperature:.1f} °C　｜　结束判据采用全空间最大含水率，而非只检查表面节点",
        ha="center",
        fontsize=10.8,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["grid"],
            "linewidth": 1.1,
        },
    )

    save_figure(fig, "10_drying_process_overview")


def key_findings_summary() -> None:
    """用中心插画和四个信息卡总结论文最重要的定量结论。"""
    summary = load_summary()
    initial_moisture = float(summary["problem4_moisture"][0][0])
    stable_temperature = float(summary["T_const_c"])
    drying_time_h = float(summary["problem4_drying_time_h"])
    final_radius_cm = float(summary["problem4_radius_at_end_cm"])
    reduction_percent = float(summary["appendix4_shrinkage_reduction_percent"])

    fig, ax = new_canvas((14.6, 7.8))
    fig.suptitle(
        "模型关键发现概览",
        fontsize=21,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.965,
    )
    ax.text(
        0.5,
        0.905,
        "用温度、水分、尺寸和敏感度四个维度概括主要结论",
        ha="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    cards = [
        (0.035, 0.60, COLORS["orange"], "温度演化", rf"28 °C  →  ≈{stable_temperature:.0f} °C", "表面先升温，中心随后趋稳"),
        (0.70, 0.60, COLORS["blue"], "水分迁移", rf"{initial_moisture:.2f}  →  ≤0.15 kg/kg", "表面先失水，全空间最大值控制结束"),
        (0.035, 0.18, COLORS["purple"], "尺寸收缩", rf"2.0 cm  →  {final_radius_cm:.1f} cm", f"同为附录 4 物性时，时间缩短 {reduction_percent:.2f}%"),
        (0.70, 0.18, COLORS["teal"], "主导因素", "温度  >  半径  >  扩散系数", "轴向长度为零敏感仅源于一维模型假设"),
    ]
    for x_pos, y_pos, color, title, metric, note in cards:
        add_card(ax, x_pos, y_pos, 0.265, 0.205, facecolor=COLORS["white"], edgecolor=color + "80", linewidth=1.5)
        ax.add_patch(Circle((x_pos + 0.042, y_pos + 0.153), 0.023, facecolor=color + "20", edgecolor=color, linewidth=1.4))
        ax.text(x_pos + 0.042, y_pos + 0.153, "●", ha="center", va="center", fontsize=9, color=color)
        ax.text(x_pos + 0.077, y_pos + 0.151, title, va="center", fontsize=12.2, fontweight="bold", color=color)
        ax.text(x_pos + 0.1325, y_pos + 0.098, metric, ha="center", va="center", fontsize=12.0, fontweight="bold", color=COLORS["ink"])
        ax.text(x_pos + 0.1325, y_pos + 0.041, note, ha="center", va="center", fontsize=9.3, color=COLORS["muted"])

    # 中央截面把三种同时发生的趋势压缩为一个视觉核心。
    center = (0.5, 0.49)
    display_radius = 0.15
    x_radius = display_radius / (14.6 / 7.8)
    layer_specs = [
        (1.0, "#ECA35C"),
        (0.72, "#F2C67E"),
        (0.43, "#9ED8E4"),
    ]
    for frac, color in layer_specs:
        ax.add_patch(
            Ellipse(
                center,
                width=2 * x_radius * frac,
                height=2 * display_radius * frac,
                facecolor=color,
                edgecolor=COLORS["navy"] if frac == 1.0 else COLORS["white"],
                linewidth=2.2 if frac == 1.0 else 1.2,
                zorder=3,
            )
        )
    ax.text(0.5, 0.515, "干燥", ha="center", va="center", fontsize=17, color=COLORS["navy"], fontweight="bold", zorder=5)
    ax.text(0.5, 0.458, "T ↑   C ↓   R ↓", ha="center", va="center", fontsize=11.5, color=COLORS["ink"], zorder=5)
    ax.text(
        0.5,
        0.295,
        rf"问题 4：{drying_time_h:.3f} h 达标",
        ha="center",
        va="center",
        fontsize=11.2,
        color=COLORS["teal"],
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": COLORS["teal"] + "12",
            "edgecolor": COLORS["teal"] + "60",
            "linewidth": 1.1,
        },
    )

    # 从四个结论指向中心现象，视觉上表达这些量共同描述同一干燥过程。
    connector_specs = [
        ((0.30, 0.63), (0.425, 0.545), COLORS["orange"]),
        ((0.70, 0.63), (0.575, 0.545), COLORS["blue"]),
        ((0.30, 0.37), (0.425, 0.44), COLORS["purple"]),
        ((0.70, 0.37), (0.575, 0.44), COLORS["teal"]),
    ]
    for start, end, color in connector_specs:
        add_arrow(ax, start, end, color=color, linewidth=1.6, mutation_scale=13)

    ax.text(
        0.5,
        0.075,
        "结论边界：60.65% 仅表示附录 4 物性保持一致时的纯几何对照，不等同于问题 3 与问题 4 的直接差值。",
        ha="center",
        fontsize=10.3,
        color=COLORS["muted"],
    )

    save_figure(fig, "11_key_findings_summary")


def main() -> None:
    """生成全部论文插图。"""
    configure_style()
    physical_model_schematic()
    material_coordinate_mapping()
    model_framework()
    drying_process_overview()
    key_findings_summary()
    print("Generated paper illustrations in figures/ (PNG + SVG).")


if __name__ == "__main__":
    main()
