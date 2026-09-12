"""统一生成误差分析、合理性指标和全部论文图。"""

from __future__ import annotations

import argparse

from problem_utils import ROOT

from analyze_errors import main as analyze_errors
from generate_figures import main as generate_figures
from generate_illustrations import main as generate_illustrations
from generate_method_figures import main as generate_method_figures
from sensitivity_analysis import main as analyze_sensitivity


def main(*, reuse_sensitivity: bool = False) -> None:
    """按依赖顺序运行分析与绘图，避免结果表、数据图和插图不同步。"""
    # 图 1~5 读取正式 Excel，先检查必需输入，避免跑到中途才发现文件缺失。
    required = [ROOT / "intermediate/summary.json"]
    required += [ROOT / "results" / f"result{i}.xlsx" for i in range(1, 5)]
    if reuse_sensitivity:
        required += [ROOT / "results/sensitivity_summary.json", ROOT / "results/sensitivity_sweep.csv"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少分析输入：" + ", ".join(missing) + "；请先求解并导出 Excel。")
    analyze_errors()
    generate_figures()
    analyze_sensitivity(plot_only=reuse_sensitivity)
    generate_illustrations()
    generate_method_figures()
    print("全部误差、敏感度和论文图已更新")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-sensitivity", action="store_true", help="仅重绘已有敏感度数据；误差分析和图 15 仍重新求解")
    main(reuse_sensitivity=parser.parse_args().reuse_sensitivity)
