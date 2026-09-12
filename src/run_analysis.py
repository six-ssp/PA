"""统一生成误差分析、合理性指标和全部论文图。"""

from __future__ import annotations

from analyze_errors import main as analyze_errors
from generate_figures import main as generate_figures
from generate_illustrations import main as generate_illustrations
from generate_method_figures import main as generate_method_figures
from sensitivity_analysis import main as analyze_sensitivity


def main() -> None:
    """按依赖顺序运行分析与绘图，避免结果表、数据图和插图不同步。"""
    analyze_errors()
    generate_figures()
    analyze_sensitivity()
    generate_illustrations()
    generate_method_figures()
    print("全部误差、敏感度和论文图已更新")


if __name__ == "__main__":
    main()
