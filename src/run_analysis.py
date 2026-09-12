"""统一生成误差分析、合理性指标和全部论文图。"""

from __future__ import annotations

from analyze_errors import main as analyze_errors
from generate_figures import main as generate_figures
from sensitivity_analysis import main as analyze_sensitivity


def main() -> None:
    """按依赖顺序运行三组分析，避免结果表与图片不同步。"""
    analyze_errors()
    generate_figures()
    analyze_sensitivity()
    print("全部误差、敏感度和论文图已更新")


if __name__ == "__main__":
    main()
