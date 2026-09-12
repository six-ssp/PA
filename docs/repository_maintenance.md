# 仓库维护与复现约定

## 本次维护范围

保持现有一维径向模型和正式求解参数不变，维护代码入口、结果检查、文档和提交边界。其他任务正在生成的论文工程、实验目录和未发布诊断文件不属于本次发布范围，不删除、不覆盖、不混入提交。

- 增加 `src/check_repository.py`：只读检查已发布成果，失败时返回非零退出码。
- 增加 `tests/`：覆盖环境插值、单位换算、经验物性、输出舍入、平衡解及数值回归。
- 增加 GitHub Actions：提交或 PR 时自动运行快速测试和成果检查。
- 导出脚本改为按脚本位置定位仓库，增加四份 JSON 的输入校验和 `--check-inputs` 只读模式。
- 分析入口增加缺失输入提示和 `--reuse-sensitivity`，可复用已有扫描数据重绘。
- 纠正问题 1 “常物性”简称，明确只有热物性固定，扩散系数仍为 `D(C)`。
- 问题 4 文档明确均匀比例收缩与材料导数假设，不能仅凭换元省略运动项。

## 检查命令

在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/check_repository.py
```

默认检查支持刚克隆的仓库，不要求被忽略的 `intermediate/result*.json`。若本地已有四问 JSON，再执行：

```powershell
python src/check_repository.py --with-intermediate
npm run check:export
```

前者逐单元格核对 JSON 和 Excel，包括药材外部应为空的位置；后者只校验导出输入，不重写 Excel，仍需要已有 Node.js 和 `@oai/artifact-tool` 环境。

完整数值回归重新求解问题 2 的 3 小时温度、水分场以及问题 3、4 的 641 节点烘干事件，但不覆盖发布文件：

```powershell
$env:PA_FULL_REGRESSION = "1"
python -m unittest discover -s tests -v
Remove-Item Env:PA_FULL_REGRESSION
```

默认 CI 只执行快速回归，长时回归按需本地执行。跨平台事件时间比较允许 1 s 浮点与求解器差异，不将它理解成实验精度。

## 更新顺序与依赖

1. `python src/run_all.py` 更新四问 JSON、摘要和收敛 CSV。
2. `npm run export` 将同一批 JSON 导出至官方模板。
3. `python src/check_repository.py --tables-only --with-intermediate` 检查四问表格与 JSON 一致性，此时分析数据尚未更新。
4. `python src/run_analysis.py` 更新误差分析、196 条扫描记录和 15 组图。
5. 再次运行测试与一致性检查，审查 diff 后按文件提交。

仅改排版时可使用 `python src/run_analysis.py --reuse-sensitivity`，但它仍重新运行误差分析和图 15 的事件求解；改变模型、附件或扫描参数后不能复用旧敏感度记录。Excel 导出依赖独立工具环境，Python 依赖安装成功并不意味着导出依赖也可用。

## 数值和解释边界

- 问题 3 正式事件时间为 `57.480707 h`，问题 4 为 `51.092979 h`；Excel 的最后一个规则时刻不是连续烘干事件时刻。
- 问题 4 在烘干事件时刻半径是 `1.200 cm`。附件 2 在 `259200 s` 的最后半径是 `1.198 cm`，附件结束后保持的是这个末值；二者不能混用。
- 敏感度扫描基准为 321 节点，正式四问长时结果为 641 节点；比较时必须按相同网格匹配。
- 只改变几何条件、保持附录 4 物性不变时，当前模型给出 `60.653334%` 的时长降幅；这不是所有真实药材的一般规律。
- 15 组 PNG/SVG 检查覆盖文件存在性与基本格式，不能自动证明没有文字重叠；排版调整仍需人工看图。
- Markdown 检查覆盖已跟踪文件的 UTF-8 替代字符、简单本地链接及块公式裸百分号，并非完整 LaTeX 渲染器。
- 数值回归和成果一致性通过不代表模型已获实验验证。收缩运动假设、长期边界处理和忽略端部效应仍须在论文中说明。

## 2026-09-12 本地核验记录

正式主模型与包含本地可选诊断代码的工作副本分别通过四问数值回归；未改动四问正式数值。最终本地测试共 13 项通过。已发布表格与本地 JSON 的 `669208` 个场值逐项一致，15 组 PNG/SVG、4 档收敛网格、196 条敏感度扫描记录及图 15 事件时刻检查通过。导出输入只读预检通过，未重写 Excel。GitHub Actions 已配置，远端执行结果应以 Actions 页面为准。

## 提交与隐私

不要使用 `git add .`。先运行 `git status --short` 和 `git diff --check`，逐个暂存本次修改，再查看 `git diff --cached --stat` 与暂存内容。

私人论文留在仓库外，或放在被忽略的 `private/` 下；`.gitignore` 另行忽略名为 `论文.pdf` 的文件。忽略规则不是安全隔离，也不会停止跟踪历史已提交文件，因此提交前仍需人工核对。
