# 问题 3：固定半径条件下烘干结束时间

## 1. 建模目标与思路

问题 3 使用问题 2 的附录 3 变物性模型，将计算时间延长到药材所有位置的含水率均不高于 `0.15 kg/kg`。半径保持 `R=0.02 m`，使用事件函数对全空间最大含水率进行连续检测，而不是只判断中心节点。

## 2. 控制方程与物性

控制方程仍为

$$
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=\frac1r\frac{\partial}{\partial r}
\left[rk(C)\frac{\partial T}{\partial r}\right],
$$

$$
\frac{\partial C}{\partial t}
=\frac1r\frac{\partial}{\partial r}
\left[rD(C,T)\frac{\partial C}{\partial r}\right].
$$

附录 3 物性为

$$
\rho=650+128C,
\qquad
c_p=1450+\frac{2736C}{C+1},
$$

$$
k=0.21+\frac{0.38C}{C+1},
$$

$$
D=2.4\times10^{-3}
\exp\left(-\frac{0.45}{C}\right)
\exp\left(-\frac{3850}{T+273.15}\right).
$$

初始条件、中心对称边界和表面 Robin 边界与问题 2 相同。

## 3. 长时间环境边界

附件 1 数据范围内使用线性插值。对 `t>14400 s`，不保持最后一个带有随机波动的测量点，而是取稳定平台 `12000~14400 s` 的均值：

$$
T_{\mathrm{const}}=\frac1N\sum_{t_j\in[12000,14400]}T_{\infty,j}
=49.99875610\ ^\circ\mathrm C,
$$

$$
C_{\mathrm{const}}=\frac1N\sum_{t_j\in[12000,14400]}C_{\infty,j}
=0.04998415\ \mathrm{kg/kg}.
$$

因此

$$
T_\infty(t)=T_{\mathrm{const}},\quad
C_\infty(t)=C_{\mathrm{const}},\qquad t>14400\ \mathrm s.
$$

## 4. 烘干事件与两遍求解

定义事件函数

$$
g(t)=\max_{0\le r\le R}C(r,t)-0.15.
$$

当 `g(t)` 由正变负并首次等于零时，记为连续烘干结束时刻 `t_dry`。

为同时满足精确事件检测和 Excel 严格 60 s 输出，采用两遍求解：

1. 第一遍以 `g(t)=0` 为终止事件，得到连续的 `t_dry`；
2. 第二遍计算至

$$
t_{\mathrm{out}}=60\left\lceil\frac{t_{\mathrm{dry}}}{60}\right\rceil,
$$

并只输出 `0,60,120,...,t_out`，不把非整点事件时刻插入 Excel。

## 5. 网格无关性与最终结果

| 径向节点数 | 烘干时间/h |
|---:|---:|
| 81 | 57.648612 |
| 161 | 57.537932 |
| 321 | 57.495499 |
| 641 | 57.480707 |

321 到 641 节点的变化约为 53.25 s，正式结果采用 641 节点：

$$
\boxed{t_{\mathrm{dry},3}=206930.545593\ \mathrm s
=57.480707\ \mathrm h}.
$$

Excel 输出终点为 `206940 s`，完整结果见 `results/result3.xlsx`，连续事件时刻见 `intermediate/summary.json`。
