# 问题 4：材料坐标下的半径收缩烘干模型

## 1. 建模目标与思路

问题 4 使用附件 2 的收缩半径 `R(t)` 和附录 4 的经验物性。定义

$$
x=\frac{r}{R(t)},\qquad 0\le x\le1.
$$

这里的 `x` 是随固体骨架收缩的材料坐标，固定 `x` 对应同一材料位置，而不是固定空间位置。因此方程中不再人为加入额外的收缩对流项；几何收缩通过 `1/R(t)^2` 的扩散、导热尺度及表面 Robin 边界中的 `R(t)` 体现。

附件 2 数据范围内对 `R(t)` 分段线性插值，数据结束后保持末个半径值。

## 2. 材料坐标下的控制方程

在固定材料坐标 `x` 下，温度方程写为

$$
\rho(C)c_p(C)\left.\frac{\partial T}{\partial t}\right|_x
=\frac{1}{R(t)^2}\frac1x\frac{\partial}{\partial x}
\left[xk(C)\frac{\partial T}{\partial x}\right].
$$

水分方程为

$$
\left.\frac{\partial C}{\partial t}\right|_x
=\frac{1}{R(t)^2}\frac1x\frac{\partial}{\partial x}
\left[xD(C,T)\frac{\partial C}{\partial x}\right].
$$

附录 4 的物性公式为

$$
\rho(C)=760+90C,
$$

$$
c_p(C)=1850+\frac{2150C}{C+1},
$$

$$
k(C)=0.12+\frac{0.20C}{C+1},
$$

$$
D(C,T)=4.2\times10^{-4}
\exp\left(-\frac{0.30}{C}\right)
\exp\left(-\frac{3850}{T+273.15}\right).
$$

## 3. 初始条件与边界条件

初始条件为

$$
T(x,0)=28\ ^\circ\mathrm C,
\qquad C(x,0)=2.55\ \mathrm{kg/kg}.
$$

材料中心 `x=0` 满足

$$
T_x(0,t)=0,\qquad C_x(0,t)=0.
$$

由 `\partial/\partial r=(1/R)\partial/\partial x`，实时表面 `x=1` 的 Robin 边界变为

$$
k(C_s)T_x(1,t)=R(t)h[T_\infty(t)-T_s(t)],
$$

$$
D(C_s,T_s)C_x(1,t)=R(t)h_m[C_\infty(t)-C_s(t)].
$$

长时间环境边界与问题 3 相同，即 `14400 s` 后采用稳定段均值。

## 4. 数值求解与输出

使用 641 个材料坐标节点、守恒有限体积离散和隐式 BDF 积分。烘干事件仍为

$$
g(t)=\max_{0\le x\le1}C(x,t)-0.15=0.
$$

同样使用两遍求解保证 Excel 时间列严格间隔 60 s。

`result4.xlsx` 同时提供两类位置：

- 固定物理坐标 `r=0,0.1,\ldots,2.0 cm`；当 `r>R(t)` 时该点已在药材外部，对应单元格留空；
- 动态“药材表面”列，始终输出材料坐标 `x=1` 的含水率。

网格收敛结果为

| 径向节点数 | 烘干时间/h |
|---:|---:|
| 81 | 51.126092 |
| 161 | 51.102741 |
| 321 | 51.095162 |
| 641 | 51.092979 |

最终得到

$$
\boxed{t_{\mathrm{dry},4}=183934.724387\ \mathrm s
=51.092979\ \mathrm h}.
$$

Excel 输出终点为 `183960 s`。

## 5. 收缩效应的独立对照

问题 3 与问题 4 同时改变了物性公式和半径模型，因此二者约 11.1128% 的时间差不能全部归因于收缩。为隔离纯几何影响，保持附录 4 物性不变，只比较：

| 附录 4 对照 | 烘干时间/h |
|---|---:|
| 固定 `R=2 cm` | 129.853389 |
| 附件 2 收缩 `R(t)` | 51.092979 |

定义

$$
\eta_{\mathrm{shrink}}
=\frac{t_{\mathrm{fixed,app4}}-t_{\mathrm{shrink,app4}}}
{t_{\mathrm{fixed,app4}}},
$$

可得

$$
\boxed{\eta_{\mathrm{shrink}}=60.653334\%}.
$$

该数值才表示在附录 4 物性一致条件下，几何收缩对预测烘干时间的单独影响。
