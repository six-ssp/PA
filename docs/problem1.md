# 问题 1：常物性条件下 30 分钟温度场与水分场

## 1. 建模目标与思路

将药材近似为半径 `R=0.02 m` 的均匀长圆柱，忽略轴向传递和端面效应，仅计算轴向中部截面的一维径向温度 `T(r,t)` 与干基含水率 `C(r,t)`。问题 1 的物性取附录 2，温度场为常物性非稳态导热，水分场采用含水率相关扩散系数的 Fick 第二定律。烘房环境使用附件 1 相邻测点间的线性插值。

求解区间为 `0≤t≤1800 s`，最终按每 1 s、每 0.1 cm 输出一次结果。

## 2. 控制方程

圆柱坐标下的径向导热方程为

$$
\rho c_p\frac{\partial T}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left(rk\frac{\partial T}{\partial r}\right),
\qquad 0<r<R.
$$

水分迁移方程为

$$
\frac{\partial C}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rD(C)\frac{\partial C}{\partial r}\right].
$$

附录 2 的物性关系为

$$
\rho=820\ \mathrm{kg/m^3},\qquad
c_p=2600\ \mathrm{J/(kg\cdot K)},\qquad
k=0.36\ \mathrm{W/(m\cdot K)},
$$

$$
D(C)=7.0\times10^{-9}\exp\left(-\frac{0.89}{C}\right)\ \mathrm{m^2/s}.
$$

## 3. 初始条件与边界条件

初始状态均匀：

$$
T(r,0)=28\ ^\circ\mathrm C,\qquad C(r,0)=2.55\ \mathrm{kg/kg}.
$$

圆柱中心满足对称条件：

$$
\left.\frac{\partial T}{\partial r}\right|_{r=0}=0,
\qquad
\left.\frac{\partial C}{\partial r}\right|_{r=0}=0.
$$

表面采用第三类 Robin 边界：

$$
k\left.\frac{\partial T}{\partial r}\right|_{r=R}
=h\,[T_\infty(t)-T_s(t)],
$$

$$
D(C_s)\left.\frac{\partial C}{\partial r}\right|_{r=R}
=h_m\,[C_\infty(t)-C_s(t)],
$$

其中

$$
h=25\ \mathrm{W/(m^2\cdot K)},\qquad h_m=8\times10^{-7}\ \mathrm{m/s}.
$$

## 4. 数值方法

令 `x=r/R`，在 `x∈[0,1]` 上划分 321 个径向节点。第 `i` 个控制体的无量纲体积权重为

$$
V_i^*=\frac12\left(x_{i+1/2}^2-x_{i-1/2}^2\right).
$$

对一般扩散项的守恒离散为

$$
\left[\frac1x\frac{\partial}{\partial x}
\left(x\Gamma\frac{\partial\phi}{\partial x}\right)\right]_i
\approx
\frac{x_{i+1/2}F_{i+1/2}-x_{i-1/2}F_{i-1/2}}{V_i^*},
$$

其中

$$
F_{i+1/2}=\Gamma_{i+1/2}
\frac{\phi_{i+1}-\phi_i}{\Delta x}.
$$

空间离散后得到常微分方程组，使用 `scipy.solve_ivp(method="BDF")` 积分；最后把节点值插值到 `0,0.1,\ldots,2.0 cm`。

## 5. 结果说明

30 分钟内热量由表面向中心传递，表面升温快于中心；水分由中心向表面扩散，表面含水率下降最快。完整逐秒结果见 `results/result1.xlsx`。
