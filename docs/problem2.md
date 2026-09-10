# 问题 2：变物性条件下 3 小时温度场与水分场

## 1. 建模目标与思路

问题 2 沿用问题 1 的长圆柱、一维径向传热传质模型和 Robin 边界，但按照附录 3 引入随含水率和温度变化的密度、比热、导热系数与水分扩散系数。这样可以描述干燥过程中材料性质随状态变化的非线性影响。

半径固定为 `R=0.02 m`，求解区间为 `0≤t≤10800 s`，最终按每 1 s、每 0.1 cm 输出。

## 2. 控制方程

温度方程为

$$
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=\frac1r\frac{\partial}{\partial r}
\left[rk(C)\frac{\partial T}{\partial r}\right].
$$

水分方程为

$$
\frac{\partial C}{\partial t}
=\frac1r\frac{\partial}{\partial r}
\left[rD(C,T)\frac{\partial C}{\partial r}\right].
$$

附录 3 的经验物性公式为

$$
\rho(C)=650+128C,
$$

$$
c_p(C)=1450+\frac{2736C}{C+1},
$$

$$
k(C)=0.21+\frac{0.38C}{C+1},
$$

$$
D(C,T)=2.4\times10^{-3}
\exp\left(-\frac{0.45}{C}\right)
\exp\left(-\frac{3850}{T_K}\right),
\qquad T_K=T+273.15.
$$

其中 `T` 的求解和输出单位为摄氏度，只有 Arrhenius 指数中的温度使用开尔文。

## 3. 初始条件与边界条件

初始条件为

$$
T(r,0)=28\ ^\circ\mathrm C,\qquad C(r,0)=2.55\ \mathrm{kg/kg}.
$$

中心对称边界为

$$
T_r(0,t)=0,\qquad C_r(0,t)=0.
$$

表面 Robin 边界为

$$
k(C_s)T_r(R,t)=25[T_\infty(t)-T_s(t)],
$$

$$
D(C_s,T_s)C_r(R,t)=8\times10^{-7}[C_\infty(t)-C_s(t)].
$$

附件 1 的 `T∞(t)`、`C∞(t)` 在实测点之间分段线性插值。

## 4. 非线性界面物性的处理

在相邻节点之间先构造界面状态

$$
C_{i+1/2}=\frac{C_i+C_{i+1}}2,
\qquad
T_{i+1/2}=\frac{T_i+T_{i+1}}2,
$$

再计算

$$
k_{i+1/2}=k(C_{i+1/2}),
\qquad
D_{i+1/2}=D(C_{i+1/2},T_{i+1/2}).
$$

这种处理直接以界面热力学状态评价连续非线性物性。空间采用 321 节点守恒有限体积法，时间采用隐式 BDF 法。

## 5. 结果说明

3 小时内温度逐渐接近烘房稳定温度，水分场始终呈现中心高、表面低的径向梯度。完整逐秒结果见 `results/result2.xlsx`。
