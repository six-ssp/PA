"""2026 CUMCM A 题：药材烘干的一维径向传热传质模型。

模型把长圆柱的轴向端部效应忽略，只求轴向中部截面上的径向分布。
空间离散使用守恒型有限体积法，时间积分使用 SciPy 的隐式 BDF 方法。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Environment:
    """烘房温度和水分浓度的分段线性插值函数。"""

    time_s: np.ndarray
    temperature_c: np.ndarray
    moisture: np.ndarray

    def temperature(self, t: float | np.ndarray) -> np.ndarray:
        # np.interp 在数据区间外自动保持首/末测量值。
        return np.interp(t, self.time_s, self.temperature_c)

    def water(self, t: float | np.ndarray) -> np.ndarray:
        return np.interp(t, self.time_s, self.moisture)


@dataclass(frozen=True)
class MaterialLaw:
    """材料热物性和水分扩散系数经验式。"""

    density: Callable[[np.ndarray], np.ndarray]
    heat_capacity: Callable[[np.ndarray], np.ndarray]
    conductivity: Callable[[np.ndarray], np.ndarray]
    diffusivity: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SimulationResult:
    """连续时间数值解及其空间网格。"""

    solution: object
    x: np.ndarray
    radius: Callable[[float | np.ndarray], np.ndarray]
    drying_time_s: float | None = None

    def sample(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回网格节点处的温度和水分浓度，数组形状为 (nt, nx)。"""
        values = self.solution.sol(np.asarray(times_s, dtype=float))
        n = self.x.size
        return values[:n].T, values[n:].T


def load_environment_xlsx(path: str | Path) -> Environment:
    """只读解析附件 1。

    为使数值模型本身不依赖 Excel 写入库，这里直接读取 xlsx(zip/XML) 中
    Sheet1 的前三列。附件 1 的内容均为数值，且字符串只出现在表头。
    """
    import zipfile
    import xml.etree.ElementTree as ET

    path = Path(path)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[float | str | None]] = []
    for row in sheet.findall(".//m:sheetData/m:row", ns):
        current: list[float | str | None] = []
        for cell in row.findall("m:c", ns):
            ref = cell.attrib["r"]
            col = 0
            for ch in ref:
                if not ch.isalpha():
                    break
                col = col * 26 + ord(ch.upper()) - 64
            while len(current) < col - 1:
                current.append(None)
            value_node = cell.find("m:v", ns)
            if value_node is None:
                value: float | str | None = None
            elif cell.attrib.get("t") == "s":
                value = shared[int(value_node.text)]
            else:
                value = float(value_node.text)
            current.append(value)
        rows.append(current)

    data = np.asarray([[float(v) for v in row[:3]] for row in rows[1:] if len(row) >= 3], dtype=float)
    return Environment(data[:, 0], data[:, 1], data[:, 2])


def load_radius_xlsx(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取附件 2 的时间(s)-半径(cm)数据。"""
    # 复用附件 1 的轻量 XML 读取思路，只取前两列。
    import zipfile
    import xml.etree.ElementTree as ET

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(Path(path)) as archive:
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    result: list[tuple[float, float]] = []
    for row in sheet.findall(".//m:sheetData/m:row", ns)[1:]:
        values = [c.find("m:v", ns) for c in row.findall("m:c", ns)]
        if len(values) >= 2 and values[0] is not None and values[1] is not None:
            result.append((float(values[0].text), float(values[1].text)))
    arr = np.asarray(result, dtype=float)
    return arr[:, 0], arr[:, 1] / 100.0  # cm -> m


def law_problem1() -> MaterialLaw:
    """附录 2 的常物性与浓度相关扩散系数。"""
    return MaterialLaw(
        density=lambda c: np.full_like(c, 820.0),
        heat_capacity=lambda c: np.full_like(c, 2600.0),
        conductivity=lambda c: np.full_like(c, 0.36),
        diffusivity=lambda c, tk: 7.0e-9 * np.exp(-0.89 / np.maximum(c, 1.0e-8)),
    )


def law_problem23() -> MaterialLaw:
    """附录 3 的问题 2、3 变物性经验式。"""
    return MaterialLaw(
        density=lambda c: 650.0 + 128.0 * c,
        heat_capacity=lambda c: 1450.0 + 2736.0 * c / (c + 1.0),
        conductivity=lambda c: 0.21 + 0.38 * c / (c + 1.0),
        diffusivity=lambda c, tk: 2.4e-3
        * np.exp(-0.45 / np.maximum(c, 1.0e-8))
        * np.exp(-3850.0 / tk),
    )


def law_problem4() -> MaterialLaw:
    """附录 4 的问题 4 变物性经验式。"""
    return MaterialLaw(
        density=lambda c: 760.0 + 90.0 * c,
        heat_capacity=lambda c: 1850.0 + 2150.0 * c / (c + 1.0),
        conductivity=lambda c: 0.12 + 0.20 * c / (c + 1.0),
        diffusivity=lambda c, tk: 4.2e-4
        * np.exp(-0.30 / np.maximum(c, 1.0e-8))
        * np.exp(-3850.0 / tk),
    )


def simulate(
    environment: Environment,
    material: MaterialLaw,
    end_time_s: float,
    *,
    radius: Callable[[float | np.ndarray], np.ndarray] | None = None,
    nodes: int = 161,
    stop_at_dry: bool = False,
    rtol: float = 2.0e-6,
    atol: float = 2.0e-8,
) -> SimulationResult:
    """求解温度-水分场。

    x=r/R(t) 是无量纲半径。固定半径时 R(t)=0.02 m；问题 4 使用附件 2
    给出的收缩半径。把 x 视作随固体收缩的材料坐标，因此移动网格中不再额外
    添加对流项；收缩通过 1/R(t)^2 扩散尺度和表面对流边界进入方程。
    """
    if nodes < 5:
        raise ValueError("nodes 至少为 5")
    x = np.linspace(0.0, 1.0, nodes)
    dx = x[1] - x[0]
    # 节点控制体的左右边界及无量纲面积积分 int(x dx)。
    left = np.maximum(0.0, x - 0.5 * dx)
    right = np.minimum(1.0, x + 0.5 * dx)
    volume = 0.5 * (right**2 - left**2)
    radius_fn = radius or (lambda t: np.full_like(np.asarray(t, dtype=float), 0.02))

    h_heat = 25.0       # W/(m^2 K)
    h_mass = 8.0e-7     # m/s

    def divergence(field: np.ndarray, coeff: np.ndarray, boundary_gradient_flux: float) -> np.ndarray:
        """计算 (1/x)d(x*coeff*d(field)/dx)/dx 的有限体积离散。"""
        # k(C)、D(C,T) 是连续状态函数而非分层介质参数。界面中心的二阶近似
        # 应采用相邻节点系数的算术平均；调和平均会在低含水表层人为放大阻力。
        face_coeff = 0.5 * (coeff[:-1] + coeff[1:])
        interior_flux = face_coeff * np.diff(field) / dx
        flux_left = np.zeros(nodes)
        flux_right = np.zeros(nodes)
        flux_left[1:] = interior_flux
        flux_right[:-1] = interior_flux
        flux_right[-1] = boundary_gradient_flux
        return (right * flux_right - left * flux_left) / volume

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        temperature = y[:nodes]
        water = np.maximum(y[nodes:], 1.0e-10)
        radius_now = float(np.asarray(radius_fn(t)))
        rho = material.density(water)
        cp = material.heat_capacity(water)
        k = material.conductivity(water)
        d = material.diffusivity(water, temperature + 273.15)

        # Robin 边界在 x 坐标中：k*T_x=R*h*(T_env-T_surface)。
        heat_boundary = radius_now * h_heat * (float(environment.temperature(t)) - temperature[-1])
        mass_boundary = radius_now * h_mass * (float(environment.water(t)) - water[-1])
        d_temperature = divergence(temperature, k, heat_boundary) / (radius_now**2 * rho * cp)
        d_water = divergence(water, d, mass_boundary) / radius_now**2
        return np.concatenate((d_temperature, d_water))

    initial = np.concatenate((np.full(nodes, 28.0), np.full(nodes, 2.55)))

    events = None
    if stop_at_dry:
        def dry_event(t: float, y: np.ndarray) -> float:
            # 中心通常最后达到阈值，但这里检查所有节点，更贴合“各处低于 0.15”。
            return float(np.max(y[nodes:]) - 0.15)

        dry_event.terminal = True
        dry_event.direction = -1
        events = dry_event

    solution = solve_ivp(
        rhs,
        (0.0, float(end_time_s)),
        initial,
        method="BDF",
        dense_output=True,
        events=events,
        max_step=60.0,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    drying_time = None
    if stop_at_dry and solution.t_events and solution.t_events[0].size:
        drying_time = float(solution.t_events[0][0])
    return SimulationResult(solution, x, radius_fn, drying_time)


def interpolate_fixed_radius(
    result: SimulationResult,
    times_s: np.ndarray,
    distances_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把数值解从内部网格插值到指定物理半径。"""
    temperature, water = result.sample(times_s)
    radius_m = np.asarray(result.radius(times_s), dtype=float)
    if radius_m.ndim == 0:
        radius_m = np.full(times_s.size, float(radius_m))
    target_m = np.asarray(distances_cm, dtype=float) / 100.0
    out_t = np.empty((times_s.size, target_m.size))
    out_c = np.empty_like(out_t)
    for i, current_radius in enumerate(radius_m):
        physical_nodes = result.x * current_radius
        out_t[i] = np.interp(target_m, physical_nodes, temperature[i])
        out_c[i] = np.interp(target_m, physical_nodes, water[i])
    return out_t, out_c
