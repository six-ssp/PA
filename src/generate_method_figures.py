"""论文推导用示意图：物理抽象、控制体、材料坐标与事件判据。

借鉴板凳龙优秀论文中“局部几何图—公式—结果验证”的表达方式；
所有图形均为本题原创绘制。仅图 15 使用数值解，其余为标明假设的示意图。
运行：python src/generate_method_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as DrawingPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK, GRAY = "#253849", "#768491"
HEAT, WATER, PURPLE, GREEN = "#C56336", "#237DA0", "#77619B", "#258677"
LIGHT = "#DCE3E8"


def style():
    """文本保留为 SVG 文本，图形与字体尺寸按实际论文栏宽统一调整。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.edgecolor": LIGHT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "PA-method-figures-v1",
    })


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=.16)
    path = OUT / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=.16, metadata={"Date": None})
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def diagram(ax, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5)):
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")


def arrow(ax, start, end, color=INK, lw=1.5, both=False, ls="-"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "<->" if both else "-|>",
                            "color": color, "lw": lw, "linestyle": ls,
                            "shrinkA": 0, "shrinkB": 0})


def panel(ax, label):
    ax.set_title(label, loc="left", fontsize=11, fontweight="bold", pad=16)


def cylinder(ax, x, y, length, radius):
    ax.add_patch(Rectangle((x, y-radius), length, 2*radius,
                           facecolor="#F4E9D6", edgecolor="#AD8861", lw=1.3))
    for cap_x in (x, x+length):
        ax.add_patch(Ellipse((cap_x, y), .45*radius, 2*radius,
                             facecolor="#E7D1AD", edgecolor="#AD8861", lw=1.3))
    ax.plot([x-.2, x+length+.3], [y,y], color=GRAY, ls=(0,(4,3)), lw=.8)


def abstraction():
    """实物示意→几何模型→径向截面；用颜色区分热流与水分通量。"""
    fig, axes = plt.subplots(1, 3, figsize=(12,4.2), gridspec_kw={"width_ratios":[1.05,1.2,1]})
    fig.subplots_adjust(left=.03, right=.98, bottom=.22, top=.84, wspace=.28)
    for ax in axes:
        diagram(ax, (-1.6,1.6), (-1.35,1.35))
    a,b,c = axes
    panel(a, "(a) 药材外形与几何近似")
    # 轮廓仅示意长条状药材，不代表特定品种或真实测量形状。
    verts=[(-1.35,-.1),(-1.4,.38),(-.9,.46),(-.45,.39),(.05,.52),(.64,.27),
           (1.3,.29),(1.49,-.12),(1.1,-.34),(.55,-.38),(.05,-.26),(-.63,-.43),(-1.35,-.1)]
    codes=[DrawingPath.MOVETO]+[DrawingPath.CURVE3]*12
    a.add_patch(PathPatch(DrawingPath(verts,codes),fc="#EBD8B9",ec="#A9845A",lw=1.3))
    for x in [-.85,-.35,.2,.73]:
        a.plot([x-.1,x+.06],[.23,-.20],color="#A9845A",lw=.9,alpha=.55)
    a.text(0,-.70,"长条状药材（外形示意）",ha="center",color=GRAY)
    a.text(0,-1.09,r"长圆柱近似：$L\gg 2R$",ha="center")
    panel(b, "(b) 中部截面与传递方向")
    cylinder(b,-1.18,0,2.3,.37)
    b.add_patch(Ellipse((0,0),.17,.74,fill=False,ec=PURPLE,lw=1.5,ls="--"))
    for x in [-.8,.65]:
        arrow(b,(x,.88),(x,.42),HEAT)
        arrow(b,(x,-.42),(x,-.88),WATER)
    b.text(0,1.06,"热量向内",ha="center",color=HEAT)
    b.text(0,-1.07,"水分向外",ha="center",color=WATER)
    b.annotate("取轴向中部",xy=(.04,-.31),xytext=(-.4,-.64),fontsize=9,color=PURPLE,
               arrowprops={"arrowstyle":"-","color":PURPLE})
    panel(c, "(c) 一维径向计算域")
    c.add_patch(Circle((0,0),.8,fc="#F7F4EC",ec=INK,lw=1.5))
    for r in [.2,.4,.6]:
        c.add_patch(Circle((0,0),r,fill=False,ec=LIGHT,lw=.9))
    c.plot([0,.8],[0,0],color=INK,lw=1)
    c.scatter(np.linspace(0,.8,5),np.zeros(5),s=21,c=WATER,zorder=3)
    arrow(c,(0,-.27),(.8,-.27),INK)
    c.text(.4,-.48,r"$r\in[0,R(t)]$",ha="center")
    c.text(0,.14,"中心",ha="center",fontsize=9)
    c.text(.87,.12,"表面",ha="left",fontsize=9)
    c.text(0,-1.07,r"$T=T(r,t),\quad C=C(r,t)$",ha="center")
    fig.text(.5,.07,"仅考虑径向传递；忽略轴向梯度与端面效应。橙色：热流；蓝色：水分通量。",ha="center",fontsize=10,color=GRAY)
    save(fig,"12_model_abstraction")


def control_volume():
    """环形体积权重和局部面通量，直接对应 model.divergence 的离散。"""
    fig, axes = plt.subplots(1,2,figsize=(12,5),gridspec_kw={"width_ratios":[1,1.55]})
    fig.subplots_adjust(left=.045,right=.97,bottom=.26,top=.83,wspace=.27)
    a,b=axes
    diagram(a,(-1.28,1.35),(-1.22,1.18))
    panel(a,"(a) 第 i 个环形控制体")
    a.add_patch(Circle((0,0),1,fill=False,ec=INK,lw=1.3))
    a.add_patch(Wedge((0,0),.7,0,360,width=.2,fc="#DDEFF4",ec=WATER,lw=1.2))
    for r in [.3,.9]:
        a.add_patch(Circle((0,0),r,fill=False,ec=LIGHT,lw=.8))
    a.plot([0,1.15],[0,0],color=GRAY,lw=.8)
    a.scatter([0,.4,.6,.8,1],np.zeros(5),s=22,c=INK,zorder=4)
    a.text(.6,.13,r"$i$",ha="center",fontsize=12)
    a.text(0,-.18,"O",ha="center",fontsize=10)
    for r,theta,lab in [(.5,135,r"$r_{i-1/2}$"),(.7,62,r"$r_{i+1/2}$")]:
        pt=np.array([np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))])*r
        arrow(a,(0,0),pt,GRAY,lw=1)
        a.text(pt[0]-.09,pt[1]+.13,lab,ha="center",fontsize=12)
    a.text(0,-1.14,"高亮区域为一个控制体；每单位轴向长度",ha="center",fontsize=9,color=GRAY)
    diagram(b,(-.45,3.45),(-.9,1.04))
    panel(b,"(b) 邻接节点与界面通量")
    b.add_patch(Rectangle((1, -.32),1,.72,fc="#DDEFF4",ec="none"))
    b.plot([.3,2.8],[0,0],color=GRAY,lw=1)
    for x,lab in [(.5,r"$i-1$"),(1.5,r"$i$"),(2.5,r"$i+1$")]:
        b.scatter([x],[0],s=35,c=INK,zorder=4)
        b.text(x,-.18,lab,ha="center",fontsize=12)
        b.text(x,.15,r"$u_{"+lab.strip("$")+r"}$",ha="center",fontsize=12)
    for x,lab in [(1,r"$r_{i-1/2}$"),(2,r"$r_{i+1/2}$")]:
        b.plot([x,x],[-.35,.48],color=WATER,lw=1.2,ls="--")
        b.text(x,-.52,lab,ha="center",fontsize=12,color=WATER)
    arrow(b,(.7,.65),(1.3,.65),HEAT)
    arrow(b,(1.7,.65),(2.3,.65),HEAT)
    b.text(1,.79,r"$J_{i-1/2}$",ha="center",color=HEAT,fontsize=12)
    b.text(2,.79,r"$J_{i+1/2}$",ha="center",color=HEAT,fontsize=12)
    arrow(b,(1,-.71),(2,-.71),GRAY,both=True,lw=1)
    b.text(1.5,-.9,r"$\Delta r$",ha="center",fontsize=12)
    b.text(3.06,.02,r"$+r$",ha="center",fontsize=11)
    arrow(b,(2.87,0),(3.18,0),GRAY,lw=1)
    fig.text(.5,.18,r"$V_i=\pi\left(r_{i+1/2}^{\,2}-r_{i-1/2}^{\,2}\right),\qquad A_{i\pm1/2}=2\pi r_{i\pm1/2}$",ha="center",fontsize=13)
    fig.text(.5,.105,r"$b_i V_i\,\dot u_i=A_{i-1/2}J_{i-1/2}-A_{i+1/2}J_{i+1/2},\qquad J=-a\,\partial u/\partial r$",ha="center",fontsize=13)
    fig.text(.5,.025,r"$u=T$: $a=k,\ b=\rho c_p$；$u=C$: $a=D,\ b=1$。箭头定义通量的正方向，实际符号由梯度决定。",ha="center",fontsize=10,color=GRAY)
    save(fig,"13_radial_control_volume")


def material_coordinates():
    """用同色节点连线强调固定材料坐标；明确这还需要均匀径向收缩假设。"""
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={"width_ratios":[1.2,1]})
    fig.subplots_adjust(left=.055,right=.97,bottom=.26,top=.82,wspace=.24)
    a,b=axes
    diagram(a,(-.6,2.55),(-.42,1.6))
    panel(a,"(a) 物理域：材料点随收缩移动")
    fractions=np.array([0,.25,.5,.75,1.])
    colors=[INK,WATER,GREEN,PURPLE,HEAT]
    for y,radius,label in [(1.22,2.,r"$t_0$"),(.22,1.2,r"$t_1$")]:
        a.plot([0,radius],[y,y],color=GRAY,lw=1.5)
        a.text(-.3,y,label,fontsize=12,va="center",ha="right")
        for f,col in zip(fractions,colors):
            a.scatter([radius*f],[y],s=45,c=col,zorder=3,edgecolors="white",linewidths=.6)
        a.plot([radius,radius],[y-.1,y+.1],color=HEAT,lw=1)
    for f,col in zip(fractions[1:],colors[1:]):
        arrow(a,(2*f,1.1),(1.2*f,.34),col,lw=1,ls="--")
    a.text(2.14,1.19,r"初始表面 $R_0$",fontsize=9,color=HEAT)
    a.text(1.36,.15,r"当前表面 $R_1$",fontsize=9,color=HEAT)
    a.text(.8,-.3,r"$r_i(t)=x_iR(t)$",ha="center",fontsize=13)
    diagram(b,(-.2,1.25),(-.45,.78))
    panel(b,"(b) 计算域：同一材料点的 x 坐标固定")
    for y,label in [(.48,r"$t_0$"),(-.02,r"$t_1$")]:
        b.plot([0,1],[y,y],color=GRAY,lw=1.4)
        b.text(-.15,y,label,va="center",fontsize=12)
        for f,col in zip(fractions,colors):
            b.scatter([f],[y],s=45,c=col,zorder=3,edgecolors="white",linewidths=.6)
    for f,col in zip(fractions,colors):
        b.plot([f,f],[.48,-.02],ls="--",color=col,lw=.9,alpha=.6)
    b.text(0,-.18,"0（中心）",ha="center",fontsize=9)
    b.text(1,-.18,"1（表面）",ha="center",fontsize=9)
    b.text(.5,-.38,r"$x_i=r_i(t)/R(t)$",ha="center",fontsize=13)
    fig.text(.5,.16,r"均匀径向收缩假设：$v_r=(\dot R/R)r$，因此材料点满足 $\mathrm{d}x_i/\mathrm{d}t=0$。",ha="center",fontsize=11)
    fig.text(.5,.085,r"$\left.\partial_t u\right|_x=\left.\partial_t u\right|_r+v_r\,\partial_r u,\qquad \mathcal{L}_r=R(t)^{-2}\mathcal{L}_x$",ha="center",fontsize=13)
    fig.text(.5,.02,"几何比例仅作示意；材料坐标解释依赖运动假设，仅有坐标换元并不自动消去输运项。",ha="center",fontsize=9.5,color=GRAY)
    save(fig,"14_material_coordinate_trajectories")


def event_criterion():
    """重采样 641 节点问题 3 解，并核对其事件时间与已发布摘要一致。"""
    from model import load_environment_xlsx, law_problem23, simulate
    summary=json.loads((ROOT/"intermediate/summary.json").read_text(encoding="utf-8"))
    env=load_environment_xlsx(ROOT/"附件1.xlsx")
    event=simulate(env,law_problem23(),72*3600,nodes=641,stop_at_dry=True)
    tstar=event.drying_time_s
    if tstar is None or abs(tstar-summary["problem3_drying_time_s"])>1:
        raise RuntimeError("当前模型事件时间与正式摘要不一致，请先核对主模型结果。")
    # 使用同一次事件求解的连续输出；仅展示事件前及事件处，不外推终止后的解。
    times=np.array([18*3600,48*3600,tstar])
    _,profiles=event.sample(times)
    trace_time=np.unique(np.r_[np.linspace(0,tstar,1801),times])
    _,field=event.sample(trace_time)
    max_c=field.max(axis=1)
    if abs(profiles[-1].max()-.15)>1e-6 or not np.all(np.isfinite(field)):
        raise RuntimeError("判据图数值校验失败")
    if not (profiles[0,-1]<.15<profiles[0].max()):
        raise RuntimeError("18 h 的剖面已不满足表面先达标示例，请重新选择示例时刻。")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    fig.subplots_adjust(left=.07,right=.98,bottom=.26,top=.82,wspace=.29)
    a,b=axes
    colors=[HEAT,WATER,GREEN]
    panel(a,"(a) 空间分布：表面达标不等于整体达标")
    a.axhspan(0,.15,facecolor=GREEN,alpha=.065)
    for tt,values,col in zip(times,profiles,colors):
        lab=f"{tt/3600:.0f} h" if tt!=tstar else rf"$t_*$ = {tstar/3600:.3f} h"
        a.plot(event.x*2,values,color=col,lw=1.8,label=lab)
    a.axhline(.15,color=GRAY,lw=1,ls="--")
    a.text(1.05,.157,r"$C_{\mathrm{crit}}=0.15$",fontsize=10,color=GRAY)
    a.scatter([2],[profiles[0,-1]],s=32,color=HEAT,zorder=4)
    a.annotate("表面已达标",xy=(2,profiles[0,-1]),xytext=(1.05,.033),fontsize=9,
               color=HEAT,arrowprops={"arrowstyle":"->","color":HEAT,"lw":1})
    a.set(xlabel="半径 r / cm",ylabel="干基含水率 C / (kg/kg)",xlim=(0,2.04),ylim=(0,profiles[0].max()*1.12))
    a.legend(frameon=False,fontsize=9,loc="upper right")
    panel(b,"(b) 时间判据：全空间最大值首次达到阈值")
    mask=trace_time>=36*3600
    hours=trace_time[mask]/3600
    b.plot(hours,max_c[mask],color=INK,lw=1.8,label="全空间最大含水率")
    b.plot(hours,field[mask,-1],color=WATER,lw=1.4,ls="--",label="表面含水率（对照）")
    b.axhspan(0,.15,facecolor=GREEN,alpha=.065)
    b.axhline(.15,color=GRAY,lw=1,ls="--")
    b.vlines(tstar/3600,0,.15,color=GREEN,lw=1,ls="--")
    b.scatter([tstar/3600],[.15],s=42,color=GREEN,zorder=4)
    b.annotate(rf"$t_*={tstar/3600:.6f}\ \mathrm{{h}}$",xy=(tstar/3600,.15),xytext=(43,.112),
               color=GREEN,fontsize=11,arrowprops={"arrowstyle":"->","color":GREEN,"lw":1})
    b.set(xlabel="时间 t / h",ylabel="干基含水率 C / (kg/kg)",xlim=(36,tstar/3600+1),ylim=(.035,.2))
    b.legend(frameon=False,fontsize=9,loc="upper right")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",color=LIGHT,lw=.6,alpha=.55)
    fig.text(.5,.125,r"$g(t)=\max_i C_i(t)-C_{\mathrm{crit}},\qquad t_*=\inf\{t:\ g(t)\leq0\}$",ha="center",fontsize=14)
    fig.text(.5,.035,"问题 3，641 个径向节点；连续事件定位。中心恰为本算例最大值所在位置，判据仍检查全部节点。",ha="center",fontsize=10,color=GRAY)
    save(fig,"15_drying_event_criterion")
    # 保存轻量核验摘要，避免把大量插值点重复写入正式结果表。
    record={"source":"problem3, 641-node continuous solution", "nodes":641,
            "drying_time_s":float(tstar),"reference_drying_time_s":summary["problem3_drying_time_s"],
            "event_residual":float(profiles[-1].max()-.15),
            "profile_times_h":(times/3600).tolist(),
            "first_profile_center_C":float(profiles[0,0]),"first_profile_surface_C":float(profiles[0,-1])}
    (OUT/"method_figures_validation.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"Event figure validated: t*={tstar:.6f} s, residual={record['event_residual']:.2e}")


def main():
    style()
    abstraction()
    control_volume()
    material_coordinates()
    event_criterion()
    print("Generated figures 12-15 (PNG + SVG).")


if __name__=="__main__":
    main()
