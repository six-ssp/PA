"""只画真实工况；时间上限从结果读取，超时三角不表示实际烘干时间。"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl_robustness'))
import csv,json,hashlib
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from robustness_analysis import fingerprint

DATA=ROOT/'results/robustness'
FIGURES=ROOT/'figures'
COLORS={3:'#247f91',4:'#db7656'}

def load():
    meta=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    if meta['source_sha256']!=fingerprint():raise ValueError('鲁棒性缓存来源与当前求解代码不一致')
    rows=json.loads((DATA/'coarse.json').read_text(encoding='utf-8'))['rows']
    return rows,meta

def style():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'font.size':10.5,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#7a8b97',
        'text.color':'#243b4b','axes.labelcolor':'#243b4b','xtick.color':'#506675',
        'ytick.color':'#506675','axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})

def save(fig,name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(FIGURES/f'{name}.{ext}',dpi=220,bbox_inches='tight',
                    metadata={'CreationDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None)
    plt.close(fig)

def heatmaps(rows,meta):
    h=meta['horizon_h']
    old=json.loads((DATA/'reference_120h.json').read_text(encoding='utf-8'))['rows']
    ts=meta['joint_temperature_levels'];rs=meta['joint_radius_scales']
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.7),layout='constrained',sharey=True)
    cmap=mpl.colormaps['YlGnBu'].copy();cmap.set_bad('#e6e9eb')
    norm=Normalize(0,h)
    for ax,q in zip(axes,(3,4)):
        lut={(r['temperature_c'],r['radius_scale']):r for r in rows if r['role']=='joint' and r['problem']==q}
        z=np.array([[lut[(t,s)]['drying_time_h'] if lut[(t,s)]['status']=='dry' else np.nan for t in ts] for s in rs])
        ax.pcolormesh(np.array(ts+ [ts[-1]+10])-5,np.array(rs+[rs[-1]+.25])-.125,z,
                      cmap=cmap,norm=norm,shading='flat',edgecolors='white',linewidth=.65)
        for t in ts:
            for s in rs:
                r=lut[(t,s)]
                if r['status']=='dry':ax.plot(t,s,'.',color='white' if r['drying_time_h']>.58*h else '#466774',ms=2.8)
                elif r['status']=='deadline_exceeded':ax.plot(t,s,'x',color='#a0aab0',ms=3.5,mew=.75)
                else:ax.plot(t,s,'s',color='#b8223d',ms=5)
        # This boundary separates sampled cells, not a new PDE solution.
        good=np.isfinite(z).astype(float)
        if good.min()!=good.max():ax.contour(ts,rs,good,levels=[.5],colors='#2d4252',linestyles='--',linewidths=1)
        prev={(r['temperature_c'],r['radius_scale']):r for r in old if r['role']=='joint' and r['problem']==q}
        z0=np.array([[float(prev[t,s]['status']=='dry') for t in ts] for s in rs])
        if z0.min()!=z0.max():ax.contour(ts,rs,z0,levels=[.5],colors='#ac4265',linestyles=':',linewidths=1.5)
        ax.set(xlabel='稳定阶段环境温度 / °C',xticks=ts[::2],yticks=rs[::2])
        ax.set_title(f'({chr(94+q)}) 问题 {q}：{h:g} h 内达标范围',loc='left',pad=10)
    axes[0].set_ylabel('径向尺度倍数 $s_R$')
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,shrink=.82,pad=.025)
    cb.set_label('已达标工况的烘干时间 / h')
    handles=[Patch(facecolor='#e6e9eb',label=f'灰格 ×：{h:g} h 内未达标'),
             Line2D([],[],color='#2d4252',ls='--',label=f'{h:g} h 网格分类边界'),
             Line2D([],[],color='#ac4265',ls=':',label='原120 h 网格分类边界')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'23_robustness_feasibility_map')

def curves(rows,meta):
    h=meta['horizon_h']
    low_grid=json.loads((DATA/'low_moisture_grid.json').read_text(encoding='utf-8'))['rows']
    specs=[('temperature_c','稳定温度 / °C','linear'),('radius_scale','径向尺度倍数','log'),
           ('diffusivity_scale','扩散率倍数','log'),('moisture','环境等效含水率 / (kg/kg)','linear'),
           ('boundary_time_scale','边界时间尺度倍数','log'),('adverse','不利扰动强度 ε','linear')]
    fig,axes=plt.subplots(2,3,figsize=(13.8,8.3),layout='constrained')
    for ax,(p,label,scale),letter in zip(axes.flat,specs,'abcdef'):
        ax.axhline(h,color='#a75c64',lw=.8,ls='--')
        ax.axhline(120,color='#98a4ad',lw=.7,ls=':')
        for q in (3,4):
            g=sorted([r for r in rows if r['problem']==q and r['parameter']==p and r['role'] in ('oat','stress')],key=lambda r:r['value'])
            x=np.array([r['value'] for r in g]);y=np.array([r['drying_time_h'] if r['status']=='dry' else np.nan for r in g])
            ax.plot(x,y,color=COLORS[q],marker='o' if q==3 else 's',ms=3,lw=1.6,label=f'问题 {q}')
            fails=np.array([r['status']=='deadline_exceeded' for r in g])
            ax.scatter(x[fails],np.full(fails.sum(),h*(1.06 if q==3 else 1.11)),marker='^',color=COLORS[q],s=18)
            for bound in meta['thresholds']:
                if (bound['problem']==q and bound['parameter']==p and bound['status']=='bracketed'
                        and bound['fine_641'].get('status')!='not_bracketed'):
                    b=bound['fine_641']; mid=(b['low']+b['high'])/2
                    ax.axvline(mid,color=COLORS[q],lw=.75,ls=':',alpha=.8)
        ax.set(xlabel=label,ylabel='烘干时间 / h',ylim=(0,1.19*h),xscale=scale,yticks=np.linspace(0,h,6))
        ax.set_title(f'({letter}) '+('多因素同步不利变化' if p=='adverse' else label.split(' / ')[0]),loc='left')
        ax.grid(axis='y',color='#e6ecef',lw=.6);ax.set_axisbelow(True)
        if p=='moisture':
            ax.axvline(.15,color='#596673',lw=1,ls='-.')
            ax.text(.153,.12*h,'达标阈值',rotation=90,fontsize=8,color='#596673')
            for q in (3,4):
                fine=sorted([r for r in low_grid if r['problem']==q and r['nodes']==1281],key=lambda r:r['moisture'])
                ax.plot([r['moisture'] for r in fine],[r['drying_time_h'] for r in fine],
                        color=COLORS[q],ls='--',marker='D',mfc='white',ms=4,lw=1.1)
            ax.text(.008,.78*h,'低湿度段需加密复核',fontsize=9,color='#935265')
    handles=[Line2D([],[],color=COLORS[q],marker='o' if q==3 else 's',label=f'问题 {q}') for q in (3,4)]
    handles += [Line2D([],[],color='#5c6c78',marker='^',ls='none',label='顶端三角：超时，不表示具体时长'),
                Line2D([],[],color='#5c6c78',ls=':',label='竖虚线：641节点临界区间中点'),
                Line2D([],[],color='#5c6c78',ls='--',marker='D',mfc='white',label='低湿度空心菱形：1281节点复算'),
                Line2D([],[],color='#98a4ad',ls=':',label='灰色横线：原120 h窗口')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    save(fig,'24_robustness_stress_curves')

def main():
    rows,meta=load();style();heatmaps(rows,meta);curves(rows,meta)
    src={'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (DATA/'coarse.json',DATA/'summary.json',DATA/'all_cases.csv',DATA/'low_moisture_grid.json',DATA/'reference_120h.json')},
            'plot_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'figures':['23_robustness_feasibility_map','24_robustness_stress_curves'],
            'horizon_h':meta['horizon_h'],
            'notes':'Only real solves are colored. Gray cells and top triangles denote censoring at the recorded horizon, not solver failure.'}
    (DATA/'figure_sources.json').write_text(json.dumps(src,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Generated robustness figures 23 and 24.')

if __name__=='__main__':main()
