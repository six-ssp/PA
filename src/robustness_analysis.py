"""扩大输入范围并定位500小时达标边界；不修改四问的核心方程。

没有事件是有限时域内未达标，不是求解器失败。粗扫161节点/120秒最大步长，
临界区321节点/60秒，并用641节点复查临界两端。所有测试均为确定性工况。
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
import csv, hashlib, json, time
import numpy as np
from model import load_environment_xlsx, law_problem23, law_problem4, simulate
from problem_utils import load_radius_function

OUT=ROOT/'results/robustness'
HMAX=500.0  # 测试窗口/h；积分步长仍以秒计，不随窗口一起放大。
TS=np.arange(10.,91.,10.)
RS=np.arange(.5,3.001,.25)

@lru_cache(maxsize=1)
def inputs():
    return load_environment_xlsx(ROOT/'附件1.xlsx'),load_radius_function()[2]

def fingerprint():
    names=('src/model.py','src/problem_utils.py','src/robustness_analysis.py','附件1.xlsx','附件2.xlsx',
           'results/robustness/reference_120h.json','results/robustness/reference_120h_summary.json')
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}

def case(q, p='baseline', v=0., role='oat', nodes=161, **changes):
    """q为题号，p为因素，v为取值；cfg保存一个工况，输出字段保留单位。"""
    env,_=inputs()
    cfg=dict(problem=q,parameter=p,value=float(v),role=role,nodes=nodes,
              temperature_c=float(env.stable_temperature_c),radius_scale=1.,diffusivity_scale=1.,
              moisture=float(env.stable_moisture),boundary_time_scale=1.,horizon_h=HMAX,
              max_step_s=120. if nodes==161 else 60.)
    if p in ('temperature_c','radius_scale','diffusivity_scale','moisture','boundary_time_scale'):
        cfg[p]=float(v)
    elif p=='adverse':
        # 多因素同时变化的一条路径，不代表整个参数空间的最坏情况。
        cfg.update(temperature_c=env.stable_temperature_c-20*v,
                    radius_scale=1+.5*v,diffusivity_scale=1-.5*v,
                    moisture=env.stable_moisture*(1+v),boundary_time_scale=1+v)
    cfg.update(changes)
    return cfg

def input_error(cfg):
    fields=('temperature_c','radius_scale','diffusivity_scale','moisture','boundary_time_scale','horizon_h','max_step_s')
    if not all(np.isfinite(cfg[k]) for k in fields):return 'nonfinite_input'
    if cfg['temperature_c']<=-273.15:return 'nonpositive_absolute_temperature'
    if cfg['moisture']<0:return 'negative_environment_moisture'
    for k in ('radius_scale','diffusivity_scale','boundary_time_scale','horizon_h','max_step_s'):
        if cfg[k]<=0:return 'nonpositive_'+k
    if cfg['problem'] not in (3,4) or cfg['nodes']<5:return 'invalid_problem_or_grid'
    return None

def solve_case(cfg):
    """env0/r0为原始环境/半径；st为时间倍数；T/C为温度/含水率。"""
    start=time.perf_counter()
    row={**cfg,'status':'pending','drying_time_h':None,'end_max_moisture':None,
         'min_moisture':None,'max_moisture':None,'min_temperature':None,'max_temperature':None,
         'event_residual':None,'nfev':None,'nlu':None,'runtime_s':0.,'message':''}
    error=input_error(cfg)
    if error:
        row.update(status='invalid_input',message=error)
        return row
    env0,r0=inputs()
    st=cfg['boundary_time_scale']
    env=replace(env0,time_s=env0.time_s*st,
                stable_start_s=env0.stable_start_s*st,
                stable_temperature_c=cfg['temperature_c'],stable_moisture=cfg['moisture'])
    law=law_problem23() if cfg['problem']==3 else law_problem4()
    d0=law.diffusivity
    law=replace(law,diffusivity=lambda c,t:cfg['diffusivity_scale']*d0(c,t))
    radius=(lambda t:np.full_like(np.asarray(t,dtype=float),.02*cfg['radius_scale'])) if cfg['problem']==3 else (
        lambda t:cfg['radius_scale']*np.asarray(r0(t)))
    try:
        res=simulate(env,law,cfg['horizon_h']*3600,radius=radius,nodes=cfg['nodes'],
                        max_step_s=cfg['max_step_s'],stop_at_dry=True)
        T,C=res.solution.y[:cfg['nodes']],res.solution.y[cfg['nodes']:]
        # 检查全部已保存状态；不把负值截成零来隐藏问题。
        if not np.all(np.isfinite(res.solution.y)):
            row.update(status='nonphysical',message='nonfinite_state')
        else:
            row.update(end_max_moisture=float(C[:,-1].max()),min_moisture=float(C.min()),
                       max_moisture=float(C.max()),min_temperature=float(T.min()),max_temperature=float(T.max()),
                       nfev=int(res.solution.nfev),nlu=int(res.solution.nlu))
            tlo=min(28.,float(env.temperature_c.min()),cfg['temperature_c'])
            thi=max(28.,float(env.temperature_c.max()),cfg['temperature_c'])
            clo=min(2.55,float(env.moisture.min()),cfg['moisture'])
            chi=max(2.55,float(env.moisture.max()),cfg['moisture'])
            if C.min()<clo-2e-5 or C.max()>chi+2e-5 or T.min()<tlo-2e-3 or T.max()>thi+2e-3:
                row.update(status='nonphysical',message='state_outside_boundary_envelope')
            elif res.drying_time_s is None:
                row['status']='deadline_exceeded'
            else:
                residual=abs(row['end_max_moisture']-.15)
                row.update(status='dry' if residual<2e-5 else 'nonphysical',
                           drying_time_h=float(res.drying_time_s/3600),event_residual=residual)
    except Exception as exc:
        row.update(status='solver_failure',message=f'{type(exc).__name__}: {exc}')
    row['runtime_s']=time.perf_counter()-start
    return row

def coarse_cases():
    ranges={'temperature_c':np.arange(0.,100.01,5.),
            'radius_scale':np.geomspace(.25,4.,17),
            'diffusivity_scale':np.geomspace(.02,5.,17),
            'moisture':[0.,.025,.05,.075,.1,.12,.13,.14,.145,.148,.149,.15,.155,.17,.2,.3],
            'boundary_time_scale':[.1,.25,.5,1.,2.,4.,8.,16.,32.,64.]}
    res=[]
    for q in (3,4):
        res.append(case(q,role='baseline'))
        for p,values in ranges.items():
            res.extend(case(q,p,float(v)) for v in values)
        res.extend(case(q,'adverse',float(v),role='stress') for v in np.linspace(0,1.8,19))
        res.extend(case(q,role='joint',temperature_c=float(t),radius_scale=float(s))
                      for t in TS for s in RS)
        for p,v in [('temperature_c',-273.15),('radius_scale',0),('diffusivity_scale',0),
                    ('moisture',-.01),('boundary_time_scale',0)]:
            res.append(case(q,p,v,role='guard'))
    return res

def threshold_jobs(rows):
    jobs=[]
    # Only bracket crossings actually observed; do not extrapolate beyond scan.
    for q in (3,4):
        for p in ('temperature_c','radius_scale','diffusivity_scale','moisture','boundary_time_scale','adverse'):
            group=sorted([r for r in rows if r['problem']==q and r['parameter']==p and r['role'] in ('oat','stress')],key=lambda r:r['value'])
            changes=[]
            for a,b in zip(group,group[1:]):
                if {a['status'],b['status']}=={'dry','deadline_exceeded'}:changes.append((a,b))
            for a,b in changes:jobs.append((q,p,a['value'],b['value']))
    return jobs

def refine(job):
    """单个临界工况异常只记为未完成，不让整批测试丢失。"""
    runs=[]
    try:
        return _refine(job,runs)
    except RuntimeError as exc:
        q,p,lo,hi=job
        return dict(problem=q,parameter=p,status='refinement_failed',low=lo,high=hi,
                    message=str(exc),records=runs)


def _refine(job,runs):
    q,p,lo,hi=job
    def run(v,n=321,h=HMAX):
        r=solve_case(case(q,p,v,role='threshold' if n==321 else 'grid_check',nodes=n,horizon_h=h))
        runs.append(r)
        if r['status'] not in ('dry','deadline_exceeded'):raise RuntimeError(f'Cannot bracket {q,p,v}: {r}')
        return r
    a,b=run(lo),run(hi)
    if a['status']==b['status']:
        return dict(problem=q,parameter=p,status='lost_bracket',low=lo,high=hi,records=runs)
    # 10 bisections report an interval, not an unjustified exact critical value.
    for _ in range(10):
        mid=(lo+hi)/2; c=run(mid)
        if c['status']==a['status']:lo,a=mid,c
        else:hi,b=mid,c
    checks=[run(lo,641),run(hi,641)]
    # Fine-grid threshold may shift outside the tight 321 interval. Expand a
    # separate 641 bracket and refine it, preserving both grids in the record.
    lo2,hi2=lo,hi
    if checks[0]['status']==checks[1]['status']:
        pad=max((hi-lo)*32,1e-5)
        lo2=max(0.,lo-pad);hi2=hi+pad
        checks=[run(lo2,641),run(hi2,641)]
    if checks[0]['status']!=checks[1]['status']:
        fa,fb=checks
        for _ in range(5):
            m=(lo2+hi2)/2; fm=run(m,641)
            if fm['status']==fa['status']:lo2,fa=m,fm
            else:hi2,fb=m,fm
        fine=dict(low=lo2,high=hi2,low_status=fa['status'],high_status=fb['status'])
    else:fine=dict(low=lo2,high=hi2,status='not_bracketed')
    # A longer horizon distinguishes deadline loss from permanent failure.
    failed=a if a['status']=='deadline_exceeded' else b
    ext=solve_case(case(q,p,failed['value'],role='extension',nodes=321,horizon_h=2*HMAX))
    runs.append(ext)  # 辅助延时可能异常，保留原始状态，不中断主窗口的临界定位。
    return dict(problem=q,parameter=p,status='bracketed',low=lo,high=hi,
                low_status=a['status'],high_status=b['status'],fine_641=fine,
                extension_horizon_h=2*HMAX,extension_status=ext['status'],
                extension_drying_time_h=ext['drying_time_h'],records=runs)

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--reuse-coarse',action='store_true')
    parser.add_argument('--coarse-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    src0=fingerprint()
    if args.reuse_coarse:
        saved=json.loads((OUT/'coarse.json').read_text(encoding='utf-8'))
        if saved['source_sha256']!=src0:raise ValueError('Coarse cache source changed')
        rows=saved['rows']
    else:
        todo=coarse_cases();rows=[]
        with (OUT/'progress.jsonl').open('w',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(solve_case,todo,chunksize=1):
                rows.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                if len(rows)%20==0 or len(rows)==len(todo):print(f'coarse {len(rows)}/{len(todo)}',flush=True)
        if fingerprint()!=src0:raise RuntimeError('Sources changed during computation')
        (OUT/'coarse.json').write_text(json.dumps(dict(source_sha256=src0,rows=rows),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        write_csv(OUT/'coarse.csv',rows)
    if args.coarse_only:return
    jobs=threshold_jobs(rows);bounds=[]
    print(f'Refining {len(jobs)} observed threshold crossings',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(refine,job):job for job in jobs}
        from concurrent.futures import as_completed
        for task in as_completed(tasks):
            cfg=task.result()
            bounds.append(cfg)
            # 每完成一组就落盘，异常/中断时也不会丢掉已完成的细网格结果。
            (OUT/'threshold_progress.json').write_text(json.dumps(
                dict(source_sha256=src0,thresholds=bounds),ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in cfg.items() if k!='records'},ensure_ascii=False),flush=True)
    bounds.sort(key=lambda r:(r['problem'],r['parameter']))
    checks=[]
    # Baseline discretization and integration-step checks are separate from failure tests.
    base_cases=[case(q,role='baseline_check',nodes=n,max_step_s=step)
                 for q in (3,4) for n,step in ((321,60.),(321,30.),(641,60.))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:checks=list(pool.map(solve_case,base_cases))
    if fingerprint()!=src0:raise RuntimeError('Sources changed during computation')
    all_rows=rows+checks+[r for t in bounds for r in t.pop('records')]
    write_csv(OUT/'all_cases.csv',all_rows)
    from collections import Counter
    meta=dict(source_sha256=src0,horizon_h=HMAX,coarse_nodes=161,coarse_max_step_s=120,
                 refined_nodes=321,refined_max_step_s=60,fine_nodes=641,
                 coarse_cases=len(rows),all_cases=len(all_rows),status_counts=dict(Counter(r['status'] for r in all_rows)),
                 coarse_status_counts=dict(Counter(r['status'] for r in rows)),thresholds=bounds,
                 baseline_checks=checks,joint_temperature_levels=TS.tolist(),joint_radius_scales=RS.tolist(),
                 interpretation='Deterministic finite-domain stress tests; deadline loss is not solver failure; no probability or global safety guarantee.',
                 adverse_path='T=T0-20e; sR=1+0.5e; sD=1-0.5e; Cenv=C0(1+e); st=1+e; 0<=e<=1.8',
                 calibration_warning='No validity ranges for empirical properties supplied; broad scans are mathematical extrapolations, not validated process recommendations.')
    (OUT/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(meta['status_counts'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
