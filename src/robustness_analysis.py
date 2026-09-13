"""扩大输入范围并定位120小时达标边界；不修改四问的核心方程。

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import csv, hashlib, json, time
import numpy as np
from model import load_environment_xlsx, law_problem23, law_problem4, simulate
from problem_utils import load_radius_function

OUT=ROOT/'results/robustness'
HORIZON_H=120.0
TEMP_LEVELS=np.arange(10.,91.,10.)
RADIUS_LEVELS=np.arange(.5,3.001,.25)

@lru_cache(maxsize=1)
def inputs():
    return load_environment_xlsx(ROOT/'附件1.xlsx'),load_radius_function()[2]

def fingerprint():
    names=('src/model.py','src/problem_utils.py','src/robustness_analysis.py','附件1.xlsx','附件2.xlsx')
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}

def case(q, parameter='baseline', value=0., role='oat', nodes=161, **changes):
    env,_=inputs()
    item=dict(problem=q,parameter=parameter,value=float(value),role=role,nodes=nodes,
              temperature_c=float(env.stable_temperature_c),radius_scale=1.,diffusivity_scale=1.,
              moisture=float(env.stable_moisture),boundary_time_scale=1.,horizon_h=HORIZON_H,
              max_step_s=120. if nodes==161 else 60.)
    if parameter in ('temperature_c','radius_scale','diffusivity_scale','moisture','boundary_time_scale'):
        item[parameter]=float(value)
    elif parameter=='adverse':
        # A specified simultaneous-stress path, NOT a guaranteed worst-case box.
        item.update(temperature_c=env.stable_temperature_c-20*value,
                    radius_scale=1+.5*value,diffusivity_scale=1-.5*value,
                    moisture=env.stable_moisture*(1+value),boundary_time_scale=1+value)
    item.update(changes)
    return item

def input_error(item):
    fields=('temperature_c','radius_scale','diffusivity_scale','moisture','boundary_time_scale','horizon_h','max_step_s')
    if not all(np.isfinite(item[k]) for k in fields):return 'nonfinite_input'
    if item['temperature_c']<=-273.15:return 'nonpositive_absolute_temperature'
    if item['moisture']<0:return 'negative_environment_moisture'
    for k in ('radius_scale','diffusivity_scale','boundary_time_scale','horizon_h','max_step_s'):
        if item[k]<=0:return 'nonpositive_'+k
    if item['problem'] not in (3,4) or item['nodes']<5:return 'invalid_problem_or_grid'
    return None

def solve_case(item):
    start=time.perf_counter()
    row={**item,'status':'pending','drying_time_h':None,'end_max_moisture':None,
         'min_moisture':None,'max_moisture':None,'min_temperature':None,'max_temperature':None,
         'event_residual':None,'nfev':None,'nlu':None,'runtime_s':0.,'message':''}
    error=input_error(item)
    if error:
        row.update(status='invalid_input',message=error)
        return row
    environment,base_radius=inputs()
    scale=item['boundary_time_scale']
    env=replace(environment,time_s=environment.time_s*scale,
                stable_start_s=environment.stable_start_s*scale,
                stable_temperature_c=item['temperature_c'],stable_moisture=item['moisture'])
    law=law_problem23() if item['problem']==3 else law_problem4()
    base_d=law.diffusivity
    law=replace(law,diffusivity=lambda c,t:item['diffusivity_scale']*base_d(c,t))
    radius=(lambda t:np.full_like(np.asarray(t,dtype=float),.02*item['radius_scale'])) if item['problem']==3 else (
        lambda t:item['radius_scale']*np.asarray(base_radius(t)))
    try:
        result=simulate(env,law,item['horizon_h']*3600,radius=radius,nodes=item['nodes'],
                        max_step_s=item['max_step_s'],stop_at_dry=True)
        T,C=result.solution.y[:item['nodes']],result.solution.y[item['nodes']:]
        # Check every stored solver state; never hide a negative value via clipping.
        if not np.all(np.isfinite(result.solution.y)):
            row.update(status='nonphysical',message='nonfinite_state')
        else:
            row.update(end_max_moisture=float(C[:,-1].max()),min_moisture=float(C.min()),
                       max_moisture=float(C.max()),min_temperature=float(T.min()),max_temperature=float(T.max()),
                       nfev=int(result.solution.nfev),nlu=int(result.solution.nlu))
            tlo=min(28.,float(env.temperature_c.min()),item['temperature_c'])
            thi=max(28.,float(env.temperature_c.max()),item['temperature_c'])
            clo=min(2.55,float(env.moisture.min()),item['moisture'])
            chi=max(2.55,float(env.moisture.max()),item['moisture'])
            if C.min()<clo-2e-5 or C.max()>chi+2e-5 or T.min()<tlo-2e-3 or T.max()>thi+2e-3:
                row.update(status='nonphysical',message='state_outside_boundary_envelope')
            elif result.drying_time_s is None:
                row['status']='deadline_exceeded'
            else:
                residual=abs(row['end_max_moisture']-.15)
                row.update(status='dry' if residual<2e-5 else 'nonphysical',
                           drying_time_h=float(result.drying_time_s/3600),event_residual=residual)
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
    result=[]
    for q in (3,4):
        result.append(case(q,role='baseline'))
        for p,values in ranges.items():
            result.extend(case(q,p,float(v)) for v in values)
        result.extend(case(q,'adverse',float(v),role='stress') for v in np.linspace(0,1.8,19))
        result.extend(case(q,role='joint',temperature_c=float(t),radius_scale=float(s))
                      for t in TEMP_LEVELS for s in RADIUS_LEVELS)
        for p,v in [('temperature_c',-273.15),('radius_scale',0),('diffusivity_scale',0),
                    ('moisture',-.01),('boundary_time_scale',0)]:
            result.append(case(q,p,v,role='guard'))
    return result

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
    q,p,lo,hi=job
    records=[]
    def run(v,n=321,h=120.):
        r=solve_case(case(q,p,v,role='threshold' if n==321 else 'grid_check',nodes=n,horizon_h=h))
        records.append(r)
        if r['status'] not in ('dry','deadline_exceeded'):raise RuntimeError(f'Cannot bracket {q,p,v}: {r}')
        return r
    a,b=run(lo),run(hi)
    if a['status']==b['status']:
        return dict(problem=q,parameter=p,status='lost_bracket',low=lo,high=hi,records=records)
    # 10 bisections report an interval, not an unjustified exact critical value.
    for _ in range(10):
        mid=(lo+hi)/2; c=run(mid)
        if c['status']==a['status']:lo,a=mid,c
        else:hi,b=mid,c
    checks=[run(lo,641),run(hi,641)]
    # Fine-grid threshold may shift outside the tight 321 interval. Expand a
    # separate 641 bracket and refine it, preserving both grids in the record.
    fine_lo,fine_hi=lo,hi
    if checks[0]['status']==checks[1]['status']:
        pad=max((hi-lo)*32,1e-5)
        fine_lo=max(0.,lo-pad);fine_hi=hi+pad
        checks=[run(fine_lo,641),run(fine_hi,641)]
    if checks[0]['status']!=checks[1]['status']:
        fa,fb=checks
        for _ in range(5):
            m=(fine_lo+fine_hi)/2; fm=run(m,641)
            if fm['status']==fa['status']:fine_lo,fa=m,fm
            else:fine_hi,fb=m,fm
        fine=dict(low=fine_lo,high=fine_hi,low_status=fa['status'],high_status=fb['status'])
    else:fine=dict(low=fine_lo,high=fine_hi,status='not_bracketed')
    # A longer horizon distinguishes deadline loss from permanent failure.
    failed=a if a['status']=='deadline_exceeded' else b
    extension=run(failed['value'],321,240.)
    return dict(problem=q,parameter=p,status='bracketed',low=lo,high=hi,
                low_status=a['status'],high_status=b['status'],fine_641=fine,
                extension_240h_status=extension['status'],extension_drying_time_h=extension['drying_time_h'],records=records)

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
    original=fingerprint()
    if args.reuse_coarse:
        saved=json.loads((OUT/'coarse.json').read_text(encoding='utf-8'))
        if saved['source_sha256']!=original:raise ValueError('Coarse cache source changed')
        rows=saved['rows']
    else:
        todo=coarse_cases();rows=[]
        with (OUT/'progress.jsonl').open('w',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(solve_case,todo,chunksize=1):
                rows.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                if len(rows)%20==0 or len(rows)==len(todo):print(f'coarse {len(rows)}/{len(todo)}',flush=True)
        if fingerprint()!=original:raise RuntimeError('Sources changed during computation')
        (OUT/'coarse.json').write_text(json.dumps(dict(source_sha256=original,rows=rows),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        write_csv(OUT/'coarse.csv',rows)
    if args.coarse_only:return
    jobs=threshold_jobs(rows);thresholds=[]
    print(f'Refining {len(jobs)} observed threshold crossings',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(refine,jobs,chunksize=1):
            thresholds.append(item)
            print(json.dumps({k:v for k,v in item.items() if k!='records'},ensure_ascii=False),flush=True)
    checks=[]
    # Baseline discretization and integration-step checks are separate from failure tests.
    check_cases=[case(q,role='baseline_check',nodes=n,max_step_s=step)
                 for q in (3,4) for n,step in ((321,60.),(321,30.),(641,60.))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:checks=list(pool.map(solve_case,check_cases))
    if fingerprint()!=original:raise RuntimeError('Sources changed during computation')
    all_rows=rows+checks+[r for t in thresholds for r in t.pop('records')]
    write_csv(OUT/'all_cases.csv',all_rows)
    from collections import Counter
    summary=dict(source_sha256=original,horizon_h=HORIZON_H,coarse_nodes=161,coarse_max_step_s=120,
                 refined_nodes=321,refined_max_step_s=60,fine_nodes=641,
                 coarse_cases=len(rows),all_cases=len(all_rows),status_counts=dict(Counter(r['status'] for r in all_rows)),
                 coarse_status_counts=dict(Counter(r['status'] for r in rows)),thresholds=thresholds,
                 baseline_checks=checks,joint_temperature_levels=TEMP_LEVELS.tolist(),joint_radius_scales=RADIUS_LEVELS.tolist(),
                 interpretation='Deterministic finite-domain stress tests; deadline loss is not solver failure; no probability or global safety guarantee.',
                 adverse_path='T=T0-20e; sR=1+0.5e; sD=1-0.5e; Cenv=C0(1+e); st=1+e; 0<=e<=1.8',
                 calibration_warning='No validity ranges for empirical properties supplied; broad scans are mathematical extrapolations, not validated process recommendations.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary['status_counts'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
