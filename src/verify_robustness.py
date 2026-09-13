"""独立核对鲁棒性扫描数量、状态、单调方向、基准和临界区间来源。"""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
import numpy as np
from robustness_analysis import HMAX,fingerprint
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/robustness'

def compare_windows(old, new):
    """逐工况配对，检查延长窗口不会让原先已达标的工况退回超时。"""
    cols=('problem','parameter','value','role','nodes','temperature_c','radius_scale',
          'diffusivity_scale','moisture','boundary_time_scale','max_step_s')
    key=lambda r:tuple(r[k] for k in cols)
    a={key(r):r for r in old}; b={key(r):r for r in new}
    assert len(a)==len(old) and len(b)==len(new) and a.keys()==b.keys()
    pairs=[]; dt=[]
    for k,r in a.items():
        s=b[k]
        assert r['horizon_h']==120. and s['horizon_h']==HMAX
        if r['status']=='dry':
            assert s['status']=='dry',k
            dt.append(abs(r['drying_time_h']-s['drying_time_h'])*3600)
        pairs.append({'problem':r['problem'],'before':r['status'],'after':s['status']})
    assert max(dt,default=0.)<1.,'原先达标时间变化超过1秒'
    out={}
    for q in (3,4):
        g=[r for r in pairs if r['problem']==q]
        out[str(q)]={'before':dict(Counter(r['before'] for r in g)),
                     'after':dict(Counter(r['after'] for r in g)),
                     'newly_dry':sum(r['before']=='deadline_exceeded' and r['after']=='dry' for r in g)}
    return {'old_horizon_h':120.,'new_horizon_h':HMAX,'paired_cases':len(pairs),
            'old_success_max_time_change_s':max(dt,default=0.),'by_problem':out}

def main():
    meta=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    assert meta['source_sha256']==fingerprint()
    assert meta['horizon_h']==HMAX
    cache=json.loads((DATA/'coarse.json').read_text(encoding='utf-8'))
    assert cache['source_sha256']==fingerprint()
    coarse=cache['rows']
    assert all(r['horizon_h']==HMAX for r in coarse)
    with (DATA/'all_cases.csv').open(encoding='utf-8-sig',newline='') as f:all_rows=list(csv.DictReader(f))
    assert len(coarse)==410 and len(all_rows)==meta['all_cases']
    assert Counter(r['status'] for r in all_rows)==Counter(meta['status_counts'])
    assert all(r['status'] in ('dry','deadline_exceeded','invalid_input','solver_failure','nonphysical') for r in all_rows)
    assert all(float(r['horizon_h']) in (HMAX,2*HMAX) for r in all_rows)
    assert sum(r['role']=='guard' for r in coarse)==10
    assert all(r['status']=='invalid_input' for r in coarse if r['role']=='guard')
    assert all(r['drying_time_h'] is None for r in coarse if r['status']!='dry')
    warns=[]
    for q in (3,4):
        grid=[r for r in coarse if r['role']=='joint' and r['problem']==q]
        assert len(grid)==99
        assert len({(r['temperature_c'],r['radius_scale']) for r in grid})==99
        for p in ('temperature_c','radius_scale','diffusivity_scale','moisture','boundary_time_scale','adverse'):
            g=sorted([r for r in coarse if r['problem']==q and r['parameter']==p and r['role'] in ('oat','stress')],key=lambda r:r['value'])
            if not all(r['status'] in ('dry','deadline_exceeded') for r in g):continue
            vals=[r['drying_time_h'] if r['status']=='dry' else float('inf') for r in g]
            if p in ('temperature_c','diffusivity_scale'):vals=vals[::-1]
            if not all(a<=b+1e-5 for a,b in zip(vals,vals[1:])):
                warns.append({'problem':q,'parameter':p,
                    'action':'Requires separate grid/closure investigation; do not silently impose monotonicity.'})
    formal=json.loads((ROOT/'intermediate/summary.json').read_text(encoding='utf-8'))
    base_err={};step_err={}
    for q in (3,4):
        rows=[r for r in meta['baseline_checks'] if r['problem']==q]
        fine=next(r for r in rows if r['nodes']==641)
        base_err[str(q)]=abs(fine['drying_time_h']-formal[f'problem{q}_drying_time_h'])*3600
        assert base_err[str(q)]<1
        a=next(r for r in rows if r['nodes']==321 and r['max_step_s']==60)
        b=next(r for r in rows if r['nodes']==321 and r['max_step_s']==30)
        step_err[str(q)]=abs(a['drying_time_h']-b['drying_time_h'])*3600
    for t in meta['thresholds']:
        if t['status']!='bracketed' or t['fine_641'].get('status')=='not_bracketed':
            warns.append({'problem':t['problem'],'parameter':t['parameter'],
                          'action':'临界区间未完成，不得当作已定位边界。'})
            continue
        fine=t['fine_641']
        assert {fine['low_status'],fine['high_status']}=={'dry','deadline_exceeded'}
        assert fine['low']<fine['high']
        for side in ('low','high'):
            found=[r for r in all_rows if int(r['problem'])==t['problem'] and r['parameter']==t['parameter']
                   and int(r['nodes'])==641 and float(r['horizon_h'])==HMAX
                   and abs(float(r['value'])-fine[side])<1e-12]
            assert found and found[-1]['status']==fine[side+'_status']
        assert t['extension_horizon_h']==2*HMAX
        assert t['extension_status'] in ('dry','deadline_exceeded','solver_failure','nonphysical')
        if t['extension_status']=='dry':
            assert HMAX<t['extension_drying_time_h']<=2*HMAX
        else:
            assert t['extension_drying_time_h'] is None
    low_grid=json.loads((DATA/'low_moisture_grid.json').read_text(encoding='utf-8'))
    assert low_grid['source_sha256']==fingerprint()
    assert low_grid['check_code_sha256']==hashlib.sha256((ROOT/'src/robustness_low_moisture_check.py').read_bytes()).hexdigest()
    assert len(low_grid['rows'])==18 and all(r['status']=='dry' for r in low_grid['rows'])
    grid_err=[]
    for q in (3,4):
        for c in (0.,.025,.05):
            data={r['nodes']:r['drying_time_h'] for r in low_grid['rows'] if r['problem']==q and r['moisture']==c}
            grid_err.append({'problem':q,'moisture':c,'time_321_h':data[321],
                'time_641_h':data[641],'time_1281_h':data[1281],
                'difference_641_to_1281_s':abs(data[641]-data[1281])*3600})
    old=json.loads((DATA/'reference_120h.json').read_text(encoding='utf-8'))['rows']
    bad=[{k:r[k] for k in ('problem','parameter','value','role','nodes','horizon_h','status','message')}
         for r in all_rows if r['status'] in ('solver_failure','nonphysical')]
    out={'checks':'PASS_WITH_WARNINGS' if warns or bad else 'PASS',
            'verify_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'horizon_h':HMAX,'window_comparison':compare_windows(old,coarse),
            'monotonicity_warnings':warns,'coarse_cases':len(coarse),'all_cases':len(all_rows),
            'status_counts':meta['status_counts'],
            'thresholds_checked':sum(t['status']=='bracketed' and t['fine_641'].get('status')!='not_bracketed' for t in meta['thresholds']),
            'extension_status_counts':dict(Counter(t.get('extension_status','not_run') for t in meta['thresholds'])),
            'main_window_status_counts':dict(Counter(r['status'] for r in all_rows if float(r['horizon_h'])==HMAX)),
            'numerical_anomalies':bad,
            'baseline_641_error_s':base_err,'baseline_step_halving_difference_s':step_err,
            'low_moisture_grid_check_cases':18,'low_moisture_grid_differences':grid_err,
            'q4_successful_coarse_cases_using_radius_hold_after_72h':sum(r['problem']==4 and r['status']=='dry' and r['drying_time_h']>72 for r in coarse),
            'warning':'Finite-domain solver checks do not prove empirical validity or global robustness.'}
    (DATA/'validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
