"""独立核对鲁棒性扫描数量、状态、单调方向、基准和临界区间来源。"""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
import numpy as np
from robustness_analysis import fingerprint
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/robustness'

def main():
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    assert summary['source_sha256']==fingerprint()
    coarse=json.loads((DATA/'coarse.json').read_text(encoding='utf-8'))['rows']
    with (DATA/'all_cases.csv').open(encoding='utf-8-sig',newline='') as f:all_rows=list(csv.DictReader(f))
    assert len(coarse)==410 and len(all_rows)==summary['all_cases']
    assert Counter(r['status'] for r in all_rows)==Counter(summary['status_counts'])
    assert sum(r['role']=='guard' for r in coarse)==10
    assert all(r['status']=='invalid_input' for r in coarse if r['role']=='guard')
    assert all(r['drying_time_h'] is None for r in coarse if r['status']!='dry')
    monotonicity_warnings=[]
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
                monotonicity_warnings.append({'problem':q,'parameter':p,
                    'action':'Requires separate grid/closure investigation; do not silently impose monotonicity.'})
    formal=json.loads((ROOT/'intermediate/summary.json').read_text(encoding='utf-8'))
    baseline_errors={};step_errors={}
    for q in (3,4):
        rows=[r for r in summary['baseline_checks'] if r['problem']==q]
        fine=next(r for r in rows if r['nodes']==641)
        baseline_errors[str(q)]=abs(fine['drying_time_h']-formal[f'problem{q}_drying_time_h'])*3600
        assert baseline_errors[str(q)]<1
        a=next(r for r in rows if r['nodes']==321 and r['max_step_s']==60)
        b=next(r for r in rows if r['nodes']==321 and r['max_step_s']==30)
        step_errors[str(q)]=abs(a['drying_time_h']-b['drying_time_h'])*3600
    for t in summary['thresholds']:
        assert t['status']=='bracketed'
        fine=t['fine_641']
        assert fine.get('status')!='not_bracketed',t
        assert {fine['low_status'],fine['high_status']}=={'dry','deadline_exceeded'}
        assert fine['low']<fine['high']
        for side in ('low','high'):
            found=[r for r in all_rows if int(r['problem'])==t['problem'] and r['parameter']==t['parameter']
                   and int(r['nodes'])==641 and float(r['horizon_h'])==120
                   and abs(float(r['value'])-fine[side])<1e-12]
            assert found and found[-1]['status']==fine[side+'_status']
    low_grid=json.loads((DATA/'low_moisture_grid.json').read_text(encoding='utf-8'))
    assert low_grid['source_sha256']==fingerprint()
    assert len(low_grid['rows'])==18 and all(r['status']=='dry' for r in low_grid['rows'])
    grid_differences=[]
    for q in (3,4):
        for c in (0.,.025,.05):
            data={r['nodes']:r['drying_time_h'] for r in low_grid['rows'] if r['problem']==q and r['moisture']==c}
            grid_differences.append({'problem':q,'moisture':c,'time_321_h':data[321],
                'time_641_h':data[641],'time_1281_h':data[1281],
                'difference_641_to_1281_s':abs(data[641]-data[1281])*3600})
    result={'checks':'PASS_WITH_WARNINGS' if monotonicity_warnings else 'PASS',
            'monotonicity_warnings':monotonicity_warnings,'coarse_cases':len(coarse),'all_cases':len(all_rows),
            'status_counts':summary['status_counts'],'thresholds_checked':len(summary['thresholds']),
            'baseline_641_error_s':baseline_errors,'baseline_step_halving_difference_s':step_errors,
            'low_moisture_grid_check_cases':18,'low_moisture_grid_differences':grid_differences,
            'q4_successful_coarse_cases_using_radius_hold_after_72h':sum(r['problem']==4 and r['status']=='dry' and r['drying_time_h']>72 for r in coarse),
            'warning':'Finite-domain solver checks do not prove empirical validity or global robustness.'}
    (DATA/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
