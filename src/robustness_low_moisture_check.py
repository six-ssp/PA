"""独立复查低环境含水率下的非单调现象，不覆盖主鲁棒性扫描。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,hashlib
from robustness_analysis import case,solve_case,fingerprint,write_csv
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/robustness'

def main():
    src0=fingerprint()
    cases=[case(q,'moisture',c,role='low_moisture_grid',nodes=n)
           for q in (3,4) for c in (0.,.025,.05) for n in (321,641,1281)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(solve_case,cases):
            rows.append(row)
            print(row['problem'],row['moisture'],row['nodes'],row['status'],row['drying_time_h'],flush=True)
    assert src0==fingerprint()
    assert all(r['status']=='dry' for r in rows)
    write_csv(DATA/'low_moisture_grid.csv',rows)
    meta={'source_sha256':src0,'check_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':rows,'purpose':'Diagnose nonmonotonicity in dry-boundary coarse-grid scan; not independent experimental validation.'}
    (DATA/'low_moisture_grid.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
