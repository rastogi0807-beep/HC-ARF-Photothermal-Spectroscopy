"""Run independent synthetic episodes; deterministic action-keyed noise streams."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from twin_core import simulate

ROOT=Path(__file__).resolve().parents[1]
POLICIES=['Goal agent','Fixed scan','Global information','Phase only']
SETTINGS=['Double noise','Extra thermal pole','Spectral interferent',
          'Line strength +10%','Concentration changes','Heavy tails']

def work(args): return simulate(*args)

def main(cohort):
    if cohort=='development': jobs=[(s,p,'Nominal') for s in range(31000,31016) for p in POLICIES]
    elif cohort=='calibration': jobs=[(s,p,'Nominal') for s in range(41000,41128) for p in POLICIES]
    elif cohort=='primary': jobs=[(s,p,'Nominal') for s in range(51000,51240) for p in POLICIES]
    elif cohort=='challenges': jobs=[(s,p,z) for s in range(71000,71096) for p in POLICIES[:2] for z in SETTINGS]
    else: raise ValueError(cohort)
    start=time.perf_counter(); result=[]
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(work,jobs,chunksize=4)):
            result.append(r)
            if (i+1)%64==0: print(f'{cohort}: {i+1}/{len(jobs)}',flush=True)
    out=ROOT/'data'/f'{cohort}_episodes.json'
    out.write_text(json.dumps(result,separators=(',',':')))
    timing=dict(cohort=cohort,episodes=len(result),seconds=time.perf_counter()-start,
                workers=4,BLAS_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                source_SHA256={n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest()
                               for n in ['twin_core.py','run_study.py']})
    (ROOT/'data'/f'{cohort}_timing.json').write_text(json.dumps(timing,indent=2))
    print(json.dumps(timing,indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1])
