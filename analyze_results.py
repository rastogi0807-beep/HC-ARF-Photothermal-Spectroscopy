from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from twin_core import BUDGET, TARGET_LOG

ROOT=Path(__file__).resolve().parents[1]
POLICIES=['Goal agent','Fixed scan','Global information','Phase only']

def load(name):return json.loads((ROOT/'data'/f'{name}_episodes.json').read_text())

def quantile(scores,level=.95):
    ordered=np.sort(scores);k=int(np.ceil((len(ordered)+1)*level))
    return float(ordered[k-1]) if k<=len(ordered) else float('inf')

def wilson(k,n):
    if not n:return [None,None]
    z=norm.ppf(.975);d=1+z*z/n;c=(k/n+z*z/(2*n))/d
    h=z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]

def calibrated(calibration):
    ans={}
    for p in POLICIES:
        es=[e for e in calibration if e['policy']==p]
        path=[max(s['log_error']/s['sd_log_c'] for s in e['states']) for e in es]
        terminal=[e['states'][-1]['log_error']/e['states'][-1]['sd_log_c'] for e in es]
        ans[p]=dict(n=len(es),rank=int(np.ceil((len(es)+1)*.95)),
                    q_path=quantile(path),q_terminal=quantile(terminal),
                    scores_path=path,scores_terminal=terminal,
                    levels={str(v):quantile(path,v) for v in [.8,.9,.95,.975,.99]})
    return ans

def row(e,cal):
    q=cal['q_path'];ss=e['states'];last=ss[-1]
    eligible=[s for s in ss if not s['flagged'] and q*s['sd_log_c']<=TARGET_LOG]
    stopped=eligible[0] if eligible else last
    terminal_error=np.exp(last['theta'][0])-last['true_c']
    terminal_z=last['log_error']/last['sd_log_c']
    max_z=max(s['log_error']/s['sd_log_c'] for s in ss)
    prior_ss=[s for s in ss if not s['flagged'] and 1.96*s['sd_log_c']<=TARGET_LOG]
    raw_stop=prior_ss[0] if prior_ss else last
    terminal_ss=[s for s in ss if not s['flagged'] and cal['q_terminal']*s['sd_log_c']<=TARGET_LOG]
    terminal_stop=terminal_ss[0] if terminal_ss else last
    idx=stopped['index'];used=e['records'][:idx+1]
    return dict(seed=e['seed'],policy=e['policy'],setting=e['setting'],
                true_c=last['true_c'],estimate=np.exp(last['theta'][0]),error=terminal_error,
                relative_error=terminal_error/last['true_c'],
                terminal_covered=terminal_z<=q,path_covered=max_z<=q,
                terminal_z=terminal_z,max_z=max_z,terminal_raw_covered=terminal_z<=1.96,
                path_raw_covered=max_z<=1.96,
                path_terminal_cal_covered=max_z<=cal['q_terminal'],
                report=bool(eligible),stop_cost=stopped['spent'] if eligible else BUDGET,
                stop_covered=stopped['log_error']<=q*stopped['sd_log_c'],
                stop_estimate=np.exp(stopped['theta'][0]),stop_true_c=stopped['true_c'],
                stop_upper_radius=np.expm1(q*stopped['sd_log_c']),
                report_miss=bool(eligible) and stopped['log_error']>q*stopped['sd_log_c'],
                raw_report=bool(prior_ss),raw_stop_cost=raw_stop['spent'] if prior_ss else BUDGET,
                raw_report_miss=bool(prior_ss) and raw_stop['log_error']>1.96*raw_stop['sd_log_c'],
                terminal_cal_report=bool(terminal_ss),
                terminal_cal_report_miss=bool(terminal_ss) and terminal_stop['log_error']>cal['q_terminal']*terminal_stop['sd_log_c'],
                final_flag=last['flagged'],records=len(e['records']),
                terminal_phase=sum(r['mode']=='P' for r in e['records']),
                terminal_transmission=sum(r['mode']=='D' for r in e['records']),
                used_phase=sum(r['mode']=='P' for r in used),
                used_transmission=sum(r['mode']=='D' for r in used),
                sd_log_c=last['sd_log_c'],q=q,
                runtime=e['elapsed'],c_initial=float(np.exp(e['truth'][0])))

def summary(frame):
    n=len(frame);rng=np.random.default_rng(91821)
    ids=rng.integers(0,n,size=(2000,n));errors=frame.error.to_numpy()
    rmse=np.sqrt(np.mean(errors**2));boot=np.sqrt(np.mean(errors[ids]**2,axis=1))
    ans=dict(n=n,RMSE_ppm=float(rmse),RMSE_CI=np.quantile(boot,[.025,.975]).tolist(),
             median_absolute_error_ppm=float(np.median(np.abs(errors))),
             median_absolute_relative_error=float(np.median(np.abs(frame.relative_error))),
             mean_stop_cost=float(frame.stop_cost.mean()),median_stop_cost=float(frame.stop_cost.median()),
             mean_stop_ms=float(frame.stop_cost.mean()*40),
             mean_terminal_transmission=float(frame.terminal_transmission.mean()),
             mean_terminal_phase=float(frame.terminal_phase.mean()),
             median_full_trajectory_compute_ms=float(frame.runtime.median()*1000),
             median_upper_radius=float(np.median(np.expm1(frame.q*frame.sd_log_c))))
    for col in ['terminal_covered','path_covered','terminal_raw_covered','path_raw_covered',
                'path_terminal_cal_covered','report','report_miss','final_flag',
                'raw_report','raw_report_miss','terminal_cal_report','terminal_cal_report_miss']:
        k=int(frame[col].sum());ans[col]=dict(count=k,total=n,rate=k/n,CI=wilson(k,n))
    emitted=frame[frame.report]
    ans['reported_band_coverage']=dict(count=int(emitted.stop_covered.sum()),total=len(emitted),
                                      CI=wilson(int(emitted.stop_covered.sum()),len(emitted)))
    return ans

def main():
    cal=calibrated(load('calibration'))
    (ROOT/'data/calibration_thresholds.json').write_text(json.dumps(cal,indent=2))
    rows=[row(e,cal[e['policy']]) for name in ['primary','challenges'] for e in load(name)]
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'data/episode_metrics.csv',index=False)
    primary=frame[frame.setting=='Nominal']
    result=dict(calibration={p:{k:v for k,v in v.items() if not k.startswith('scores_')} for p,v in cal.items()},
                primary={p:summary(primary[primary.policy==p]) for p in POLICIES},
                challenges={},paired={},concentration_bins={})
    for setting in sorted(set(frame.setting)-{'Nominal'}):
        result['challenges'][setting]={p:summary(frame[(frame.setting==setting)&(frame.policy==p)])
                                       for p in POLICIES[:2]}
    goal=primary[primary.policy=='Goal agent'].sort_values('seed');rng=np.random.default_rng(280192)
    ids=rng.integers(0,len(goal),size=(2000,len(goal)))
    for p in POLICIES[1:]:
        other=primary[primary.policy==p].sort_values('seed');assert np.array_equal(goal.seed,other.seed)
        a=goal.error.to_numpy();b=other.error.to_numpy()
        diff=np.sqrt(np.mean(a[ids]**2,axis=1))-np.sqrt(np.mean(b[ids]**2,axis=1))
        benefit=1-np.sqrt(np.mean(a[ids]**2,axis=1))/np.sqrt(np.mean(b[ids]**2,axis=1))
        c=goal.stop_cost.to_numpy()-other.stop_cost.to_numpy()
        result['paired'][p]=dict(RMSE_difference_ppm=float(np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b))),
             RMSE_difference_CI=np.quantile(diff,[.025,.975]).tolist(),
             RMSE_relative_reduction=float(1-np.sqrt(np.mean(a*a))/np.sqrt(np.mean(b*b))),
             RMSE_relative_reduction_CI=np.quantile(benefit,[.025,.975]).tolist(),
             mean_cost_difference=float(c.mean()),mean_cost_difference_CI=np.quantile(c[ids].mean(axis=1),[.025,.975]).tolist())
    for p in POLICIES:
        result['concentration_bins'][p]={}
        g=primary[primary.policy==p]
        for name,lo,hi in [('1-3 ppm',1,3),('3-8 ppm',3,8),('8-20 ppm',8,20.0001)]:
            result['concentration_bins'][p][name]=summary(g[(g.c_initial>=lo)&(g.c_initial<hi)])
    result['totals']={name:len(load(name)) for name in ['development','calibration','primary','challenges']}
    (ROOT/'data/results_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'calibration':result['calibration'],'primary':result['primary'],
                      'paired':result['paired'],'totals':result['totals']},indent=2))

if __name__=='__main__':main()
