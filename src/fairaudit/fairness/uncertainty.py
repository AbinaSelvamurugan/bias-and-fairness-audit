"""Exploratory Wilson and row-bootstrap intervals; no participant IDs => no coverage guarantee."""
import numpy as np
from scipy.stats import norm
from fairaudit.fairness.group_analysis import group_rows, range_disparities


def wilson_interval(success, n, level=0.95):
    if n<=0: return [None,None]
    z=float(norm.ppf(0.5+level/2)); p=success/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [0.0 if success==0 else float(max(0,center-rad)),
            1.0 if success==n else float(min(1,center+rad))]


def add_wilson(rows,level=0.95):
    for r in rows:
        r['selection_rate_wilson']=wilson_interval(r['predicted_positive'],r['n'],level)
        r['tpr_wilson']=wilson_interval(r['tp'],r['positive_labels'],level)
        r['fpr_wilson']=wilson_interval(r['fp'],r['negative_labels'],level)
    return rows


def exploratory_bootstrap(y, scores, groups, threshold=0.5, repeats=150, seed=271828, level=0.95):
    y=np.asarray(y); scores=np.asarray(scores); groups=np.asarray(groups).astype(str)
    rng=np.random.default_rng(seed)
    strata=[np.where(groups==g)[0] for g in np.unique(groups)]
    draw={k:[] for k in ['demographic_parity_difference','equalized_odds_difference','equal_opportunity_difference']}
    for _ in range(repeats):
        idx=np.concatenate([rng.choice(ii,size=len(ii),replace=True) for ii in strata])
        d=range_disparities(group_rows(y[idx],scores[idx],groups[idx],threshold))
        for k in draw:
            if d[k] is not None: draw[k].append(d[k])
    out={}
    for key,vals in draw.items():
        out[key]={'point_estimate':range_disparities(group_rows(y,scores,groups,threshold))[key],
                  'exploratory_row_bootstrap_interval':list(map(float,np.quantile(vals,[(1-level)/2,1-(1-level)/2]))) if len(vals)>=max(10,repeats//2) else [None,None],
                  'successful_replicates':len(vals)}
    return {'status':'EXPLORATORY_NOT_CLUSTER_VALID', 'level':level,'replicates_requested':repeats,
       'sampling':'within observed groups, resample rows with replacement',
       'warning':'Participant/household IDs are unavailable; repeated observations or site clusters may invalidate coverage.',
       'metrics':out}
