"""Auditable contingency-count metrics independent of a fairness package."""
import numpy as np
import pandas as pd
from fairaudit.models.evaluation import safe_div


def group_rows(y, scores, groups, threshold=0.5, min_n=100, min_condition=30):
    if pd.isna(groups).any(): raise ValueError('Group membership missing')
    y=np.asarray(y); s=np.asarray(scores,dtype=float); g=np.asarray(groups).astype(str)
    if len(y)!=len(s) or len(s)!=len(g): raise ValueError('Inputs have unequal lengths')
    if not np.isfinite(s).all(): raise ValueError('Missing or infinite predicted probabilities')
    if np.any(s<0) or np.any(s>1):
        raise ValueError('probabilities must be within [0, 1]')
    if not np.isin(y,[0,1]).all(): raise ValueError('Invalid/missing ground truth labels')
    yp=(s>=threshold).astype(int)
    out=[]
    for key in sorted(np.unique(g)):
        mask=g==key; yy=y[mask]; pp=yp[mask]
        tp=int(((yy==1)&(pp==1)).sum()); tn=int(((yy==0)&(pp==0)).sum())
        fp=int(((yy==0)&(pp==1)).sum()); fn=int(((yy==1)&(pp==0)).sum())
        n=len(yy); p=tp+fn; neg=tn+fp
        tpr=safe_div(tp,p); tnr=safe_div(tn,neg)
        out.append({'group':key, 'n':n, 'positive_labels':p, 'negative_labels':neg,
           'predicted_positive':tp+fp, 'predicted_negative':tn+fn,
           'tp':tp,'tn':tn,'fp':fp,'fn':fn,'label_prevalence':safe_div(p,n),
           'selection_rate':safe_div(tp+fp,n), 'accuracy':safe_div(tp+tn,n),
           'balanced_accuracy':(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
           'precision':safe_div(tp,tp+fp), 'tpr':tpr,'fnr':safe_div(fn,p),
           'fpr':safe_div(fp,neg),'specificity':tnr,
           'warning_small_n':n < min_n, 'warning_few_positives':p < min_condition,
           'warning_few_negatives':neg < min_condition})
    return out


def range_disparities(rows):
    """between-groups max-minus-min; min/max ratio when max>0 and fully defined."""
    result={}
    for name in ('selection_rate','tpr','fpr','fnr','accuracy'):
        x=[r[name] for r in rows]
        # All groups must have identifiable denominators for comparison.
        if len(x)<2 or any(a is None for a in x):
            result[name+'_difference']=None; result[name+'_ratio']=None
            continue
        result[name+'_difference']=float(max(x)-min(x))
        result[name+'_ratio']=float(min(x)/max(x)) if max(x)>0 else None
    result['demographic_parity_difference']=result['selection_rate_difference']
    result['demographic_parity_ratio']=result['selection_rate_ratio']
    result['equal_opportunity_difference']=result['tpr_difference']
    result['equalized_odds_difference']=(max(result['tpr_difference'],result['fpr_difference'])
       if result['tpr_difference'] is not None and result['fpr_difference'] is not None else None)
    result['equalized_odds_ratio']=(min(result['tpr_ratio'],result['fpr_ratio'])
       if result['tpr_ratio'] is not None and result['fpr_ratio'] is not None else None)
    result['aggregation']='between_groups (max-min; min/max), full set of reported groups'
    return result
