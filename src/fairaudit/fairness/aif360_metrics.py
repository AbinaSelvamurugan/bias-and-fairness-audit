"""AIF360's documented scikit-learn functions for binary plan-type comparison."""
import numpy as np
import pandas as pd


def optional_aif360(y, scores, groups, threshold=0.5):
    try:
        from aif360.sklearn.metrics import (statistical_parity_difference,
            disparate_impact_ratio, average_odds_difference, equal_opportunity_difference)
    except ImportError as exc:
        return {'status':'aif360_not_installed','detail':str(exc)}
    g=np.asarray(groups).astype(str)
    if set(g)!={'0_non_individual','1_individual'}:
        return {'status':'NOT_APPLICABLE','detail':'AIF360 pairwise protocol restricted to full binary plan_type attribute'}
    yp=(np.asarray(scores)>=threshold).astype(int)
    yt=pd.Series(np.asarray(y,dtype=int))
    priv='0_non_individual'
    # Prevent AIF360 zero_division=0 from concealing undefined denominators.
    r0=yp[g==priv].mean(); r1=yp[g!='0_non_individual'].mean()
    metrics={
       'statistical_parity_difference': statistical_parity_difference,
       'disparate_impact_ratio': disparate_impact_ratio,
       'average_odds_difference': average_odds_difference,
       'equal_opportunity_difference': equal_opportunity_difference,
    }
    out={}
    for k,f in metrics.items():
        if k=='disparate_impact_ratio' and r0==0: out[k]=None; continue
        try:
            v=float(f(yt, yp, prot_attr=g, priv_group=priv, pos_label=1))
            out[k]=v if np.isfinite(v) else None
        except (ValueError,ZeroDivisionError) as exc:
            out[k]=None
    return {'status':'aif360_metrics_ran', 'comparison':'1_individual minus/over 0_non_individual',
            'reference_group':priv, 'n_compared':len(g), 'metrics':out,
            'note':'AIF360 average odds is SIGNED average of TPR and FPR pairwise differences; not Fairlearn max-absolute equalized odds'}
