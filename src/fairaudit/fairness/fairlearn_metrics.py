"""Official Fairlearn MetricFrame and built-in metrics (optional integration)."""
import math
import numpy as np
from fairaudit.fairness.group_analysis import group_rows, range_disparities


def _guard(y,p):
    y=np.asarray(y)
    if len(y)==0: return float('nan')
    return None


def metricframe_audit(y, scores, groups, threshold=0.5):
    from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
        false_positive_rate, false_negative_rate, true_negative_rate,
        demographic_parity_difference, demographic_parity_ratio,
        equalized_odds_difference, equalized_odds_ratio, equal_opportunity_difference)
    from sklearn.metrics import accuracy_score, precision_score
    y=np.asarray(y).astype(int); yp=(np.asarray(scores)>=threshold).astype(int)
    g=np.asarray(groups).astype(str)
    def guarded(metric):
        def m(a,b):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                v=float(metric(a,b))
            return v if np.isfinite(v) else np.nan
        return m
    metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score,
             'precision': guarded(lambda a,b: precision_score(a,b,zero_division=np.nan)),
             'tpr': guarded(true_positive_rate), 'fpr': guarded(false_positive_rate),
             'fnr': guarded(false_negative_rate), 'specificity': guarded(true_negative_rate)}
    mf=MetricFrame(metrics=metrics,y_true=y,y_pred=yp,sensitive_features=g)
    builtins={}
    for key, fn in {'demographic_parity_difference':demographic_parity_difference,
        'demographic_parity_ratio':demographic_parity_ratio,
        'equalized_odds_difference':equalized_odds_difference,
        'equalized_odds_ratio':equalized_odds_ratio,
        'equal_opportunity_difference':equal_opportunity_difference}.items():
        try:
            val=float(fn(y,yp,sensitive_features=g,method='between_groups'))
            builtins[key]=val if np.isfinite(val) else None
        except (ValueError,ZeroDivisionError) as exc:
            builtins[key]=None
    result={}
    for group, row in mf.by_group.iterrows():
        result[str(group)]={k:(float(v) if np.isfinite(v) else None) for k,v in row.items()}
    # Avoid returning package-specific invalid ratios when manual denominators are 0/undefined.
    guards=range_disparities(group_rows(y,scores,g,threshold))
    for key in ('demographic_parity_ratio','equalized_odds_ratio'):
        if guards[key] is None: builtins[key]=None
    return {'status':'fairlearn_metricframe_ran', 'by_group':result,
            'metricframe_between_group_difference':_serialize_series(mf.difference(method='between_groups')),
            'metricframe_between_group_ratio':_serialize_series(mf.ratio(method='between_groups')),
            'fairlearn_builtins': builtins}


def _serialize_series(s):
    return {str(k):(float(v) if np.isfinite(v) else None) for k,v in s.items()}


def optional_fairlearn(y, scores, groups, threshold):
    try: return metricframe_audit(y,scores,groups,threshold)
    except ImportError as exc:
        return {'status':'fairlearn_not_installed', 'detail':str(exc)}
