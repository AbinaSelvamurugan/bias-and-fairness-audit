"""Descriptive threshold sweep on fixed scores, never used for model choice."""
from fairaudit.models.evaluation import performance
from fairaudit.fairness.group_analysis import group_rows, range_disparities


def threshold_sweep(y, scores, groups, thresholds):
    records=[]
    for t in thresholds:
        perf=performance(y,scores,t)
        disp=range_disparities(group_rows(y,scores,groups,t))
        records.append({'threshold':t, 'balanced_accuracy':perf['balanced_accuracy'],
            'accuracy':perf['accuracy'], 'selection_rate':perf['selection_rate'],
            'dp_difference':disp['demographic_parity_difference'],
            'eo_difference':disp['equalized_odds_difference'],
            'equal_opportunity_difference':disp['equal_opportunity_difference'],
            'dp_ratio':disp['demographic_parity_ratio'],
            'eo_ratio':disp['equalized_odds_ratio']})
    return records
