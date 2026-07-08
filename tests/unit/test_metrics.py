import numpy as np
import pytest
from fairaudit.fairness.group_analysis import group_rows,range_disparities
from fairaudit.models.evaluation import performance
from fairaudit.fairness.threshold_analysis import threshold_sweep
from fairaudit.fairness.uncertainty import wilson_interval, exploratory_bootstrap

# Manually checked 2x2 confusion matrices A=(TP1,FP1,TN1,FN1), B=(TP2,FP0,TN2,FN0).
y=np.array([1,1,0,0,1,1,0,0])
scores=np.array([.9,.1,.8,.1,.9,.8,.2,.1])
group=np.array(['A']*4+['B']*4)


def test_hand_verified_group_counts():
    a,b=group_rows(y,scores,group,.5)
    assert [a[k] for k in ('n','positive_labels','negative_labels','tp','fp','tn','fn')]==[4,2,2,1,1,1,1]
    assert [b[k] for k in ('n','positive_labels','negative_labels','tp','fp','tn','fn')]==[4,2,2,2,0,2,0]
    assert a['selection_rate']==b['selection_rate']==.5
    assert a['tpr']==.5 and a['fpr']==.5 and b['tpr']==1 and b['fpr']==0
    d=range_disparities([a,b])
    assert d['demographic_parity_difference']==0
    assert d['demographic_parity_ratio']==1
    assert d['equal_opportunity_difference']==.5
    assert d['equalized_odds_difference']==.5
    assert d['equalized_odds_ratio']==0


def test_overall_hand_calculated():
    p=performance(y,scores,.5)
    assert p['accuracy']==.75
    assert p['balanced_accuracy']==.75
    assert p['confusion_matrix']=={'tn':3,'fp':1,'fn':1,'tp':3}
    assert p['precision']==.75 and p['f1']==.75


def test_undefined_denominators_and_small_group():
    y0=[0,0,1,1]; score=[0.1,0.2,0.1,0.2];g=['A','A','B','B']
    a,b=group_rows(y0,score,g,.5,min_n=5,min_condition=2)
    assert a['tpr'] is None and b['fpr'] is None
    assert a['warning_small_n'] and b['warning_small_n']
    d=range_disparities([a,b])
    assert d['equalized_odds_difference'] is None
    assert d['demographic_parity_ratio'] is None # Both selection rates zero
    assert performance([0,0],[0.1,0.3])['recall_tpr'] is None
    assert wilson_interval(0,0)==[None,None]


def test_zero_ratio_can_be_defined():
    rows=group_rows(y,[0.9,0.9,0.0,0.0,0.1,0.1,0.1,0.1],group)
    assert range_disparities(rows)['demographic_parity_ratio']==0


def test_missing_labels_and_unmatched_lengths():
    with pytest.raises(ValueError): group_rows([1,None],[.1,.2],['A','B'])
    with pytest.raises(ValueError): group_rows([1,0],[.1],['A','B'])
    with pytest.raises(ValueError): group_rows([1,0],[1.2,.5],['A','B'])
    with pytest.raises(ValueError): group_rows([1,0],[np.nan,.5],['A','B'])
    with pytest.raises(ValueError): group_rows([1,0],[.5,.5],['A',None])


def test_wilson_known_coverage_limits():
    # 1/2 at z=1.95996: approx 0.09453, 0.90547.
    lo,hi=wilson_interval(1,2)
    assert abs(lo-0.09453)<.0002 and abs(hi-0.90547)<.0002
    assert wilson_interval(0,10)[0]==0
    assert wilson_interval(10,10)[1]==1


def test_threshold_sensitivity_deterministic():
    sweep=threshold_sweep(y,scores,group,[.5,.95])
    assert sweep[0]['dp_difference']==0 and sweep[0]['eo_difference']==.5
    assert sweep[1]['selection_rate']==0
    assert sweep[1]['dp_ratio'] is None


def test_bootstrap_reports_uncertainty_limit():
    b=exploratory_bootstrap(y,scores,group,.5,repeats=30,seed=5)
    assert b['status']=='EXPLORATORY_NOT_CLUSTER_VALID'
    assert b['metrics']['demographic_parity_difference']['point_estimate']==0
    assert b['metrics']['demographic_parity_difference']['successful_replicates']==30
