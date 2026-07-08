import numpy as np
import pytest
from fairaudit.fairness.group_analysis import group_rows,range_disparities


def test_fairlearn_real_metricframe_fixture():
    pytest.importorskip('fairlearn')
    from fairaudit.fairness.fairlearn_metrics import metricframe_audit
    y=np.array([1,1,0,0,1,1,0,0]);score=np.array([.9,.1,.9,.1,.9,.9,.1,.1]);g=np.array(['A']*4+['B']*4)
    m=metricframe_audit(y,score,g)
    assert m['fairlearn_builtins']['demographic_parity_difference']==pytest.approx(0)
    assert m['fairlearn_builtins']['equalized_odds_difference']==pytest.approx(.5)
    assert m['by_group']['A']['tpr']==pytest.approx(.5)
    assert m['by_group']['B']['fpr']==pytest.approx(0)


def test_aif360_real_pairwise_fixture():
    pytest.importorskip('aif360')
    from fairaudit.fairness.aif360_metrics import optional_aif360
    y=np.array([1,1,0,0,1,1,0,0]);s=np.array([.9,.1,.9,.1,.9,.9,.1,.1]);g=np.array(['0_non_individual']*4+['1_individual']*4)
    m=optional_aif360(y,s,g)
    assert m['status']=='aif360_metrics_ran'
    assert m['metrics']['statistical_parity_difference']==pytest.approx(0)
    assert m['metrics']['equal_opportunity_difference']==pytest.approx(.5)
    assert m['metrics']['average_odds_difference']==pytest.approx(0)
    assert m['metrics']['disparate_impact_ratio']==pytest.approx(1)
