import json
import numpy as np
from pathlib import Path
from sklearn.base import clone
from fairaudit.config import load_configs,ROOT
from fairaudit.data.loader import load_dataset
from fairaudit.data.splitting import split_indices
from fairaudit.models.training import create_pipeline
from fairaudit.mitigation.experiments import label_group_weights
from fairaudit.audit import jsonify
from fairaudit.reporting.findings import generate_report


def test_train_test_preprocessing_isolation():
    df=load_dataset(); c=load_configs(); split=split_indices(df,c)
    features=c['dataset']['features']; y=c['dataset']['target']
    train=split['development'];test=split['test']
    model=create_pipeline('logistic',c,features)
    model.fit(df.iloc[train][features],df.iloc[train][y])
    before=model.predict_proba(df.iloc[test[:5]][features])
    altered=df.copy()
    altered.loc[test,'disea']=1_000_000_000
    second=create_pipeline('logistic',c,features)
    second.fit(altered.iloc[train][features],altered.iloc[train][y])
    assert np.allclose(model.named_steps['preprocess'].transformers_[0][1]['impute'].statistics_,
                       second.named_steps['preprocess'].transformers_[0][1]['impute'].statistics_)
    assert np.allclose(before,second.predict_proba(df.iloc[test[:5]][features]))
    assert set(features).isdisjoint({'mdvis','physlm','idp','any_md_visit','physical_limitation','plan_type'})


def test_mitigation_weights_cannot_depend_on_test_labels():
    df=load_dataset();c=load_configs();split=split_indices(df,c)
    a,policy=label_group_weights(df,split,'physical_limitation','any_md_visit')
    altered=df.copy()
    altered.loc[split['test'],'any_md_visit']=1-altered.loc[split['test'],'any_md_visit']
    b,_=label_group_weights(altered,split,'physical_limitation','any_md_visit')
    assert np.array_equal(a,b)
    assert np.all(a[split['test']]==1)
    assert policy['group']=='physical_limitation'


def test_serialization_no_nan():
    payload=jsonify({'a':np.nan,'b':np.int64(5),'c':[np.float64(.2)]})
    assert payload=={'a':None,'b':5,'c':[.2]}
    json.dumps(payload,allow_nan=False)


def test_report_reproduction_from_real_run(tmp_path):
    source=ROOT/'results/latest/run.json'
    assert source.exists(), 'Run audit before report regression test'
    result=json.loads(source.read_text())
    files=generate_report(result,tmp_path)
    md=(tmp_path/'findings.md').read_text()
    html=(tmp_path/'findings.html').read_text()
    assert files==['findings.md','findings.html']
    assert 'fairlearn_not_installed' in md or 'fairlearn_metricframe_ran' in md
    assert '## 18. References' in md
    assert 'RAND' in html and 'Equalized odds' in html
    assert str(result['test_n']) in md
