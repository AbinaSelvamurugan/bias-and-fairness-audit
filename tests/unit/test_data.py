import numpy as np
import pytest
from fairaudit.config import load_configs
from fairaudit.data.loader import load_dataset,provenance,fingerprint
from fairaudit.data.validation import validate
from fairaudit.data.splitting import split_indices,make_profile_keys

@pytest.fixture(scope='module')
def sample():
    return load_dataset()


def test_loading_integrity(sample):
    assert len(sample)==20190
    assert {'mdvis','physlm','idp','any_md_visit','physical_limitation','plan_type'}.issubset(sample.columns)
    assert sample['any_md_visit'].isin([0,1]).all()
    assert not sample['any_md_visit'].isna().any()
    assert len(fingerprint(sample))==64


def test_source_target_and_sensitive_group_definitions(sample):
    assert (sample['any_md_visit']==(sample['mdvis']>0).astype(int)).all()
    assert sample.loc[sample['physlm']==0,'physical_limitation'].eq('0_no').all()
    assert sample.loc[sample['physlm']==1,'physical_limitation'].eq('1_yes').all()
    assert '2_other_code' in sample['physical_limitation'].unique()
    assert set(sample.plan_type)=={'0_non_individual','1_individual'}


def test_dataset_validation_and_no_silent_drops(sample):
    c=load_configs(); out=validate(sample,c)
    assert out['rows']==sum(x['n'] for x in out['groups']['physical_limitation'])
    assert out['target_counts'][1]+out['target_counts'][0]==20190
    assert out['duplicates_non_target']>0
    assert out['nonstandard_physlm_values']
    invalid=sample.copy(); invalid.loc[0,'any_md_visit']=np.nan
    with pytest.raises(ValueError,match='Target missing'): validate(invalid,c)
    invalid=sample.drop(columns=['physlm'])
    with pytest.raises(ValueError,match='Required columns'): validate(invalid,c)


def test_profile_disjoint_split_but_not_person_proof(sample):
    c=load_configs(); split=split_indices(sample,c)
    keys=make_profile_keys(sample,c['dataset']['features']+c['dataset']['sensitive_source_columns'])
    assert len(set(keys[split['development']]).intersection(keys[split['test']]))==0
    assert len(set(split['train']).intersection(split['validation']))==0
    assert len(set(split['development']).intersection(split['test']))==0
    assert not split['person_disjoint']
