"""One-way audited workflow: load -> quality -> split -> select on validation -> holdout audit."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import logging
import joblib
import numpy as np
from fairaudit.config import ROOT, load_configs
from fairaudit.data.loader import load_dataset, provenance
from fairaudit.data.validation import validate
from fairaudit.data.splitting import split_indices
from fairaudit.models.training import fit_models, create_pipeline
from fairaudit.models.evaluation import performance
from fairaudit.fairness.group_analysis import group_rows, range_disparities
from fairaudit.fairness.fairlearn_metrics import optional_fairlearn
from fairaudit.fairness.aif360_metrics import optional_aif360
from fairaudit.fairness.uncertainty import add_wilson, exploratory_bootstrap
from fairaudit.fairness.threshold_analysis import threshold_sweep
from fairaudit.mitigation.experiments import label_group_weights
from fairaudit.reporting.plots import generate_plots
from fairaudit.reporting.findings import generate_report

LOG=logging.getLogger(__name__)


def jsonify(value):
    """Serialize numpy and undefined floating metrics to JSON null (strict JSON)."""
    if isinstance(value,dict): return {str(k):jsonify(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonify(x) for x in value]
    if isinstance(value,np.ndarray): return [jsonify(x) for x in value.tolist()]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    return value


def package_versions():
    versions={}
    for p in ('numpy','pandas','scikit-learn','scipy','statsmodels','fairlearn','aif360','streamlit','matplotlib','PyYAML','Jinja2'):
        try: versions[p]=version(p)
        except PackageNotFoundError: versions[p]=None
    return versions


def eval_group(y, scores, group, threshold, cfg):
    f=cfg['fairness']
    rows=add_wilson(group_rows(y,scores,group,threshold,f['small_group_n'],f['small_condition_n']),f['ci_level'])
    manual=range_disparities(rows)
    official=optional_fairlearn(y,scores,group,threshold)
    if official['status']=='fairlearn_metricframe_ran':
        for k in ('demographic_parity_difference','equalized_odds_difference','equal_opportunity_difference'):
            ref=official['fairlearn_builtins'][k]; ours=manual[k]
            if ours is not None and ref is not None and abs(ref-ours)>1e-9:
                raise AssertionError(f'Fairlearn and contingency metric disagree on {k}: {ref} vs {ours}')
    return {'rows':rows,'disparities':manual,'fairlearn':official}


def model_audit(df, indices, estimator, cfg):
    d=cfg['dataset']; f=cfg['fairness']; test=df.iloc[indices]
    y=test[d['target']].to_numpy(dtype=int)
    score=estimator.predict_proba(test[d['features']])[:,1]
    threshold=cfg['model']['threshold']
    attrs={}
    for a in f['attributes']:
        attrs[a]=eval_group(y,score,test[a].astype(str).to_numpy(),threshold,cfg)
    g_inter=test[f['intersection']].astype(str).agg(' × '.join,axis=1).to_numpy()
    attrs['intersection']=eval_group(y,score,g_inter,threshold,cfg)
    # Sensitivity contrast of only the two documented exact binary physlm codes.
    # Main audit ALWAYS retains the otherwise-coded group.
    exact=test['physical_limitation'].isin(['0_no','1_yes']).to_numpy()
    binary_only={'n_used':int(exact.sum()), 'n_not_in_binary_comparison':int((~exact).sum()),
       'note':'Exploratory two-group sensitivity; ambiguous code rows remain in main three-group audit.',
       'disparities':range_disparities(group_rows(y[exact],score[exact],test.loc[exact,'physical_limitation'].to_numpy(),threshold))}
    aif=optional_aif360(y,score,test['plan_type'].to_numpy(),threshold)
    sens=threshold_sweep(y,score,test[f['primary_attribute']].to_numpy(),f['thresholds'])
    return {'overall':performance(y,score,threshold),'groups':attrs,
            'physical_limitation_binary_only_sensitivity':binary_only,
            'aif360_plan_type':aif,'threshold_sensitivity':sens}


def run_audit(output=None, mitigation=False, bootstrap_repeats=None):
    cfg=load_configs(); cfg['mitigation']['enabled']=bool(mitigation)
    d=cfg['dataset']; model_cfg=cfg['model']
    if bootstrap_repeats is not None: cfg['fairness']['bootstrap_replicates']=int(bootstrap_repeats)
    out=Path(output) if output else ROOT/'results'/'latest'
    out.mkdir(parents=True,exist_ok=True)
    LOG.info('Loading public-domain RAND HIE statsmodels dataset')
    df=load_dataset(); quality=validate(df,cfg); prov=provenance(df)
    split=split_indices(df,cfg)
    LOG.info('Profile-group split dev=%d train=%d validation=%d test=%d', *(len(split[k]) for k in ['development','train','validation','test']))
    models,scores,chosen=fit_models(df,split,cfg)
    test=split['test']; outcome=df.iloc[test][d['target']].to_numpy()
    result={'schema_version':1,'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
         'created_utc':datetime.now(timezone.utc).isoformat(), 'provenance':prov,
         'versions':package_versions(), 'config':cfg,
         'split':{'strategy':split['strategy'],'seed':split['seed'], 'n_development':len(split['development']),
             'n_train':len(split['train']),'n_validation':len(split['validation']),
             'n_test':len(test),'n_unique_profiles':split['n_unique_profiles'],
             'person_disjoint':False},
         'test_n':len(test), 'selected_model':chosen,
         'selection_policy':model_cfg['selection_rule'],
         'validation_scores':scores, 'models':{},
         'validation':{**quality,'group_counts':quality['groups']},
         'limitations_status':{'person_cluster_uncertainty':'no participant or cluster identifiers in data',
             'temporal_order':'source timing for some predictors not documented',
             'physlm_fractional_codebook':'fractional codes retained as 2_other_code',
             'AIF360_runtime': 'optional extra not installed', 'Fairlearn_runtime':'requires fairlearn installed'}}
    for kind,model in models.items():
        LOG.info('Auditing model %s',kind)
        result['models'][kind]=model_audit(df,test,model,cfg)
        joblib.dump(model,out/f'model_{kind}.joblib')
    if result['models'][chosen]['groups']['physical_limitation']['fairlearn']['status']=='fairlearn_metricframe_ran':
        result['limitations_status']['Fairlearn_runtime']='available'
    if result['models'][chosen]['aif360_plan_type']['status']=='aif360_metrics_ran':
        result['limitations_status']['AIF360_runtime']='available'
    primary=cfg['fairness']['primary_attribute']
    model=models[chosen]
    score=model.predict_proba(df.iloc[test][d['features']])[:,1]
    result['uncertainty']=exploratory_bootstrap(outcome,score,df.iloc[test][primary].to_numpy(),
           model_cfg['threshold'], repeats=cfg['fairness']['bootstrap_replicates'],
           seed=cfg['fairness']['bootstrap_seed'],level=cfg['fairness']['ci_level'])
    if mitigation:
        LOG.info('Running experimental training-only reweighting')
        weights,policy=label_group_weights(df,split,primary,d['target'])
        m=create_pipeline(chosen,cfg,d['features'])
        m.fit(df.iloc[split['development']][d['features']],df.iloc[split['development']][d['target']],
              classifier__sample_weight=weights[split['development']])
        result['mitigation']=model_audit(df,test,m,cfg)
        result['mitigation']['weights_policy']=policy
        result['mitigation']['group']=primary
        joblib.dump(m,out/'model_mitigated.joblib')
    result=jsonify(result)
    with (out/'run.json').open('w',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False,sort_keys=True)
    figures=generate_plots(result,out/'figures')
    report_paths=generate_report(result,out)
    manifest={'results':'run.json','models':[f'model_{k}.joblib' for k in models],
              'figures':figures,'reports':report_paths,'mitigation':mitigation,
              'fairlearn_status':result['models'][chosen]['groups'][primary]['fairlearn']['status'],
              'aif360_status':result['models'][chosen]['aif360_plan_type']['status']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    LOG.info('Audit complete: %s (%s selected); reports %s',out,chosen,report_paths)
    return result
