"""Generate a Markdown and HTML report from the saved run results."""
from pathlib import Path
import html
import json


def fmt(v,precision=3):
    if v is None: return 'undefined'
    if isinstance(v,(int,float)): return f'{v:.{precision}f}'
    return str(v)


def tab(rows,cols):
    head='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
    body=''
    for r in rows:
        body+='| '+' | '.join(fmt(r.get(k)) for k in cols)+' |\n'
    return head+body


def generate_report(result, output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    model=result['selected_model']; chosen=result['models'][model]
    p=chosen['overall']; prim=chosen['groups']['physical_limitation']; secondary=chosen['groups']['plan_type']
    pdiff=prim['disparities']; v=result['validation']; prov=result['provenance']
    group_table_cols=['group','n','positive_labels','negative_labels','tp','fp','tn','fn',
                     'label_prevalence','selection_rate','tpr','fpr','fnr','selection_rate_wilson']
    # Tables deliberately include audit denominators; no group exclusion based on n.
    lines=[
       '# FairAudit — Bias and Fairness Assessment of a Public Machine-Learning Dataset',
       f"**Run identifier:** `{result['run_id']}` | **Scope:** observational audit, not a discrimination determination",
       '## 1. Executive findings',
       f"Selected `{model}` on **training-side validation balanced accuracy**, not on final test fairness scores. Held-out records: **{result['test_n']}**; test accuracy **{fmt(p['accuracy'])}**, balanced accuracy **{fmt(p['balanced_accuracy'])}**, ROC-AUC **{fmt(p['roc_auc'])}**.",
       f"For physical-limitation coding, demographic-parity difference (maximum minus minimum group selection) = **{fmt(pdiff['demographic_parity_difference'])}**; equalized-odds difference (maximal group TPR/FPR range) = **{fmt(pdiff['equalized_odds_difference'])}**. These are descriptive associations only.",
       f"Fairlearn MetricFrame: **{prim['fairlearn']['status']}**. AIF360 plan-type comparison: **{chosen['aif360_plan_type']['status']}**.",
       '## 2. Scope and intended use',
       '**Research question:** In a historical RAND Health Insurance Experiment sample, do models that predict whether a recorded observation contains one or more outpatient physician visits make different errors and positive predictions across reported physical-limitation and insurance-plan subgroups?',
       'The positive outcome (Y=1) is *any recorded physician visit*, NOT adequate access, clinical need, quality, desirability, or an entitlement. A positive prediction is not a recommendation for or against care. This is a retrospective demonstration and must not be deployed to allocate care or benefits.',
       '## 3. Dataset provenance and usage conditions',
       f"Dataset `statsmodels.datasets.randhie`, installed statsmodels version **{prov['statsmodels_version']}**. Upstream: {prov['source_url']}. Original study: RAND Health Insurance Experiment (1974–1982), extracted subset, **{prov['rows']} rows** and **10 original variables**. The dataset-specific statsmodels copyright states **public domain**. Source extraction differs slightly from published versions; no canonical standalone release identifier. Exact content fingerprint: `{prov['input_sha256_pandas_hash']}`.",
       'Upstream references: https://www.statsmodels.org/stable/datasets/generated/randhie.html ; https://www.icpsr.umich.edu/web/NACDA/studies/6439/ . The latter documents study-site clustering, sampling, restricted population eligibility, and longitudinal structure; these cannot be reconstructed from the distributed ten-column sample.',
       '## 4. Data quality and representation',
       f"Data rows **{v['rows']}**, repeated predictor+group profiles **{v['duplicates_non_target']}**, and exact original-row repeats **{v['duplicates_exact_rows']}**. No rows dropped. Target prevalence **{fmt(v['target_prevalence'])}**. Missingness by column: `{json.dumps(v['n_missing_by_column'])}`.",
       '**Original population composition:**',
       tab(v['group_counts']['physical_limitation'], ['group','n','positive_labels','prevalence']),
       tab(v['group_counts']['plan_type'], ['group','n','positive_labels','prevalence']),
       '**Validation warnings and potential data quality issues:**',
       '\n'.join('- '+w for w in v['warnings']),
       f"Undocumented distinct nonbinary source `physlm` codes: `{v['nonstandard_physlm_values']}`. Retained in the group `2_other_code`; this is **not** a confirmed disability status or a missing-value imputation.",
       '## 5. Prediction target, potential harms, and chosen fairness questions',
       'Y=1 if `mdvis>0`. Potential quality-of-service concern: disproportionate false negatives (missed recorded utilization), though there is no clinical ground truth of need. Potential allocation concern: selection-rate differences if predictions were misused to trigger interventions. Demographic parity describes outcomes, equal opportunity isolates TPR among label-positive observations, and equalized odds compares both TPR and FPR conditional on labels. None is a universal normative objective here.',
       '## 6. Features, model and split methodology',
       f"Included predictors: `{result['config']['dataset']['features']}`; excluded `mdvis` (label leakage) and `physlm`,`idp` (evaluation-only grouping indicators). Historic health status, insurance plan terms, and coinsurance correlate with group indicators and may act as proxies; omission does not remove proxy dependence.",
       f"**Split:** {result['split']['strategy']}, seed={result['split']['seed']}; n_development={result['split']['n_development']}, n_train={result['split']['n_train']}, n_validation={result['split']['n_validation']}, n_test={result['split']['n_test']}. Validation score chooses model, after which each estimator is refit on development only. Exact matching observed feature profiles are not shared with test; without person/household identifiers a person-disjoint split is not possible.",
       f"Preprocessing: median imputation and scaling in a scikit-learn Pipeline, fitted only on development observations. Logistic, random forest and dummy use fixed settings recorded in `run.json`. All final evaluations use threshold={p['threshold']}; sensitivity thresholds are descriptive and not model selection.",
       'Training-side validation balanced accuracy:',
       tab([{'model':k,'balanced_accuracy':value} for k,value in result['validation_scores'].items()],['model','balanced_accuracy']),
       '## 7. Sensitive attributes and group definitions',
       '`physical_limitation`: 0_no (raw physlm=0), 1_yes (raw physlm=1), 2_other_code (nonzero fractional, undocumented); `plan_type`: 0_non_individual (raw idp=0), 1_individual (raw idp=1). These terms are descriptive, not assessments of advantage/privilege. The ambiguous-code group is retained everywhere. Intersectional groups are reported with explicit counts.',
       '## 8. Overall predictive performance',
       tab([{'model':k, **entry['overall']} for k,entry in result['models'].items()],
           ['model','n','accuracy','balanced_accuracy','precision','recall_tpr','specificity_tnr','false_positive_rate','false_negative_rate','f1','roc_auc','selection_rate']),
       f"Selected model confusion matrix: TP={p['confusion_matrix']['tp']}, FP={p['confusion_matrix']['fp']}, TN={p['confusion_matrix']['tn']}, FN={p['confusion_matrix']['fn']}. Decision threshold = {p['threshold']}, no excluded labels.",
       '## 9. Disaggregated group results (held-out counts)',
       '**Physical limitation:**',tab(prim['rows'],group_table_cols),
       '**Insurance plan:**',tab(secondary['rows'],group_table_cols),
       '**Intersectional groups** (interpret small cells cautiously):',
       tab(chosen['groups']['intersection']['rows'],['group','n','positive_labels','negative_labels','selection_rate','tpr','fpr','fnr']),
       '## 10. Fairness metrics: definitions, conventions and assumptions',
       'Selection rate = (TP+FP)/N. TPR=TP/(TP+FN); FPR=FP/(FP+TN); FNR=FN/(FN+TP); specificity=TN/(TN+FP). Precision=TP/(TP+FP). All are undefined when denominators are zero. Each rate conditions on the observed label, which can have measurement or selection biases.',
       'Demographic parity difference = highest minus lowest group selection rate. Demographic parity ratio = min/max selection rate, if max>0. Equal opportunity difference = max minus min TPR. Equalized odds difference = max(range TPR, range FPR). Equalized odds ratio = min(min/max TPR, min/max FPR), where defined. Comparisons use all retained groups; no favorable/unfavorable group is presupposed.',
       tab([{'metric':k,'value':v} for k,v in pdiff.items() if k.endswith(('difference','ratio'))],['metric','value']),
       f"Fairlearn MetricFrame: `{prim['fairlearn']['status']}`; metric values from the independently implemented contingency engine are always identified as such. AIF360 binary plan comparison: `{chosen['aif360_plan_type']}`. Signed AIF360 statistical-parity difference is individual minus non-individual; disparate-impact ratio is individual/non-individual. AIF360 signed average-odds difference = (TPR difference + FPR difference)/2 and is **not** Fairlearn's maximum-absolute difference.",
       '**Sensitivity restricted to the two explicitly binary physical-limitation codes** (shown *in addition* to, not instead of, all-group results):',
       f"Used {chosen['physical_limitation_binary_only_sensitivity']['n_used']} exact binary-code observations; {chosen['physical_limitation_binary_only_sensitivity']['n_not_in_binary_comparison']} ambiguous-code observations excluded **only** from this sensitivity contrast.",
       tab([{'metric':k,'value':value} for k,value in chosen['physical_limitation_binary_only_sensitivity']['disparities'].items() if k in ('demographic_parity_difference','equalized_odds_difference','equal_opportunity_difference')], ['metric','value']),
       '## 11. Statistical uncertainty and sample size',
       f"Group Wilson score intervals are included for proportions conditional on their documented denominators. Exploratory within-group row bootstrap of {result['uncertainty']['replicates_requested']} draws ({fmt(result['uncertainty']['level'])} nominal quantile interval) produced:",
       tab([{'metric':k,**entry} for k,entry in result['uncertainty']['metrics'].items()],
           ['metric','point_estimate','exploratory_row_bootstrap_interval','successful_replicates']),
       '**Strong limitation:** no person or site IDs are available; the simple Wilson intervals and row bootstrap are not guaranteed calibrated for the true correlated/clustered study. Do not use them for a significance or legal-compliance claim. Intervals excluding zero would not resolve causality, multiplicity, or selection. Missing small-group signals do not establish parity.',
       '## 12. Observed disparities and error trade-offs',
       'Compare the tabled TPR and FPR separately before drawing conclusions. The largest absolute between-group difference is an index for further review, not evidence by itself of intentional or unlawful discrimination. Differences may reflect group-dependent utilization base rates, missing structure, uncertain variable coding, measurement error, confounding, or model miscalibration; causal mechanisms are not identified.',
       '## 13. Threshold and alternative-model sensitivity',
       'The following fixed thresholds are **post-hoc diagnostic views on the test scores**, not candidate thresholds used to choose a deployable decision rule:',
       tab(chosen['threshold_sensitivity'],['threshold','accuracy','balanced_accuracy','dp_difference','eo_difference','equal_opportunity_difference']),
       'Across-model physical-limitation disparities at the same held-out threshold:',
       tab([{'model':k,**m['groups']['physical_limitation']['disparities']} for k,m in result['models'].items()],
           ['model','demographic_parity_difference','equalized_odds_difference','equal_opportunity_difference','demographic_parity_ratio']),
       '## 14. Experimental mitigation and comparison',
    ]
    if result.get('mitigation'):
        m=result['mitigation']; lines += [
          f"Experimental reweighing uses development-set `P(group)P(Y)/P(group,Y)` weights for `{m['group']}` and retains the selected model architecture. No held-out labels or test fairness metrics select weights. Model remains evaluated at the original 0.5 threshold.",
          tab([{'model':'original',**p},{'model':'reweighted',**m['overall']}],['model','accuracy','balanced_accuracy','selection_rate','recall_tpr','false_positive_rate','false_negative_rate']),
          tab([{'model':'original',**prim['disparities']},{'model':'reweighted',**m['groups']['physical_limitation']['disparities']}],['model','demographic_parity_difference','equalized_odds_difference','equal_opportunity_difference']),
          'Whether mitigation improved a targeted metric must be assessed against both figures and uncertainty, not assumed. Reweighing changes objective/sample influence, not labels or historical access patterns. A new degradation in another subgroup or metric remains possible.'
        ]
    else:
        lines.append('Not run by default. Enable `--mitigation` to evaluate a documented training-only group-label reweighing experiment without changing the held-out test records.')
    lines += [
       '## 15. Utility–fairness trade-offs',
       'Accuracy and balanced accuracy measure prediction of observed utilization; group metrics measure particular rate relationships. The same threshold may yield more false negatives, more false positives, or both in different subgroups. Model and mitigation comparisons must consider all these rates jointly.',
       '## 16. Limits of causal or ethical interpretation',
       'The sample does not reveal why physician visits did or did not occur. Observed utilization is not medical need, access, quality, patient choice, or disparate treatment. Restricted recruitment, site cluster selection, correlated repeated measures, unobserved socioeconomic attributes and uncertain fractional physical-limitation coding prevent identification of discrimination. There is no operational decision rule or affected user population in this project.',
       '## 17. Actionable next investigations',
       '- Obtain original codebooks and participant/site identifiers; investigate the nonbinary physlm codes, household clusters, and temporal ordering before making inferential claims.',
       '- Define a concrete deployment context, concrete harms, and valid labels; consult affected groups and clinical/domain experts before proposing fairness constraints.',
       '- Replicate across site/year cohorts and person-disjoint splits; compute cluster-resampled confidence intervals and appropriate multiplicity adjustments.',
       '- Conduct label-quality and measurement sensitivity studies; compare practical utility/fairness costs prospectively, never by repeated reuse of this test set.',
       '## 18. References and environment',
       '- RAND HIE statsmodels documentation: https://www.statsmodels.org/stable/datasets/generated/randhie.html',
       '- RAND HIE ICPSR documentation: https://www.icpsr.umich.edu/web/NACDA/studies/6439/',
       '- Fairlearn metric API and definitions: https://fairlearn.org/v0.13/api_reference/index.html',
       '- AIF360 sklearn metric definitions: https://aif360.readthedocs.io/en/stable/modules/sklearn.html',
       f"- Captured package versions: `{json.dumps(result['versions'],sort_keys=True)}`",
       f"- Run configurations, scores, uncertainty, warnings and dataset fingerprint: `run.json`. Fixed random seeds are documented, but cross-platform bitwise identity is not promised.",
       '\n**Conclusion:** These measurements motivate focused investigation; they neither establish a model to be fair nor establish that discrimination occurred.',
    ]
    md='\n\n'.join(lines)+'\n'
    (output_dir/'findings.md').write_text(md, encoding='utf-8')
    try:
        import mistune
        body=mistune.create_markdown(plugins=['table'])(md)
    except ImportError:
        body='<pre>'+html.escape(md)+'</pre>'
    style='body{max-width:1120px;margin:45px auto;padding:0 24px;font:16px/1.55 system-ui;color:#1d2935}h1,h2{color:#123752}table{border-collapse:collapse;display:block;overflow-x:auto}td,th{padding:7px 10px;border:1px solid #d7dde3;white-space:nowrap}th{background:#eff3f6}code{overflow-wrap:anywhere}h2{margin-top:38px}blockquote{color:#53616f}'
    html_doc=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FairAudit findings</title><style>{style}</style></head><body>{body}</body></html>'
    (output_dir/'findings.html').write_text(html_doc, encoding='utf-8')
    return ['findings.md','findings.html']
