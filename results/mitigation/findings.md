# FairAudit — Bias and Fairness Assessment of a Public Machine-Learning Dataset

**Run identifier:** `20260901T050919Z` | **Scope:** observational audit, not a discrimination determination

## 1. Executive findings

Selected `forest` on **training-side validation balanced accuracy**, not on final test fairness scores. Held-out records: **5061**; test accuracy **0.709**, balanced accuracy **0.583**, ROC-AUC **0.667**.

For physical-limitation coding, demographic-parity difference (maximum minus minimum group selection) = **0.351**; equalized-odds difference (maximal group TPR/FPR range) = **0.602**. These are descriptive associations only.

Fairlearn MetricFrame: **fairlearn_metricframe_ran**. AIF360 plan-type comparison: **aif360_not_installed**.

## 2. Scope and intended use

**Research question:** In a historical RAND Health Insurance Experiment sample, do models that predict whether a recorded observation contains one or more outpatient physician visits make different errors and positive predictions across reported physical-limitation and insurance-plan subgroups?

The positive outcome (Y=1) is *any recorded physician visit*, NOT adequate access, clinical need, quality, desirability, or an entitlement. A positive prediction is not a recommendation for or against care. This is a retrospective demonstration and must not be deployed to allocate care or benefits.

## 3. Dataset provenance and usage conditions

Dataset `statsmodels.datasets.randhie`, installed statsmodels version **0.15.0**. Upstream: https://www.statsmodels.org/stable/datasets/generated/randhie.html. Original study: RAND Health Insurance Experiment (1974–1982), extracted subset, **20190 rows** and **10 original variables**. The dataset-specific statsmodels copyright states **public domain**. Source extraction differs slightly from published versions; no canonical standalone release identifier. Exact content fingerprint: `08e1a0bdc18d728cc909e9f7ff3d22845021fec82dca9d7b4172137f1bec76a3`.

Upstream references: https://www.statsmodels.org/stable/datasets/generated/randhie.html ; https://www.icpsr.umich.edu/web/NACDA/studies/6439/ . The latter documents study-site clustering, sampling, restricted population eligibility, and longitudinal structure; these cannot be reconstructed from the distributed ten-column sample.

## 4. Data quality and representation

Data rows **20190**, repeated predictor+group profiles **17430**, and exact original-row repeats **11065**. No rows dropped. Target prevalence **0.688**. Missingness by column: `{"mdvis": 0, "lncoins": 0, "idp": 0, "lpi": 0, "fmde": 0, "physlm": 0, "disea": 0, "hlthg": 0, "hlthf": 0, "hlthp": 0, "any_md_visit": 0, "physical_limitation": 0, "plan_type": 0}`.

**Original population composition:**

| group | n | positive_labels | prevalence |
| --- | --- | --- | --- |
| 0_no | 16751.000 | 11323.000 | 0.676 |
| 1_yes | 2387.000 | 1857.000 | 0.778 |
| 2_other_code | 1052.000 | 702.000 | 0.667 |


| group | n | positive_labels | prevalence |
| --- | --- | --- | --- |
| 0_non_individual | 14941.000 | 10588.000 | 0.709 |
| 1_individual | 5249.000 | 3294.000 | 0.628 |


**Validation warnings and potential data quality issues:**

- physlm has fractional, undocumented codes: retained as 2_other_code, not imputed to disability status
- 17430 rows repeat predictor+group-source profiles; 11065 are exact original-row duplicates (may reflect repeat observations; person IDs unavailable)
- No participant/household/time identifiers are present, so independence and person-disjoint separation are not supported
- Health self-report timing relative to outpatient visits is not documented (potential temporal leakage)
- disea is described as a disease count, but the packaged sample includes fractional transformed values; original coding is not documented
- Dataset is an observational, historical experimental subset and not a deployable modern care-access target

Undocumented distinct nonbinary source `physlm` codes: `[0.0221239, 0.0268456, 0.0277778, 0.0327869, 0.0431267, 0.12982, 0.1442925, 0.1572505, 0.1981873]`. Retained in the group `2_other_code`; this is **not** a confirmed disability status or a missing-value imputation.

## 5. Prediction target, potential harms, and chosen fairness questions

Y=1 if `mdvis>0`. Potential quality-of-service concern: disproportionate false negatives (missed recorded utilization), though there is no clinical ground truth of need. Potential allocation concern: selection-rate differences if predictions were misused to trigger interventions. Demographic parity describes outcomes, equal opportunity isolates TPR among label-positive observations, and equalized odds compares both TPR and FPR conditional on labels. None is a universal normative objective here.

## 6. Features, model and split methodology

Included predictors: `['lncoins', 'lpi', 'fmde', 'disea', 'hlthg', 'hlthf', 'hlthp']`; excluded `mdvis` (label leakage) and `physlm`,`idp` (evaluation-only grouping indicators). Historic health status, insurance plan terms, and coinsurance correlate with group indicators and may act as proxies; omission does not remove proxy dependence.

**Split:** GroupShuffleSplit on hash of complete observed input profile (not a person ID), seed=20260101; n_development=15129, n_train=11344, n_validation=3785, n_test=5061. Validation score chooses model, after which each estimator is refit on development only. Exact matching observed feature profiles are not shared with test; without person/household identifiers a person-disjoint split is not possible.

Preprocessing: median imputation and scaling in a scikit-learn Pipeline, fitted only on development observations. Logistic, random forest and dummy use fixed settings recorded in `run.json`. All final evaluations use threshold=0.5; sensitivity thresholds are descriptive and not model selection.

Training-side validation balanced accuracy:

| model | balanced_accuracy |
| --- | --- |
| dummy | 0.500 |
| logistic | 0.519 |
| forest | 0.554 |


## 7. Sensitive attributes and group definitions

`physical_limitation`: 0_no (raw physlm=0), 1_yes (raw physlm=1), 2_other_code (nonzero fractional, undocumented); `plan_type`: 0_non_individual (raw idp=0), 1_individual (raw idp=1). These terms are descriptive, not assessments of advantage/privilege. The ambiguous-code group is retained everywhere. Intersectional groups are reported with explicit counts.

## 8. Overall predictive performance

| model | n | accuracy | balanced_accuracy | precision | recall_tpr | specificity_tnr | false_positive_rate | false_negative_rate | f1 | roc_auc | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy | 5061.000 | 0.673 | 0.500 | 0.673 | 1.000 | 0.000 | 1.000 | 0.000 | 0.804 | 0.500 | 1.000 |
| logistic | 5061.000 | 0.677 | 0.524 | 0.684 | 0.966 | 0.082 | 0.918 | 0.034 | 0.801 | 0.620 | 0.950 |
| forest | 5061.000 | 0.709 | 0.583 | 0.714 | 0.946 | 0.220 | 0.780 | 0.054 | 0.814 | 0.667 | 0.892 |


Selected model confusion matrix: TP=3221, FP=1291, TN=365, FN=184. Decision threshold = 0.5, no excluded labels.

## 9. Disaggregated group results (held-out counts)

**Physical limitation:**

| group | n | positive_labels | negative_labels | tp | fp | tn | fn | label_prevalence | selection_rate | tpr | fpr | fnr | selection_rate_wilson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0_no | 4174.000 | 2774.000 | 1400.000 | 2657.000 | 1144.000 | 256.000 | 117.000 | 0.665 | 0.911 | 0.958 | 0.817 | 0.042 | [0.9016013297190706, 0.9189180781589974] |
| 1_yes | 544.000 | 416.000 | 128.000 | 398.000 | 112.000 | 16.000 | 18.000 | 0.765 | 0.938 | 0.957 | 0.875 | 0.043 | [0.9139317702144806, 0.954932738551718] |
| 2_other_code | 343.000 | 215.000 | 128.000 | 166.000 | 35.000 | 93.000 | 49.000 | 0.627 | 0.586 | 0.772 | 0.273 | 0.228 | [0.533208614516221, 0.6368979240752369] |


**Insurance plan:**

| group | n | positive_labels | negative_labels | tp | fp | tn | fn | label_prevalence | selection_rate | tpr | fpr | fnr | selection_rate_wilson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0_non_individual | 3941.000 | 2713.000 | 1228.000 | 2597.000 | 975.000 | 253.000 | 116.000 | 0.688 | 0.906 | 0.957 | 0.794 | 0.043 | [0.8968739460790749, 0.9150724992813766] |
| 1_individual | 1120.000 | 692.000 | 428.000 | 624.000 | 316.000 | 112.000 | 68.000 | 0.618 | 0.839 | 0.902 | 0.738 | 0.098 | [0.8166224304351516, 0.8596295391242451] |


**Intersectional groups** (interpret small cells cautiously):

| group | n | positive_labels | negative_labels | selection_rate | tpr | fpr | fnr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0_no × 0_non_individual | 3279.000 | 2232.000 | 1047.000 | 0.926 | 0.965 | 0.842 | 0.035 |
| 0_no × 1_individual | 895.000 | 542.000 | 353.000 | 0.856 | 0.930 | 0.742 | 0.070 |
| 1_yes × 0_non_individual | 391.000 | 315.000 | 76.000 | 0.941 | 0.965 | 0.842 | 0.035 |
| 1_yes × 1_individual | 153.000 | 101.000 | 52.000 | 0.928 | 0.931 | 0.923 | 0.069 |
| 2_other_code × 0_non_individual | 271.000 | 166.000 | 105.000 | 0.624 | 0.843 | 0.276 | 0.157 |
| 2_other_code × 1_individual | 72.000 | 49.000 | 23.000 | 0.444 | 0.531 | 0.261 | 0.469 |


## 10. Fairness metrics: definitions, conventions and assumptions

Selection rate = (TP+FP)/N. TPR=TP/(TP+FN); FPR=FP/(FP+TN); FNR=FN/(FN+TP); specificity=TN/(TN+FP). Precision=TP/(TP+FP). All are undefined when denominators are zero. Each rate conditions on the observed label, which can have measurement or selection biases.

Demographic parity difference = highest minus lowest group selection rate. Demographic parity ratio = min/max selection rate, if max>0. Equal opportunity difference = max minus min TPR. Equalized odds difference = max(range TPR, range FPR). Equalized odds ratio = min(min/max TPR, min/max FPR), where defined. Comparisons use all retained groups; no favorable/unfavorable group is presupposed.

| metric | value |
| --- | --- |
| selection_rate_difference | 0.351 |
| selection_rate_ratio | 0.625 |
| tpr_difference | 0.186 |
| tpr_ratio | 0.806 |
| fpr_difference | 0.602 |
| fpr_ratio | 0.312 |
| fnr_difference | 0.186 |
| fnr_ratio | 0.185 |
| accuracy_difference | 0.063 |
| accuracy_ratio | 0.917 |
| demographic_parity_difference | 0.351 |
| demographic_parity_ratio | 0.625 |
| equal_opportunity_difference | 0.186 |
| equalized_odds_difference | 0.602 |
| equalized_odds_ratio | 0.312 |


Fairlearn MetricFrame: `fairlearn_metricframe_ran`; metric values from the independently implemented contingency engine are always identified as such. AIF360 binary plan comparison: `{'status': 'aif360_not_installed', 'detail': "No module named 'aif360'"}`. Signed AIF360 statistical-parity difference is individual minus non-individual; disparate-impact ratio is individual/non-individual. AIF360 signed average-odds difference = (TPR difference + FPR difference)/2 and is **not** Fairlearn's maximum-absolute difference.

**Sensitivity restricted to the two explicitly binary physical-limitation codes** (shown *in addition* to, not instead of, all-group results):

Used 4718 exact binary-code observations; 343 ambiguous-code observations excluded **only** from this sensitivity contrast.

| metric | value |
| --- | --- |
| demographic_parity_difference | 0.027 |
| equal_opportunity_difference | 0.001 |
| equalized_odds_difference | 0.058 |


## 11. Statistical uncertainty and sample size

Group Wilson score intervals are included for proportions conditional on their documented denominators. Exploratory within-group row bootstrap of 150 draws (0.950 nominal quantile interval) produced:

| metric | point_estimate | exploratory_row_bootstrap_interval | successful_replicates |
| --- | --- | --- | --- |
| demographic_parity_difference | 0.351 | [0.29512733703413596, 0.40436178935860057] | 150.000 |
| equalized_odds_difference | 0.602 | [0.5054337847665864, 0.696862884977651] | 150.000 |
| equal_opportunity_difference | 0.186 | [0.14003761559565617, 0.24891339606473314] | 150.000 |


**Strong limitation:** no person or site IDs are available; the simple Wilson intervals and row bootstrap are not guaranteed calibrated for the true correlated/clustered study. Do not use them for a significance or legal-compliance claim. Intervals excluding zero would not resolve causality, multiplicity, or selection. Missing small-group signals do not establish parity.

## 12. Observed disparities and error trade-offs

Compare the tabled TPR and FPR separately before drawing conclusions. The largest absolute between-group difference is an index for further review, not evidence by itself of intentional or unlawful discrimination. Differences may reflect group-dependent utilization base rates, missing structure, uncertain variable coding, measurement error, confounding, or model miscalibration; causal mechanisms are not identified.

## 13. Threshold and alternative-model sensitivity

The following fixed thresholds are **post-hoc diagnostic views on the test scores**, not candidate thresholds used to choose a deployable decision rule:

| threshold | accuracy | balanced_accuracy | dp_difference | eo_difference | equal_opportunity_difference |
| --- | --- | --- | --- | --- | --- |
| 0.200 | 0.673 | 0.500 | 0.000 | 0.000 | 0.000 |
| 0.300 | 0.683 | 0.519 | 0.158 | 0.328 | 0.060 |
| 0.400 | 0.700 | 0.550 | 0.161 | 0.305 | 0.056 |
| 0.500 | 0.709 | 0.583 | 0.351 | 0.602 | 0.186 |
| 0.600 | 0.700 | 0.619 | 0.334 | 0.523 | 0.187 |
| 0.700 | 0.617 | 0.618 | 0.250 | 0.406 | 0.114 |
| 0.800 | 0.483 | 0.573 | 0.276 | 0.289 | 0.239 |


Across-model physical-limitation disparities at the same held-out threshold:

| model | demographic_parity_difference | equalized_odds_difference | equal_opportunity_difference | demographic_parity_ratio |
| --- | --- | --- | --- | --- |
| dummy | 0.000 | 0.000 | 0.000 | 1.000 |
| logistic | 0.076 | 0.148 | 0.025 | 0.921 |
| forest | 0.351 | 0.602 | 0.186 | 0.625 |


## 14. Experimental mitigation and comparison

Experimental reweighing uses development-set `P(group)P(Y)/P(group,Y)` weights for `physical_limitation` and retains the selected model architecture. No held-out labels or test fairness metrics select weights. Model remains evaluated at the original 0.5 threshold.

| model | accuracy | balanced_accuracy | selection_rate | recall_tpr | false_positive_rate | false_negative_rate |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.709 | 0.583 | 0.892 | 0.946 | 0.780 | 0.054 |
| reweighted | 0.702 | 0.567 | 0.913 | 0.957 | 0.823 | 0.043 |


| model | demographic_parity_difference | equalized_odds_difference | equal_opportunity_difference |
| --- | --- | --- | --- |
| original | 0.351 | 0.602 | 0.186 |
| reweighted | 0.252 | 0.438 | 0.134 |


Whether mitigation improved a targeted metric must be assessed against both figures and uncertainty, not assumed. Reweighing changes objective/sample influence, not labels or historical access patterns. A new degradation in another subgroup or metric remains possible.

## 15. Utility–fairness trade-offs

Accuracy and balanced accuracy measure prediction of observed utilization; group metrics measure particular rate relationships. The same threshold may yield more false negatives, more false positives, or both in different subgroups. Model and mitigation comparisons must consider all these rates jointly.

## 16. Limits of causal or ethical interpretation

The sample does not reveal why physician visits did or did not occur. Observed utilization is not medical need, access, quality, patient choice, or disparate treatment. Restricted recruitment, site cluster selection, correlated repeated measures, unobserved socioeconomic attributes and uncertain fractional physical-limitation coding prevent identification of discrimination. There is no operational decision rule or affected user population in this project.

## 17. Actionable next investigations

- Obtain original codebooks and participant/site identifiers; investigate the nonbinary physlm codes, household clusters, and temporal ordering before making inferential claims.

- Define a concrete deployment context, concrete harms, and valid labels; consult affected groups and clinical/domain experts before proposing fairness constraints.

- Replicate across site/year cohorts and person-disjoint splits; compute cluster-resampled confidence intervals and appropriate multiplicity adjustments.

- Conduct label-quality and measurement sensitivity studies; compare practical utility/fairness costs prospectively, never by repeated reuse of this test set.

## 18. References and environment

- RAND HIE statsmodels documentation: https://www.statsmodels.org/stable/datasets/generated/randhie.html

- RAND HIE ICPSR documentation: https://www.icpsr.umich.edu/web/NACDA/studies/6439/

- Fairlearn metric API and definitions: https://fairlearn.org/v0.13/api_reference/index.html

- AIF360 sklearn metric definitions: https://aif360.readthedocs.io/en/stable/modules/sklearn.html

- Captured package versions: `{"Jinja2": "3.1.6", "PyYAML": "6.0.3", "aif360": null, "fairlearn": "0.14.0", "matplotlib": "3.11.2", "numpy": "2.4.6", "pandas": "2.3.3", "scikit-learn": "1.8.0", "scipy": "1.17.1", "statsmodels": "0.15.0", "streamlit": null}`

- Run configurations, scores, uncertainty, warnings and dataset fingerprint: `run.json`. Fixed random seeds are documented, but cross-platform bitwise identity is not promised.


**Conclusion:** These measurements motivate focused investigation; they neither establish a model to be fair nor establish that discrimination occurred.
