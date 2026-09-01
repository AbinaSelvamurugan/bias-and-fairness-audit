# FairAudit — Bias and Fairness Assessment of a Public Machine-Learning Dataset

**Run identifier:** `20260101T045721Z` | **Scope:** observational audit, not a discrimination determination

## 1. Executive findings

Selected `forest` on **training-side validation balanced accuracy**, not on final test fairness scores. Held-out records: **5872**; test accuracy **0.682**, balanced accuracy **0.559**, ROC-AUC **0.664**.

For physical-limitation coding, demographic-parity difference (maximum minus minimum group selection) = **0.147**; equalized-odds difference (maximal group TPR/FPR range) = **0.532**. These are descriptive associations only.

Fairlearn MetricFrame: **fairlearn_not_installed**. AIF360 plan-type comparison: **aif360_not_installed**.

## 2. Scope and intended use

**Research question:** In a historical RAND Health Insurance Experiment sample, do models that predict whether a recorded observation contains one or more outpatient physician visits make different errors and positive predictions across reported physical-limitation and insurance-plan subgroups?

The positive outcome (Y=1) is *any recorded physician visit*, NOT adequate access, clinical need, quality, desirability, or an entitlement. A positive prediction is not a recommendation for or against care. This is a retrospective demonstration and must not be deployed to allocate care or benefits.

## 3. Dataset provenance and usage conditions

Dataset `statsmodels.datasets.randhie`, installed statsmodels version **0.14.6**. Upstream: https://www.statsmodels.org/stable/datasets/generated/randhie.html. Original study: RAND Health Insurance Experiment (1974–1982), extracted subset, **20190 rows** and **10 original variables**. The dataset-specific statsmodels copyright states **public domain**. Source extraction differs slightly from published versions; no canonical standalone release identifier. Exact content fingerprint: `08e1a0bdc18d728cc909e9f7ff3d22845021fec82dca9d7b4172137f1bec76a3`.

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

**Split:** GroupShuffleSplit on hash of complete observed input profile (not a person ID), seed=20260101; n_development=14318, n_train=11003, n_validation=3315, n_test=5872. Validation score chooses model, after which each estimator is refit on development only. Exact matching observed feature profiles are not shared with test; without person/household identifiers a person-disjoint split is not possible.

Preprocessing: median imputation and scaling in a scikit-learn Pipeline, fitted only on development observations. Logistic, random forest and dummy use fixed settings recorded in `run.json`. All final evaluations use threshold=0.5; sensitivity thresholds are descriptive and not model selection.

Training-side validation balanced accuracy:

| model | balanced_accuracy |
| --- | --- |
| dummy | 0.500 |
| logistic | 0.533 |
| forest | 0.578 |


## 7. Sensitive attributes and group definitions

`physical_limitation`: 0_no (raw physlm=0), 1_yes (raw physlm=1), 2_other_code (nonzero fractional, undocumented); `plan_type`: 0_non_individual (raw idp=0), 1_individual (raw idp=1). These terms are descriptive, not assessments of advantage/privilege. The ambiguous-code group is retained everywhere. Intersectional groups are reported with explicit counts.

## 8. Overall predictive performance

| model | n | accuracy | balanced_accuracy | precision | recall_tpr | specificity_tnr | false_positive_rate | false_negative_rate | f1 | roc_auc | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy | 5872.000 | 0.668 | 0.500 | 0.668 | 1.000 | 0.000 | 1.000 | 0.000 | 0.801 | 0.500 | 1.000 |
| logistic | 5872.000 | 0.671 | 0.529 | 0.682 | 0.951 | 0.107 | 0.893 | 0.049 | 0.795 | 0.603 | 0.932 |
| forest | 5872.000 | 0.682 | 0.559 | 0.698 | 0.924 | 0.194 | 0.806 | 0.076 | 0.795 | 0.664 | 0.885 |


Selected model confusion matrix: TP=3627, FP=1570, TN=377, FN=298. Decision threshold = 0.5, no excluded labels.

## 9. Disaggregated group results (held-out counts)

**Physical limitation:**

| group | n | positive_labels | negative_labels | tp | fp | tn | fn | label_prevalence | selection_rate | tpr | fpr | fnr | selection_rate_wilson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0_no | 4904.000 | 3221.000 | 1683.000 | 2974.000 | 1414.000 | 269.000 | 247.000 | 0.657 | 0.895 | 0.923 | 0.840 | 0.077 | [0.8858807990965305, 0.9030607411891662] |
| 1_yes | 675.000 | 492.000 | 183.000 | 459.000 | 131.000 | 52.000 | 33.000 | 0.729 | 0.874 | 0.933 | 0.716 | 0.067 | [0.8469104491797795, 0.897004044312269] |
| 2_other_code | 293.000 | 212.000 | 81.000 | 194.000 | 25.000 | 56.000 | 18.000 | 0.724 | 0.747 | 0.915 | 0.309 | 0.085 | [0.6947084273504293, 0.7937678137247323] |


**Insurance plan:**

| group | n | positive_labels | negative_labels | tp | fp | tn | fn | label_prevalence | selection_rate | tpr | fpr | fnr | selection_rate_wilson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0_non_individual | 4344.000 | 3026.000 | 1318.000 | 2868.000 | 1057.000 | 261.000 | 158.000 | 0.697 | 0.904 | 0.948 | 0.802 | 0.052 | [0.8944063108953817, 0.9119708381350167] |
| 1_individual | 1528.000 | 899.000 | 629.000 | 759.000 | 513.000 | 116.000 | 140.000 | 0.588 | 0.832 | 0.844 | 0.816 | 0.156 | [0.8129067215975905, 0.8503472947692072] |


**Intersectional groups** (interpret small cells cautiously):

| group | n | positive_labels | negative_labels | selection_rate | tpr | fpr | fnr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0_no × 0_non_individual | 3602.000 | 2479.000 | 1123.000 | 0.924 | 0.955 | 0.857 | 0.045 |
| 0_no × 1_individual | 1302.000 | 742.000 | 560.000 | 0.813 | 0.817 | 0.807 | 0.183 |
| 1_yes × 0_non_individual | 480.000 | 359.000 | 121.000 | 0.842 | 0.919 | 0.612 | 0.081 |
| 1_yes × 1_individual | 195.000 | 133.000 | 62.000 | 0.954 | 0.970 | 0.919 | 0.030 |
| 2_other_code × 0_non_individual | 262.000 | 188.000 | 74.000 | 0.729 | 0.904 | 0.284 | 0.096 |
| 2_other_code × 1_individual | 31.000 | 24.000 | 7.000 | 0.903 | 1.000 | 0.571 | 0.000 |


## 10. Fairness metrics: definitions, conventions and assumptions

Selection rate = (TP+FP)/N. TPR=TP/(TP+FN); FPR=FP/(FP+TN); FNR=FN/(FN+TP); specificity=TN/(TN+FP). Precision=TP/(TP+FP). All are undefined when denominators are zero. Each rate conditions on the observed label, which can have measurement or selection biases.

Demographic parity difference = highest minus lowest group selection rate. Demographic parity ratio = min/max selection rate, if max>0. Equal opportunity difference = max minus min TPR. Equalized odds difference = max(range TPR, range FPR). Equalized odds ratio = min(min/max TPR, min/max FPR), where defined. Comparisons use all retained groups; no favorable/unfavorable group is presupposed.

| metric | value |
| --- | --- |
| selection_rate_difference | 0.147 |
| selection_rate_ratio | 0.835 |
| tpr_difference | 0.018 |
| tpr_ratio | 0.981 |
| fpr_difference | 0.532 |
| fpr_ratio | 0.367 |
| fnr_difference | 0.018 |
| fnr_ratio | 0.790 |
| accuracy_difference | 0.192 |
| accuracy_ratio | 0.775 |
| demographic_parity_difference | 0.147 |
| demographic_parity_ratio | 0.835 |
| equal_opportunity_difference | 0.018 |
| equalized_odds_difference | 0.532 |
| equalized_odds_ratio | 0.367 |


Fairlearn MetricFrame: `fairlearn_not_installed`; metric values from the independently implemented contingency engine are always identified as such. AIF360 binary plan comparison: `{'status': 'aif360_not_installed', 'detail': "No module named 'aif360'"}`. Signed AIF360 statistical-parity difference is individual minus non-individual; disparate-impact ratio is individual/non-individual. AIF360 signed average-odds difference = (TPR difference + FPR difference)/2 and is **not** Fairlearn's maximum-absolute difference.

**Sensitivity restricted to the two explicitly binary physical-limitation codes** (shown *in addition* to, not instead of, all-group results):

Used 5579 exact binary-code observations; 293 ambiguous-code observations excluded **only** from this sensitivity contrast.

| metric | value |
| --- | --- |
| demographic_parity_difference | 0.021 |
| equal_opportunity_difference | 0.010 |
| equalized_odds_difference | 0.124 |


## 11. Statistical uncertainty and sample size

Group Wilson score intervals are included for proportions conditional on their documented denominators. Exploratory within-group row bootstrap of 150 draws (0.950 nominal quantile interval) produced:

| metric | point_estimate | exploratory_row_bootstrap_interval | successful_replicates |
| --- | --- | --- | --- |
| demographic_parity_difference | 0.147 | [0.10974530090362958, 0.20690885558918834] | 150.000 |
| equalized_odds_difference | 0.532 | [0.43669990977540857, 0.6234308561596831] | 150.000 |
| equal_opportunity_difference | 0.018 | [0.005789489504431828, 0.06089510555644993] | 150.000 |


**Strong limitation:** no person or site IDs are available; the simple Wilson intervals and row bootstrap are not guaranteed calibrated for the true correlated/clustered study. Do not use them for a significance or legal-compliance claim. Intervals excluding zero would not resolve causality, multiplicity, or selection. Missing small-group signals do not establish parity.

## 12. Observed disparities and error trade-offs

Compare the tabled TPR and FPR separately before drawing conclusions. The largest absolute between-group difference is an index for further review, not evidence by itself of intentional or unlawful discrimination. Differences may reflect group-dependent utilization base rates, missing structure, uncertain variable coding, measurement error, confounding, or model miscalibration; causal mechanisms are not identified.

## 13. Threshold and alternative-model sensitivity

The following fixed thresholds are **post-hoc diagnostic views on the test scores**, not candidate thresholds used to choose a deployable decision rule:

| threshold | accuracy | balanced_accuracy | dp_difference | eo_difference | equal_opportunity_difference |
| --- | --- | --- | --- | --- | --- |
| 0.200 | 0.669 | 0.501 | 0.001 | 0.003 | 0.000 |
| 0.300 | 0.676 | 0.513 | 0.061 | 0.206 | 0.009 |
| 0.400 | 0.681 | 0.529 | 0.043 | 0.175 | 0.014 |
| 0.500 | 0.682 | 0.559 | 0.147 | 0.532 | 0.018 |
| 0.600 | 0.675 | 0.614 | 0.156 | 0.440 | 0.113 |
| 0.700 | 0.618 | 0.618 | 0.150 | 0.303 | 0.204 |
| 0.800 | 0.466 | 0.570 | 0.331 | 0.372 | 0.372 |


Across-model physical-limitation disparities at the same held-out threshold:

| model | demographic_parity_difference | equalized_odds_difference | equal_opportunity_difference | demographic_parity_ratio |
| --- | --- | --- | --- | --- |
| dummy | 0.000 | 0.000 | 0.000 | 1.000 |
| logistic | 0.062 | 0.078 | 0.050 | 0.938 |
| forest | 0.147 | 0.532 | 0.018 | 0.835 |


## 14. Experimental mitigation and comparison

Experimental reweighing uses development-set `P(group)P(Y)/P(group,Y)` weights for `physical_limitation` and retains the selected model architecture. No held-out labels or test fairness metrics select weights. Model remains evaluated at the original 0.5 threshold.

| model | accuracy | balanced_accuracy | selection_rate | recall_tpr | false_positive_rate | false_negative_rate |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.682 | 0.559 | 0.885 | 0.924 | 0.806 | 0.076 |
| reweighted | 0.681 | 0.554 | 0.897 | 0.933 | 0.826 | 0.067 |


| model | demographic_parity_difference | equalized_odds_difference | equal_opportunity_difference |
| --- | --- | --- | --- |
| original | 0.147 | 0.532 | 0.018 |
| reweighted | 0.140 | 0.475 | 0.024 |


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

- Captured package versions: `{"Jinja2": "3.1.6", "PyYAML": "6.0.3", "aif360": null, "fairlearn": null, "matplotlib": "3.10.8", "numpy": "2.3.5", "pandas": "2.2.3", "scikit-learn": "1.8.0", "scipy": "1.17.0", "statsmodels": "0.14.6", "streamlit": null}`

- Run configurations, scores, uncertainty, warnings and dataset fingerprint: `run.json`. Fixed random seeds are documented, but cross-platform bitwise identity is not promised.


**Conclusion:** These measurements motivate focused investigation; they neither establish a model to be fair nor establish that discrimination occurred.
