# Metric definitions and valid denominators

For label 1 = at least one recorded MD visit; predictions at threshold 0.5 use `score >= 0.5`. Every group record stores `N`, number of label-positive observations `(TP+FN)`, label-negative observations `(TN+FP)`, selected predictions `(TP+FP)` and TP/TN/FP/FN.

| Metric | Formula | Interpretation/caveat |
|---|---|---|
| Accuracy | `(TP+TN)/N` | Can hide minority-class errors |
| Balanced accuracy | `(TPR+TNR)/2` | Undefined if either actual class absent |
| Precision | `TP/(TP+FP)` | Undefined when nothing predicted positive |
| Recall / true positive rate | `TP/(TP+FN)` | Missed recorded visits, conditional on label=1 |
| Specificity / TNR | `TN/(TN+FP)` | True negative rate, conditional on label=0 |
| FPR | `FP/(FP+TN)` | False alarm rate, conditional on label=0 |
| FNR | `FN/(FN+TP)` | Miss rate, conditional on label=1 |
| F1 | `2TP/(2TP+FP+FN)` | Defined only for nonzero denominator |
| Selection rate | `(TP+FP)/N` | Predicted-positive outcome frequency, NOT entitlement |
| ROC-AUC | Probability of score-ranking positive ahead of negative | Undefined when a class is absent |
| DP difference | `max_g selection_g − min_g selection_g` | Max pairwise outcome disparity, signed direction omitted |
| DP ratio | `min_g selection_g / max_g selection_g` | Undefined if max is zero; all groups included |
| Equal opportunity difference | `max_g TPR_g − min_g TPR_g` | Does not measure FPR, prevalence or selection parity |
| Equalized odds difference | `max(max TPR−min TPR, max FPR−min FPR)` | Measures both types of conditional-rate disparities |
| Equalized odds ratio | `min(min TPR/max TPR, min FPR/max FPR)` | Undefined if any compared denominator is zero |

For all nonratio differences, undefined group denominators cause the comparison to be null, **not** calculated over a silently reduced subset. The independent engine uses manually confirmed contingency fixture values; official Fairlearn implementations are attached when importable and compared for three between-group differences. When the optional module is absent, the status field records that it is not installed.

AIF360 is run only for the exhaustive **binary** plan-type comparison; it uses `0_non_individual` as descriptive reference (not an ethical 'privileged' assertion), and `1_individual` as compared group. AIF360's signed `statistical_parity_difference` = selection(individual) − selection(nonindividual). Its `disparate_impact_ratio` = selection(individual) / selection(nonindividual), undefined when denominator 0. Its `equal_opportunity_difference` is the signed TPR difference. `average_odds_difference` is the signed mean of the FPR and TPR differences, **not** Fairlearn's equalized odds maximum absolute difference. Run AIF360 only when the optional extra is installed; do not silently substitute alternative values when it is absent. Neither the four-fifths rule nor any single ratio proves or disproves legal compliance.

For Wilson intervals and exploratory bootstrap caveats see `docs/methodology.md` and the report.
