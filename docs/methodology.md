# Methods and pre-registered fairness questions

## Intended question

How does a model predicting recorded outpatient utilization differ in group outcomes and errors across **reported physical-limitation coding** and **individual deductible plan category**? Individuals recorded as positive may have used more health care or have greater need, but `mdvis>0` does not measure clinical appropriateness or unmet need. The positive outcome is a classifier prediction about observed past records only. If wrongly used to determine outreach or benefits, missed positives could create quality-of-service harms and false alarms could waste resources; the two errors do not have a documented equal cost.

## Split and selection protocol

The raw sample lacks person/site/year ID despite repeated records. Exact predictor-profile grouping reduces literal feature-profile crossover between train/validation/test but **is not a person-disjoint split** and is not sufficient to validate out-of-sample generalization across people or sites. `GroupShuffleSplit` uses a 25% intended test split, then development-side 25% validation. A deterministic seed and all disjoint index counts are recorded. Outcome is never hashed into profile IDs. All model scoring and group attribution on final test is performed only after model selection; training pipelines are fitted without test records. Models receive fixed hyperparameters, and the decision rule maximizes development-validation balanced accuracy with a predefined simplicity tie-break within 0.002. Test fairness and threshold-sweep values must not select a new champion.

## Preprocessing / quality

All original records retained; zero observed missingness is a finding, not an imputation assumption. Numeric median imputation and standard scaling are in sklearn `ColumnTransformer`/`Pipeline`, fit on training/development splits exclusively. Only seven specified continuous/transformed predictors are fitted. Source outcome count `mdvis`, `physlm`, `idp`, and derived group attributes are excluded. Raw fractional limitation-coded records are preserved, not recoded as yes/no without documented justification. The duplicate count is descriptive, not deduplication instructions.

## Uncertainty and inference

Per-group Wilson intervals estimate binomial-proportion uncertainty **under independent observations**. Within-group row bootstrap resamples and recomputes differences across the same fixed test predictions. In this dataset repeated participant/site correlations may violate independence, so these intervals are EXPLORATORY and must not be treated as calibrated study-level confidence coverage. A fully inferential analysis requires original participant/site/year identifiers and appropriate cluster resampling. Group intersections are displayed regardless of size, with warnings for fewer than 100 group records and fewer than 30 positive or negative labels. There is no null-hypothesis p-value or unadjusted repeated-testing significance claim.

## Sensitivity and model selection separation

DummyClassifier, logistic regression and random forest share held-out rows and the fixed 0.5 decision threshold. Models are compared both by utility and by group rates. A descriptive threshold sweep from 0.2 to 0.8 uses only the already fitted selected model. Should a threshold or fairness constraint be selected based on these results, evaluation must occur on a *new* untouched test sample. Proxy variables are possible even when direct group columns are excluded. Correlations are descriptive and not evidence of causal mediation or bias.
