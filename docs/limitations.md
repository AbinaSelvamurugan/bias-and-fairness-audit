# Audit limitations and open questions

1. **Original subject IDs not available.** RAND HIE has longitudinal structure and clustered recruitment; no participant, household, year or site fields exist in the distributed ten-column sample. Exact-profile grouping does not establish subject separation, and row-level uncertainty is not cluster valid.
2. **Uninterpreted physical-limitation codes.** Several `physlm` codes lie strictly between 0 and 1, contrary to its binary documented meaning. They are retained as a separately assessed group. This may capture original recoding, imputation, or transformation; none of those is documented here.
3. **Construct validity.** Recorded utilization does not measure medical need, quality, harm, access, or appropriate care. Error asymmetry depends on a use case that is absent from the dataset.
4. **Collection and selection.** Historic geographically restricted RAND study and nonrepresentative eligibility, insurance plan assignment, unknown treatment timing, potential correlation and health-system change. Cannot generalize to present-day populations.
5. **Features, proxies and label leakage.** Plan-related transforms and self-rated health may correlate with group attributes or with contemporaneous visits. Original temporal order cannot be checked; dropping explicit group columns is not removal of proxies.
6. **No causal identification.** Differences in observed labels, rates or errors alone cannot prove discrimination, explain its causes, identify interventions, or satisfy a legal standard.
7. **Threshold/model reuse.** Post-hoc test threshold exploration and exploratory mitigation comparisons consume the same holdout. A fresh test sample must assess any model/threshold selected using them.
8. **Uncertainty.** Wilson and row-bootstrap intervals assume independent observations. Their nominal confidence level cannot be interpreted as valid subject/site-cluster coverage, and cross-group multiple comparisons are not adjusted. Rare intersection estimates can be unstable or undefined.
9. **Optional components.** The Fairlearn MetricFrame integration and the AIF360 signed pairwise metrics are optional: they run when those packages are installed, and `pytest tests/integration/test_optional_libraries.py -q` exercises them. The dashboard requires `streamlit`.

Before operational interpretation, recover original documentation and richer data with subject/site IDs, audit label meaning with subject-matter experts and affected groups, define concrete harms, then rerun validation on an independent clustered sample.
