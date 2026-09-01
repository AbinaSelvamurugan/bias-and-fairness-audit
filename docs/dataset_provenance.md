# Dataset provenance, dictionary, and permitted scope

**Name:** RAND Health Insurance Experiment (`randhie`), distributed with `statsmodels.datasets.randhie` (10-column reduced sample), 20,190 observations. **Original study:** 1974–1982, RAND Corporation; methods and sample documentation: https://www.icpsr.umich.edu/web/NACDA/studies/6439/. **Distributed extract:** https://www.statsmodels.org/stable/datasets/generated/randhie.html ; Cameron and Trivedi *Microeconometrics: Methods and Applications* (2005), a subset that may differ from their original sample. **License:** public domain according to statsmodels dataset-specific `COPYRIGHT`; statsmodels code independently uses BSD-3-Clause. **Version identifier:** installed statsmodels version and a reproducible pandas-source-content SHA256, captured in `run.json`; no stable dataset tag found.

Study population is not modern US population: sites included Seattle, Dayton, Charleston, Fitchburg-Leominster, Franklin County (MA), Georgetown County (SC); eligibility excluded Medicare-eligible individuals, families above an income limit, military households and institutionalized people. The fuller trial used cluster sampling and repeated participant-year observations. This reduced extract has **no person, time, or site IDs**; its rows cannot be assumed independent and cannot support treatment-effect estimates.

## Variable dictionary and processing

| Raw variable | Reported meaning (source) | Use and reason |
|---|---|---|
| `mdvis` | Number of outpatient visits to a medical doctor | Outcome source only; `any_md_visit=(mdvis>0)`, not a feature |
| `lncoins` | log(coinsurance+1) | Predictor, plan-related exposure proxy |
| `idp` | Individual deductible plan indicator | Audit-only `plan_type`, excludes direct group indicator from predictors |
| `lpi` | log(max(1,annual participation incentive payment)) | Predictor, study-payment exposure |
| `fmde` | transformed MDE/coinsurance term (0 if `idp=1`) | Predictor, correlated plan proxy; document proxy limitations |
| `physlm` | 1 when physical limitation is recorded per dataset description | Audit-only; exact 0 and 1 separate, fractional codes preserved as `2_other_code` |
| `disea` | Number of chronic diseases | Predictor; transformed values may be noninteger due original extract processing |
| `hlthg` | Indicator health rated good | Predictor, possible timing/label-information issue |
| `hlthf` | Indicator health rated fair | Predictor, possible timing/label-information issue |
| `hlthp` | Indicator health rated poor | Predictor, possible timing/label-information issue; omitted self-report category is excellent |

**Raw schema audit:** all records retained, numeric predictors checked; no observed missing values in the shipped extract. The dataset has surprising fractional `physlm` values even though binary coding is stated; their original meaning is not documented. No raw row-level data is copied into this repository; statsmodels is the distribution/access requirement.

**Sensitive use:** limitation indicator is disability-related and used strictly to evaluate group outcomes, not to fit the estimator. Plan-type group is a study-treatment category, not a protected demographic. No demographic sex/race columns are available in this *reduced* sample; do not pretend it supports a racial or sex-based audit. The records are historical health utilization and should not be linked, reidentified, or treated as a contemporary clinical training corpus.

**Other dataset considered:** UCI Adult Income (https://archive.ics.uci.edu/dataset/2/adult), documented CC BY 4.0, 48,842 rows and sex/race attributes. Randhie was selected because its public-domain data ships with the installed, versioned statsmodels package and loads without network access. Different societal/ethical questions would warrant other datasets.
