"""Data-quality summary; never silently drop rows or undocumented groups."""
import numpy as np
import pandas as pd


def validate(frame, cfg):
    d = cfg['dataset']
    needed = set(d['features'] + d['sensitive_source_columns'] + d['audit_attributes'] + [d['target'], d['target_source']])
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f'Required columns missing: {sorted(missing)}')
    if len(frame) != d['raw_rows_expected']:
        raise ValueError(f'Unexpected row count: {len(frame)} != {d["raw_rows_expected"]}')
    if frame[d['target']].isna().any():
        raise ValueError('Target missing; audit does not silently drop labels')
    if not frame[d['target']].isin([0, 1]).all():
        raise ValueError('Target must be binary 0/1')
    for col in d['features']:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f'Non-numeric predictor: {col}')
        if (np.isinf(frame[col].to_numpy(dtype=float))).any():
            raise ValueError(f'Infinite predictor: {col}')
    for binary in ('idp','hlthg','hlthf','hlthp'):
        if not frame[binary].isin([0,1]).all():
            raise ValueError(f'Unexpected category in documented binary field: {binary}')
    if (frame[['hlthg','hlthf','hlthp']].sum(axis=1)>1).any():
        raise ValueError('Contradictory self-rated health binary indicators')
    if frame[d['audit_attributes']].isna().any().any():
        raise ValueError('Missing audit-group identity; retain explicit category instead')
    counts = {}
    for a in d['audit_attributes']:
        g = frame.groupby(a, dropna=False, observed=True)[d['target']].agg(['size','sum','mean'])
        counts[a] = [{'group': str(k), 'n': int(r['size']), 'positive_labels': int(r['sum']),
                      'prevalence': float(r['mean'])} for k,r in g.iterrows()]
    # Distinguish exact original-record duplicates from predictor-profile repetition.
    # The raw label mdvis must NOT be included in a non-target-profile duplicate count.
    duplicates_exact = int(frame.drop(columns=[d['target'], *d['audit_attributes']], errors='ignore').duplicated().sum())
    profile_cols=d['features'] + d['sensitive_source_columns']
    duplicates = int(frame[profile_cols].duplicated().sum())
    # Aggregated distributions are safe to include in dashboard outputs (no row exports).
    feature_histograms={}
    for col in d['features']:
        finite=frame[col].dropna().to_numpy(dtype=float)
        counts_hist, bins_hist=np.histogram(finite,bins=12) if len(finite) else (np.array([],dtype=int),np.array([]))
        feature_histograms[col]={'bin_edges':bins_hist.tolist(),'counts':counts_hist.astype(int).tolist()}
    warnings = []
    if (frame['physical_limitation'] == '2_other_code').any():
        warnings.append('physlm has fractional, undocumented codes: retained as 2_other_code, not imputed to disability status')
    if duplicates:
        warnings.append(f'{duplicates} rows repeat predictor+group-source profiles; {duplicates_exact} are exact original-row duplicates (may reflect repeat observations; person IDs unavailable)')
    warnings.append('No participant/household/time identifiers are present, so independence and person-disjoint separation are not supported')
    warnings.append('Health self-report timing relative to outpatient visits is not documented (potential temporal leakage)')
    if (frame['disea']%1 != 0).any():
        warnings.append('disea is described as a disease count, but the packaged sample includes fractional transformed values; original coding is not documented')
    warnings.append('Dataset is an observational, historical experimental subset and not a deployable modern care-access target')
    return {
        'rows': len(frame), 'features': d['features'], 'n_missing_by_column': frame.isna().sum().astype(int).to_dict(),
        'duplicates_non_target': duplicates, 'duplicates_exact_rows': duplicates_exact, 'target_counts': frame[d['target']].value_counts().sort_index().to_dict(),
        'target_prevalence': float(frame[d['target']].mean()), 'groups': counts,
        'predictor_summary': frame[d['features']].describe().round(5).to_dict(),
        'pearson_correlations': frame[d['features']].corr().round(4).to_dict(),
        'feature_histograms': feature_histograms,
        'nonstandard_physlm_values': sorted(float(v) for v in frame['physlm'].unique() if v not in (0,1)),
        'identifier_leakage': 'No participant ID is present; no identifier included as feature',
        'label_leakage': 'mdvis excluded from predictors; some health indicators may overlap the same period as the outcome',
        'warnings': warnings,
    }
