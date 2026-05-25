"""Load public-domain RAND HIE sample from installed statsmodels (no network)."""
import hashlib
from importlib.metadata import version
import pandas as pd
from statsmodels.datasets import randhie

SOURCE_URL = 'https://www.statsmodels.org/stable/datasets/generated/randhie.html'


def load_dataset():
    frame = randhie.load_pandas().data.copy().reset_index(drop=True)
    # Construct outcome without discarding participants or silently coercing labels.
    if frame['mdvis'].isna().any() or (frame['mdvis'] < 0).any():
        raise ValueError('Invalid or missing source labels mdvis')
    frame['any_md_visit'] = (frame['mdvis'] > 0).astype('int8')
    # The distributed data contains undocumented fractional physlm values.
    # Retain ALL records, expose the ambiguous code as an explicit audit group.
    frame['physical_limitation'] = frame['physlm'].map({0.: '0_no', 1.: '1_yes'}).fillna('2_other_code')
    frame['plan_type'] = frame['idp'].map({0: '0_non_individual', 1: '1_individual'})
    if frame['plan_type'].isna().any():
        raise ValueError('Undocumented individual deductible plan category')
    return frame


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def provenance(frame):
    return {
        'identifier': 'statsmodels:randhie',
        'statsmodels_version': version('statsmodels'),
        'source_url': SOURCE_URL,
        'upstream_copyright': 'public domain (statsmodels documentation)',
        'rows': len(frame), 'columns': len(frame.columns),
        'input_sha256_pandas_hash': fingerprint(frame),
        'data_collected': 'RAND Health Insurance Experiment, 1974–1982; bundled study subset',
        'version_note': 'No unique stable upstream data release tag; version and SHA256 both recorded',
    }
