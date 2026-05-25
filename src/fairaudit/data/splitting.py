"""Separation by full observed predictor profile to limit exact-profile overlap."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_profile_keys(df, columns):
    """Stable exact-observed-profile hashes; NOT a substitute for person IDs."""
    # Do not include outcome in hash: identical predictors with distinct labels stay together.
    return pd.util.hash_pandas_object(df[list(columns)].fillna(-999999), index=False).to_numpy()


def split_indices(df, cfg):
    mc = cfg['model']; dc = cfg['dataset']
    profile_cols = dc['features'] + dc['sensitive_source_columns']
    groups = make_profile_keys(df, profile_cols)
    outer = GroupShuffleSplit(n_splits=1, test_size=mc['test_fraction'], random_state=mc['seed'])
    dev, test = next(outer.split(df, df[dc['target']], groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=mc['validation_fraction_of_development'], random_state=mc['seed']+1)
    subtrain_local, val_local = next(inner.split(df.iloc[dev], groups=groups[dev]))
    train, val = dev[subtrain_local], dev[val_local]
    if set(groups[dev]).intersection(groups[test]):
        raise AssertionError('Exact observed profiles cross held-out boundary')
    if set(groups[train]).intersection(groups[val]):
        raise AssertionError('Exact observed profiles cross train/validation boundary')
    return {'development': dev, 'train': train, 'validation': val, 'test': test,
            'strategy': 'GroupShuffleSplit on hash of complete observed input profile (not a person ID)',
            'seed': mc['seed'], 'n_unique_profiles': len(np.unique(groups)),
            'person_disjoint': False}
