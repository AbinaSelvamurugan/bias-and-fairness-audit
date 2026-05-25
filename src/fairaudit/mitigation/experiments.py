"""Experimental train-side group x label reweighting; NO held-out labels used."""
import numpy as np


def label_group_weights(df, split, group_col, target):
    dev=df.iloc[split['development']]
    n=len(dev); g=dev[group_col].astype(str); y=dev[target]
    p_g=g.value_counts(normalize=True); p_y=y.value_counts(normalize=True)
    p_joint=dev.groupby([group_col,target], observed=True).size()/n
    cell={}
    for (group,label),joint in p_joint.items():
        cell[(str(group),int(label))]=float(p_g[str(group)]*p_y[label]/joint)
    weights=np.ones(len(df),dtype=float)
    for idx in split['development']:
        row=df.iloc[idx]; weights[idx]=cell[(str(row[group_col]),int(row[target]))]
    return weights, {'group':group_col,'formula':'P(group)*P(label)/P(group,label), development only',
        'cell_weights':{str(k):v for k,v in cell.items()},
        'source_indices':'development only (used for refit and validation subtrain; no held-out test labels)'}
