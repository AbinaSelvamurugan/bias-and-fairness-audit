"""Conventional predictive utility; nan for undefined precision/rates."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_auc_score)


def safe_div(n,d):
    return float(n/d) if d else None


def performance(y_true, scores, threshold=0.5):
    y=np.asarray(y_true, dtype=int)
    if not set(np.unique(y)).issubset({0,1}): raise ValueError('Expected binary labels')
    if len(y)!=len(scores) or not len(y): raise ValueError('Missing or inconsistent predictions')
    scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Probabilities must be finite within [0,1]')
    yp=(scores >= threshold).astype(int)
    tn,fp,fn,tp=(int(x) for x in confusion_matrix(y,yp,labels=[0,1]).ravel())
    return {'n':len(y), 'threshold':float(threshold), 'positive_label':1,
      'accuracy':float(accuracy_score(y,yp)),
      'balanced_accuracy':safe_div(safe_div(tp,tp+fn) + safe_div(tn,tn+fp), 2) if (tp+fn and tn+fp) else None,
      'precision':safe_div(tp,tp+fp), 'recall_tpr':safe_div(tp,tp+fn),
      'f1':safe_div(2*tp,2*tp+fp+fn), 'specificity_tnr':safe_div(tn,tn+fp),
      'false_positive_rate':safe_div(fp,fp+tn), 'false_negative_rate':safe_div(fn,fn+tp),
      'selection_rate':safe_div(tp+fp,len(y)),
      'roc_auc':float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
      'confusion_matrix':{'tn':tn,'fp':fp,'fn':fn,'tp':tp}}
