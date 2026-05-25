"""Fit-on-train-only pipelines; test data is never used for model selection."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


def create_pipeline(kind, cfg, features):
    par = cfg['model']['models'][kind]
    prep = ColumnTransformer([('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                           ('scale', StandardScaler())]), features)], remainder='drop')
    if kind == 'dummy': est = DummyClassifier(**par)
    elif kind == 'logistic': est = LogisticRegression(random_state=cfg['model']['seed'], **par)
    elif kind == 'forest': est = RandomForestClassifier(random_state=cfg['model']['seed'], **par)
    else: raise ValueError(f'Unknown classifier: {kind}')
    return Pipeline([('preprocess', prep), ('classifier', est)])


def fit_models(df, split, cfg, weights=None):
    feat=cfg['dataset']['features']; target=cfg['dataset']['target']; threshold=cfg['model']['threshold']
    models={}; val_scores={}
    for k in cfg['model']['models']:
        pipe=create_pipeline(k,cfg,feat)
        fit_kwargs = {'classifier__sample_weight': weights[split['train']]} if weights is not None else {}
        pipe.fit(df.iloc[split['train']][feat], df.iloc[split['train']][target], **fit_kwargs)
        pred=(pipe.predict_proba(df.iloc[split['validation']][feat])[:,1] >= threshold).astype(int)
        val_scores[k]=float(balanced_accuracy_score(df.iloc[split['validation']][target], pred))
    best = max(val_scores.values())
    # Preannounced simplicity-based tie-breaking, never based on test outcomes or fairness.
    chosen = next(k for k in ('dummy','logistic','forest') if val_scores[k] >= best-0.002)
    for k in cfg['model']['models']:
        pipe=create_pipeline(k,cfg,feat)
        fit_kwargs = {'classifier__sample_weight': weights[split['development']]} if weights is not None else {}
        pipe.fit(df.iloc[split['development']][feat], df.iloc[split['development']][target], **fit_kwargs)
        models[k]=pipe
    return models, val_scores, chosen
