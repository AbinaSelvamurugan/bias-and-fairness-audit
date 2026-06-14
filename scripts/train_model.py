#!/usr/bin/env python3
"""Fit and persist three train/validation-selected models; run complete audit as integrity check."""
import logging
from fairaudit.audit import run_audit
logging.basicConfig(level=logging.INFO)
r=run_audit()
print('Saved fitted pipelines for:',list(r['models']), '| selected:',r['selected_model'])
