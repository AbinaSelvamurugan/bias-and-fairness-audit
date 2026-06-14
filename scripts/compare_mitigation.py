#!/usr/bin/env python3
"""Recompute test comparison with optional train-only group/label weighting."""
from fairaudit.audit import run_audit
r=run_audit(mitigation=True)
print('Results at results/latest. Original vs mitigation:')
for m in [r['models'][r['selected_model']],r['mitigation']]:
 print(m['overall']['balanced_accuracy'],m['groups']['physical_limitation']['disparities']['demographic_parity_difference'])
