#!/usr/bin/env python3
"""Run complete reproducible audit and emit all artifacts."""
import argparse
import logging
from fairaudit.audit import run_audit
p=argparse.ArgumentParser()
p.add_argument('--output', default=None, help='Artifact directory (default results/latest)')
p.add_argument('--mitigation',action='store_true',help='Experimental training-only group-label reweighting')
p.add_argument('--bootstrap-repeats',type=int,default=None)
a=p.parse_args()
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
r=run_audit(a.output,a.mitigation,a.bootstrap_repeats)
chosen=r['selected_model']; x=r['models'][chosen]
print('Selected:', chosen, '| test N:',r['test_n'],'| balanced accuracy:',round(x['overall']['balanced_accuracy'],4))
print('Physical limitation disparity:',x['groups']['physical_limitation']['disparities']['demographic_parity_difference'])
print('Fairlearn:',x['groups']['physical_limitation']['fairlearn']['status'],'| AIF360:',x['aif360_plan_type']['status'])
