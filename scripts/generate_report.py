#!/usr/bin/env python3
"""Re-render reports and figures strictly from existing run.json, no model refits."""
import argparse, json
from pathlib import Path
from fairaudit.reporting.findings import generate_report
from fairaudit.reporting.plots import generate_plots
p=argparse.ArgumentParser();p.add_argument('--results',default='results/latest');a=p.parse_args()
path=Path(a.results)
r=json.loads((path/'run.json').read_text(encoding='utf-8'))
print('Generated:',generate_report(r,path),generate_plots(r,path/'figures'))
