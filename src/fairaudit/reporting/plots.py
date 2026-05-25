"""Generate figures from the run results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(result, dest):
    dest=Path(dest); dest.mkdir(parents=True, exist_ok=True)
    selected=result['selected_model']; blocks=result['models'][selected]['groups']
    for attribute in ['physical_limitation','plan_type']:
        rows=blocks[attribute]['rows']
        names=[r['group'] for r in rows]
        metrics=['selection_rate','tpr','fpr','fnr']
        x=np.arange(len(names)); width=.19
        fig,ax=plt.subplots(figsize=(9,5))
        for j,k in enumerate(metrics):
            ax.bar(x+(j-1.5)*width,[r[k] if r[k] is not None else np.nan for r in rows],width,label=k)
        ax.set_xticks(x,names,rotation=12,ha='right'); ax.set_ylim(0,1)
        ax.set_ylabel('Rate, proportion');ax.set_title(f'Selected model: {attribute} disaggregation')
        ax.legend(ncol=4,fontsize=8);fig.tight_layout()
        fig.savefig(dest/f'group_rates_{attribute}.png',dpi=145);plt.close(fig)
    sweeps=result['models'][selected]['threshold_sensitivity']
    fig,ax=plt.subplots(figsize=(9,5))
    for k in ['balanced_accuracy','dp_difference','eo_difference']:
        ax.plot([r['threshold'] for r in sweeps],[r[k] if r[k] is not None else np.nan for r in sweeps],marker='o',label=k)
    ax.set_ylim(0,1);ax.set_xlabel('Decision threshold (analysis only)')
    ax.set_ylabel('Performance or absolute disparity');ax.legend();ax.grid(alpha=.2)
    ax.set_title('Threshold sensitivity: selected model, physical limitation')
    fig.tight_layout();fig.savefig(dest/'threshold_sensitivity.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    data=result['validation']['group_counts']['physical_limitation']
    ax.bar([x['group'] for x in data],[x['n'] for x in data]);ax.set_ylabel('Rows')
    ax.set_title('Original audit population: physical limitation coding')
    fig.tight_layout();fig.savefig(dest/'group_population.png',dpi=145);plt.close(fig)
    return sorted(p.name for p in dest.glob('*.png'))
