"""FairAudit — read-only dashboard of generated audit results."""
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'/'latest'
st.set_page_config(page_title='FairAudit | RAND HIE', layout='wide', page_icon='⚖️')

@st.cache_data

def load_artifacts(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

if not (RESULTS/'run.json').exists():
    st.error('No audit outputs found. Run `python scripts/run_audit.py` from repository root first.')
    st.stop()
r=load_artifacts(str(RESULTS/'run.json'))
models=r['models']
st.sidebar.title('FairAudit')
page=st.sidebar.radio('Page', ['Overview','Dataset analysis','Model performance','Fairness metrics',
                                 'Disparity analysis','Threshold sensitivity','Mitigation comparison','Findings and reports'])
model=st.sidebar.selectbox('Audited model', list(models), index=list(models).index(r['selected_model']))
a=st.sidebar.selectbox('Comparison groups', ['physical_limitation','plan_type','intersection'])
result=models[model]; groups=result['groups'][a]; rows=pd.DataFrame(groups['rows']); disparities=groups['disparities']
st.title(f'{page}')
st.caption(f"RAND HIE historical public-domain sample · selected: {r['selected_model']} · displayed: {model} · {r['test_n']} held-out records · decision threshold {r['config']['model']['threshold']}")


def pretty_rate(v):
    return 'Undefined' if v is None else f'{v:.3f}'


def show_metrics(m, labels):
    cs=st.columns(len(labels))
    for c,(title,field) in zip(cs,labels): c.metric(title,pretty_rate(m.get(field)))


def rate_chart(df, fields):
    plot_df=df.melt(id_vars=['group'], value_vars=fields,var_name='Metric',value_name='Rate')
    st.plotly_chart(px.bar(plot_df,x='group',y='Rate',color='Metric',barmode='group',range_y=[0,1],
                   title='Group-conditional rates (undefined denominators remain empty)'), use_container_width=True)

if page=='Overview':
    st.subheader('Observed utilization, not health need')
    st.write('Predicts whether an observation contains at least one outpatient MD visit. This historical sample does not validate any care-allocation decision.')
    show_metrics(result['overall'], [('Accuracy','accuracy'),('Balanced accuracy','balanced_accuracy'),
              ('ROC-AUC','roc_auc'),('False-negative rate','false_negative_rate')])
    show_metrics(disparities,[('Demographic parity Δ','demographic_parity_difference'),
             ('Equalized odds Δ','equalized_odds_difference'),('Equal opportunity Δ','equal_opportunity_difference')])
    st.warning('A disparity is a diagnostic, not a causal or legal conclusion. Group coding, absence of participant IDs, and correlated observations restrict inference.')
    st.caption(f"Fairlearn MetricFrame: {groups['fairlearn']['status']} · AIF360: {result['aif360_plan_type']['status']}")

elif page=='Dataset analysis':
    v=r['validation']
    c1,c2=st.columns(2)
    c1.metric('Dataset rows',v['rows']);c2.metric('Repeated observed profiles',v['duplicates_non_target'])
    st.subheader('Target class balance')
    st.bar_chart(pd.DataFrame([{'label':str(k),'count':count} for k,count in v['target_counts'].items()]).set_index('label'))
    st.subheader('Group size and recorded-visit prevalence')
    for cat, data in v['group_counts'].items():
        st.markdown(f'**{cat}**'); df=pd.DataFrame(data)
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(df,x='group',y='n',title=f'{cat}: observation counts'),use_container_width=True)
    st.subheader('Missing values')
    st.dataframe(pd.DataFrame(list(v['n_missing_by_column'].items()),columns=['Column','Missing']),hide_index=True)
    st.subheader('Feature distributions (aggregate histograms)')
    f=st.selectbox('Predictor',list(v['feature_histograms']))
    d=v['feature_histograms'][f]
    mid=[(left+right)/2 for left,right in zip(d['bin_edges'][:-1],d['bin_edges'][1:])]
    st.plotly_chart(px.bar(x=mid,y=d['counts'],labels={'x':f,'y':'Rows'},title=f'Binned distribution: {f}'),use_container_width=True)
    st.subheader('Feature correlations (descriptive, not causal)')
    st.dataframe(pd.DataFrame(v['pearson_correlations']).round(3),use_container_width=True)
    with st.expander('Known limitations and data-quality warnings',expanded=True):
        for w in v['warnings']:st.write('• '+w)

elif page=='Model performance':
    st.subheader('Overall held-out performance')
    st.dataframe(pd.DataFrame([{'model':k,**m['overall']} for k,m in models.items()]).drop(columns='confusion_matrix'),hide_index=True,use_container_width=True)
    st.subheader('Fixed-threshold confusion matrix')
    cm=result['overall']['confusion_matrix']; mat=[[cm['tn'],cm['fp']],[cm['fn'],cm['tp']]]
    st.plotly_chart(px.imshow(mat,text_auto=True,x=['predicted 0','predicted 1'],y=['actual 0','actual 1'],aspect='auto'),use_container_width=True)
    st.subheader('Training-side validation')
    st.dataframe(pd.DataFrame([{'model':k,'balanced_accuracy':v} for k,v in r['validation_scores'].items()]),hide_index=True)
    st.json(r['config']['model']);st.caption('Final test set never used to choose a model or fit preprocessing.')

elif page=='Fairness metrics':
    st.subheader(f'Full counts and denominators — {a}')
    st.dataframe(rows,hide_index=True,use_container_width=True)
    rate_chart(rows,['selection_rate','tpr','fpr','fnr'])
    st.subheader('Between-group differences and ratios')
    st.dataframe(pd.DataFrame([{'metric':k,'value':v} for k,v in disparities.items() if isinstance(v,(float,int)) or v is None]),hide_index=True)
    st.caption('Δ=max−min; ratio=min/max. TPR conditions on true positive labels; FPR on true negative labels. A missing rate is undefined, not zero.')
    st.json(groups['fairlearn'])
    if a=='plan_type': st.json(result['aif360_plan_type'])

elif page=='Disparity analysis':
    d=[]
    for k,entry in models.items():
        for name,block in entry['groups'].items():
            d.append({'model':k,'attribute':name,**{m:block['disparities'][m] for m in ['demographic_parity_difference','equalized_odds_difference','equal_opportunity_difference']}})
    full=pd.DataFrame(d)
    st.dataframe(full,hide_index=True,use_container_width=True)
    chart=full[full['attribute']==a].melt(id_vars=['model','attribute'],value_vars=['demographic_parity_difference','equalized_odds_difference','equal_opportunity_difference'],var_name='measure',value_name='difference')
    st.plotly_chart(px.bar(chart,x='model',y='difference',color='measure',barmode='group',title=f'Model comparisons — {a}'),use_container_width=True)
    st.info('Differences are flagged for investigation, not classified automatically as discrimination. Compare prevalence, subgroup n and uncertainty before drawing conclusions.')

elif page=='Threshold sensitivity':
    st.subheader('Test-set threshold sweep (exploratory only)')
    sweep=pd.DataFrame(result['threshold_sensitivity'])
    st.dataframe(sweep,hide_index=True,use_container_width=True)
    x=sweep.melt(id_vars='threshold',value_vars=['balanced_accuracy','dp_difference','eo_difference','equal_opportunity_difference'],var_name='measure',value_name='value')
    st.plotly_chart(px.line(x,x='threshold',y='value',color='measure',markers=True,title='Performance and physical-limitation disparity'),use_container_width=True)
    st.warning('Do not pick a new threshold from the held-out test sweep and then treat its test metrics as independently validated.')

elif page=='Mitigation comparison':
    if 'mitigation' not in r:
        st.info('Optional mitigation was not run. Execute `python scripts/compare_mitigation.py` to compare training-only reweighing.')
    else:
        m=r['mitigation']; original=models[r['selected_model']]
        d=pd.DataFrame([{'model':'baseline',**original['overall']},{'model':'reweighted',**m['overall']}])
        st.subheader('Held-out model utility');st.dataframe(d,hide_index=True,use_container_width=True)
        fair=pd.DataFrame([{'model':'baseline',**original['groups']['physical_limitation']['disparities']},
          {'model':'reweighted',**m['groups']['physical_limitation']['disparities']}])
        st.subheader('Group fairness measures');st.dataframe(fair,hide_index=True,use_container_width=True)
        st.subheader('Group error-rate changes');base=pd.DataFrame(original['groups']['physical_limitation']['rows']);new=pd.DataFrame(m['groups']['physical_limitation']['rows']);
        vals=base[['group','fpr','fnr','tpr']].merge(new[['group','fpr','fnr','tpr']],on='group',suffixes=('_original','_mitigated'))
        st.dataframe(vals,hide_index=True,use_container_width=True)
        st.json(m['weights_policy'])
        st.caption('Reweighing was learned from development records only; changes to all rates and utility must be considered.')

elif page=='Findings and reports':
    st.subheader('Generated from current run.json')
    report=RESULTS/'findings.md'
    if report.exists():
        st.markdown(report.read_text(encoding='utf-8'))
        st.download_button('Download Markdown report',report.read_bytes(),file_name='findings.md',mime='text/markdown')
    else: st.warning('Run `python scripts/generate_report.py` to generate the report.')
    html=RESULTS/'findings.html'
    if html.exists(): st.download_button('Download HTML report',html.read_bytes(),file_name='findings.html',mime='text/html')
    st.download_button('Download structured results',json.dumps(r,indent=2),file_name='run.json',mime='application/json')
