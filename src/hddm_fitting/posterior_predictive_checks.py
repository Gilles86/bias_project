import matplotlib
matplotlib.use('Agg')
import os
from kabuki.analyze import gelman_rubin
import pandas
import hddm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = '/home/gdholla1/projects/bias/data/hddm_fits'

def get_model(model):
    df = pandas.read_pickle('/home/gdholla1/projects/bias/data/behavior/behavior.pandas')

    # Prepare data
    data = df[['cue_validity', 'rt', 'correct', 'difficulty', 'subj_idx']]
    data['response'] = data['correct']
    data['rt'] = data['rt'] / 1000.
    data = data[(data.rt > .2) & (data.rt < 1.5)]
    data = data[~data.rt.isnull()]
    data['cue_coding'] = data.cue_validity.map({'valid':1, 'neutral':0, 'invalid':-1})

    def z_link_func(x, data=data):
        return 1 / (1 + np.exp(-(x.values.ravel())))

    regs, free_pars = model.split('_')

    if regs == 'drift':
        z_reg = {'model': 'z ~ 0', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty) + cue_coding', 'link_func': lambda x: x}
    elif regs == 'startpoint':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)', 'link_func': lambda x: x}
    elif regs == 'both':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty) + cue_coding', 'link_func': lambda x: x}

    reg_descr = [z_reg, v_reg]

    if free_pars == 'sv':
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])
    elif free_pars == 'sz':
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sz'), bias=True, group_only_regressors=False, group_only_nodes=['sz'])
    elif free_pars == 'svsz':
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz'])
    elif free_pars == 'all':
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz', 'st'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz', 'st'])
    
    
    # NOW FIND, LOAD, AND CONCAT TRACES
    reg = re.compile('.*/traces_{model}_[0-9]+.pkl'.format(**locals()))
    fns = [fn for fn in glob.glob('/home/gdholla1/projects/bias/data/hddm_fits/traces_{model}_*.pkl'.format(**locals())) if reg.match(fn)]

    hddm_model.load_db(fns[0], db='pickle')

    for fn in fns[1:]:
        traces = pkl.load(open(fn))    
        for node in hddm_model.get_stochastics().node:
            node.trace._trace[0] = np.concatenate([traces[str(node)][0], node.trace[:]])        


    return hddm_model

def get_quantile_dataframe(d, q=(0.1, 0.3, 0.5, 0.7, 0.9)):
    q_lb, q_ub, prop_ub = hddm.utils.data_quantiles(d)
    
    df_ub = pandas.DataFrame({'q_rt':q_ub, 'prop':np.array(q)*prop_ub, 'q':q})
    df_ub['bound'] = 'ub'
    
    df_lb = pandas.DataFrame({'q_rt':q_lb, 'prop':np.array(q)*(1-prop_ub), 'q':q})    
    df_lb['bound'] = 'lb'
    
    return pandas.concat((df_ub, df_lb))

models = ['drift_all', 'drift_sv', 'drift_sz', 'drift_svsz',
          'startpoint_all', 'startpoint_sv', 'startpoint_sz', 'startpoint_svsz',
          'both_all', 'both_sv', 'both_sz', 'both_svsz',
          'drift_none', 'startpoint_none', 'both_none']

import glob
import re

import pickle as pkl

sns.set_context('paper')
sns.set_style('whitegrid')

#for model_str in models[1:]:
for model_str in models:
    print model_str
    try: 
        model = get_model(model_str)

        fn = '../../data/hddm_fits/ppc_{}.pkl'.format(model_str)

        if os.path.exists(fn):
            ppc_data = pandas.read_pickle(fn)
        else:
            ppc_data = hddm.utils.post_pred_gen(model, samples=500)
            ppc_data.to_pickle('../../data/hddm_fits/ppc_{}.pkl'.format(model_str))

        # Make a column that make clear which nth sample from the posterior was used
        ppc_data['sample'] = ppc_data.index.get_level_values(1)


        # Merge with real data, so we know the corresponding trial conditions
        ppc_data_merged = model.data[['cue_validity', 'difficulty', 'subj_idx']].merge(ppc_data.set_index(ppc_data.index.get_level_values(2)), left_index=True, right_index=True)

        # Make quantiles per subject
        data_quantiles = model.data.groupby(['subj_idx', 'cue_validity', 'difficulty']).apply(get_quantile_dataframe)

        # Mean quantiles over subjects
        data_quantiles = data_quantiles.reset_index().groupby(['cue_validity', 'difficulty', 'q', 'bound'], as_index=False).mean()

        model_quantiles_subj = ppc_data_merged.groupby(['sample', 'subj_idx', 'cue_validity', 'difficulty']).apply(get_quantile_dataframe)
        model_quantiles = model_quantiles_subj.reset_index().groupby(['sample', 'cue_validity', 'difficulty', 'q', 'bound'], as_index=False).mean()

        def plot_order(x, y, **kwargs):
            y = y.values[np.argsort(x.values)]
            x = np.sort(x.values)
            plt.plot(x, y, **kwargs)
            
            plt.scatter(x[:2], y[:2], marker='x', s=40,  color='white', edgecolor='white', linewidth='2', alpha=.7)
            plt.scatter(x[2:], y[2:], marker='o', s=40,  color='white', edgecolor='white', linewidth='2', alpha=.7)


        def hexbin(x, y, color, **kwargs):
            plt.hexbin(x, y, **kwargs)

        sns.set_context('paper')

        data_quantiles['cue congruency'] = data_quantiles['cue_validity'].map({'invalid':'incongruent', 'neutral':'neutral', 'valid':'congruent'})
        model_quantiles['cue congruency'] = model_quantiles['cue_validity'].map({'invalid':'incongruent', 'neutral':'neutral', 'valid':'congruent'})

        fac_model = sns.FacetGrid(model_quantiles, col='cue congruency')
        fac_model.map(hexbin, 'prop_bound', 'q_rt', gridsize=50, cmap=plt.cm.inferno, extent=[0, 1, 0.5, 1.3])

        fac = sns.FacetGrid(data_quantiles, col='cue congruency', hue='q')
        fac.fig = fac_model.fig
        fac.axes = fac_model.axes
        # fac.map(plot_order, 'prop_bound', 'q_rt', lw=3, color='w', marker='o', markeredgewidth=2, alpha=.7)
        fac.map(plot_order, 'prop_bound', 'q_rt', lw=3, color='w', markeredgewidth=2, alpha=.7)
        fac.set_xlabels('Response proportion')
        fac.set_ylabels('Response time quantiles')
        fac.set_titles('{col_name}')
        fac.fig.savefig('/home/gdholla1/projects/bias/reports/hddm_fit_model_{model_str}.pdf'.format(**locals()))

        # fac.fig.set_size_inches(30, 25)
    except Exception as e:
        print model_str, e
