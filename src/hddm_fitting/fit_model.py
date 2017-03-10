import argparse
import os
from kabuki.analyze import gelman_rubin
import pandas
import hddm
import numpy as np

path = '/home/gholland/projects/bias/data/hddm_fits'

# SELECT MASK
models = ['drift_all', 'drift_sv', 'drift_sz', 'drift_svsz',
          'startpoint_all', 'startpoint_sv', 'startpoint_sz', 'startpoint_svsz',
          'both_all', 'both_sv', 'both_sz', 'both_svsz',
          'drift_none', 'startpoint_none', 'both_none']

if 'PBS_ARRAYID' in os.environ:
    model_idx = int(os.environ['PBS_ARRAYID'])
else:
    model_idx = 0

model = models[model_idx]


def get_model(model):
    df = pandas.read_pickle('/home/gholland/projects/bias/data/behavior/behavior.pandas')

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
    elif free_pars == 'none':
        hddm_model = hddm.HDDMRegressor(data, reg_descr, bias=True, group_only_regressors=False)


    return hddm_model 

def fit_model(id, model=model, path=path):
    import numpy as np
    import hddm
    import pandas
    import os
    
    df = pandas.read_pickle('/home/gholland/projects/bias/data/behavior/behavior.pandas')

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
    elif free_pars == 'none':
        hddm_model = hddm.HDDMRegressor(data, reg_descr, bias=True, group_only_regressors=False)

    hddm_model.find_starting_values()
    db_fn = os.path.join(path, 'traces_{}_{}.pkl'.format(model, id))

    hddm_model.sample(20000, 1000, dbname=db_fn, db='pickle')
    return db_fn


from IPython.parallel import Client

json_path = os.path.join(os.environ['TMPDIR'], 'hddm'.format(model, id), 'security', 'ipcontroller-client.json')
v = Client(json_path)[:]

jobs = v.map(fit_model, range(15)) 

db_fns = jobs.get()
hddm_model = get_model(model, mask)
models = [hddm_model.load_db(db_fn, db='pickle') for db_fn in db_fns]

gr =  gelman_rubin(models)
pandas.DataFrame(gr).to_csv(os.path.join(path, 'gelman_rubin_{}.csv'.format(model))) 


# Create a new model that has all traces concatenated
# of individual models.
combined_model = kabuki.utils.concat_models(models)

combined_model.print_stats()

model_fn = os.path.join(path, 'model_{}'.format(model))
combined_model.save(model_fn)

