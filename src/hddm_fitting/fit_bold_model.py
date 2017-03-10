import argparse
import os
from kabuki.analyze import gelman_rubin
import pandas
import hddm
import numpy as np

path = '/home/gholland/projects/bias/data/hddm_fits'

# SELECT MASK
masks = ['STh_L', 'STh_R', 'STh_L_A', 'STh_L_B', 'STh_L_C', 'STh_R_A', 'STh_R_B', 'STh_R_C']
models = ['drift_bold', 'drift_errors_bold', 'start_point_bold', 'all_bold', 'start_point_directional_bold', 'drift_simple_bold']

if 'PBS_ARRAYID' in os.environ:
    mask_idx = int(os.environ['PBS_ARRAYID']) / len(models)
    model_idx = int(os.environ['PBS_ARRAYID']) % len(models)
else:
    mask_idx = 0
    model_idx = 0

mask = masks[mask_idx]
model = models[model_idx]

def get_model(model, mask):
    df = pandas.read_pickle('/home/gholland/projects/bias/data/behavior/behavior_and_single_trial_estimates.pandas')

    mask_stim = '{}_stim'.format(mask)
    mask_cue = '{}_cue'.format(mask)

    # Prepare data
    data = df[['cue_validity', 'rt', 'correct', 'difficulty', 'subj_idx', mask_stim, mask_cue]]
    data['response'] = data['correct']
    data['rt'] = data['rt'] / 1000.
    data = data[(data.rt > .2) & (data.rt < 1.5)]
    data = data[~data.rt.isnull()]
    data['cue_coding'] = data.cue_validity.map({'valid':1, 'neutral':0, 'invalid':-1})

    def z_link_func(x, data=data):
        return 1 / (1 + np.exp(-(x.values.ravel())))

    if model == 'drift_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{}'.format(mask_stim), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'drift_errors_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{} + {}:correct'.format(mask_stim, mask_stim), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'start_point_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}'.format(mask_cue), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)', 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'all_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}'.format(mask_cue), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{} + {}:correct'.format(mask_stim, mask_stim), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'start_point_directional_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{} + cue:cue_coding + cue:{}'.format(mask_cue, mask_cue), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)', 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'drift_simple_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty) + {}'.format(mask_stim), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    return hddm_model

def fit_model(id, model=model, mask=mask, path=path):
    import numpy as np
    import hddm
    import pandas
    import os
    
    df = pandas.read_pickle('/home/gholland/projects/bias/data/behavior/behavior_and_single_trial_estimates.pandas')

    mask_stim = '{}_stim'.format(mask)
    mask_cue = '{}_cue'.format(mask)

    # Prepare data
    data = df[['cue_validity', 'rt', 'correct', 'difficulty', 'subj_idx', mask_stim, mask_cue]]
    data['response'] = data['correct']
    data['rt'] = data['rt'] / 1000.
    data = data[(data.rt > .2) & (data.rt < 1.5)]
    data = data[~data.rt.isnull()]
    data['cue_coding'] = data.cue_validity.map({'valid':1, 'neutral':0, 'invalid':-1})

    def z_link_func(x, data=data):
        return 1 / (1 + np.exp(-(x.values.ravel())))

    if model == 'drift_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{}'.format(mask_stim), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'drift_errors_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{} + {}:correct'.format(mask_stim, mask_stim), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'start_point_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}'.format(mask_cue), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)', 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'all_bold':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}'.format(mask_cue), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{} + {}:correct'.format(mask_stim, mask_stim), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    hddm_model.find_starting_values()
    db_fn = os.path.join(path, 'traces_{}_{}_{}.pkl'.format(model, mask, id))

    hddm_model.sample(10000, 1000, dbname=db_fn, db='pickle')
    return db_fn


from IPython.parallel import Client

json_path = os.path.join(os.environ['TMPDIR'], 'hddm'.format(model, id), 'security', 'ipcontroller-client.json')
v = Client(json_path)[:]

jobs = v.map(fit_model, range(15)) 
db_fns = jobs.get()

hddm_model = get_model(model, mask)

models = [hddm_model.load_db(db_fn, db='pickle') for db_fn in db_fns]

gr =  gelman_rubin(models)
pandas.DataFrame(gr).to_csv(os.path.join(path, 'gelman_rubin_{}_{}.csv'.format(model, mask))) 


# Create a new model that has all traces concatenated
# of individual models.
combined_model = kabuki.utils.concat_models(models)

combined_model.print_stats()

model_fn = os.path.join(path, 'model_{}_{}'.format(model, mask))
combined_model.save(model_fn)

