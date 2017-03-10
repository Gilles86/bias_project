import argparse
import os
from kabuki.analyze import gelman_rubin
import pandas
import hddm


path = '/home/gholland/projects/bias/data/hddm_fits'

# SELECT MASK
models = ['final_super_model_startpoint', 'final_super_model_drift']
hemispheres = ['L', 'R']

if 'PBS_ARRAYID' in os.environ:
    hemisphere_idx = int(os.environ['PBS_ARRAYID']) / len(models)
    model_idx = int(os.environ['PBS_ARRAYID']) % len(models)
else:
    hemisphere_idx = 0
    model_idx = 0

hemisphere = hemispheres[hemisphere_idx]
model = models[model_idx]


def get_model(model, hemisphere):
    df = pandas.read_pickle('/home/gdholla1/projects/bias/data/behavior/behavior_and_single_trial_estimates.pandas')

    # Prepare data
    data = df
    data['response'] = data['correct']
    data['rt'] = data['rt'] / 1000.
    data = data[(data.rt > .2) & (data.rt < 1.5)]
    data = data[~data.rt.isnull()]
    data['cue_coding'] = data.cue_validity.map({'valid':1, 'neutral':0, 'invalid':-1})

    def z_link_func(x, data=data):
        return 1 / (1 + np.exp(-(x.values.ravel())))

    if model == 'drift_super':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + cue_coding + C(difficulty)*{}_stim + C(difficulty)*{}_stim + C(difficulty)*{}_stim'.format('STh_{}_A'.format(hemisphere),
                                                                                                                 'STh_{}_B'.format(hemisphere),
                                                                                                                 'STh_{}_C'.format(hemisphere),), 
                 'link_func': lambda x: x}
        
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])

    if model == 'drift_errors_super':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim + {}_stim:correct + '
                          'C(difficulty)*{}_stim + {}_stim:correct + ' 
                          'C(difficulty)*{}_stim + {}_stim:correct').format('STh_{}_A'.format(hemisphere), 'STh_{}_A'.format(hemisphere),
                                                                            'STh_{}_B'.format(hemisphere), 'STh_{}_B'.format(hemisphere),
                                                                            'STh_{}_C'.format(hemisphere), 'STh_{}_C'.format(hemisphere)), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'start_point_super':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + cue_coding + C(difficulty)', 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])

    if model == 'all_super':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        #v_reg = {'model': ('v ~ 1 + '
                          #'C(difficulty)*{}_stim + {}_stim:correct + '
                          #'C(difficulty)*{}_stim + {}_stim:correct + ' 
                          #'C(difficulty)*{}_stim + {}_stim:correct').format('STh_{}_A'.format(hemisphere), 'STh_{}_A'.format(hemisphere),
                                                                            #'STh_{}_B'.format(hemisphere), 'STh_{}_B'.format(hemisphere),
                                                                            #'STh_{}_C'.format(hemisphere), 'STh_{}_C'.format(hemisphere)), 'link_func': lambda x: x}        
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  + '
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        


    if model == 'all_super_drift_bias_cue':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  +'
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim + '
                          'cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), 'STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        

        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])

    if model == 'all_super_drift_bias_stim':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  +'
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim + '
                          'cue_coding + cue_coding:{}_stim + cue_coding:{}_stim + cue_coding:{}_stim').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), 'STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        


        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])

    if model == 'all_super_drift_bias_stim_sv_sz_p_outlier':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  +'
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim + '
                          'cue_coding + cue_coding:{}_stim + cue_coding:{}_stim + cue_coding:{}_stim').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), 'STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        


        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz'], p_outlier=0.05)

    if model == 'final_super_model_startpoint':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty) + cue_coding', 'link_func': lambda x: x}        


        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz', 'st'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz', 'st'])
    
    if model == 'final_super_model_drift':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{}_stim + C(difficulty)*{}_stim +  C(difficulty)*{}_stim + cue_coding'.format('STh_{}_A'.format(hemisphere),
                                                                                                                               'STh_{}_B'.format(hemisphere),
                                                                                                                               'STh_{}_C'.format(hemisphere),), 'link_func': lambda x: x}        


        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz', 'st'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz', 'st'])

    return hddm_model

def fit_model(id, model=model, hemisphere=hemisphere, path=path):
    import numpy as np
    import hddm
    import pandas
    import os
    
    df = pandas.read_pickle('/home/gholland/projects/bias/data/behavior/behavior_and_single_trial_estimates.pandas')

    data = df
    data['response'] = data['correct']
    data['rt'] = data['rt'] / 1000.
    data = data[(data.rt > .2) & (data.rt < 1.5)]
    data = data[~data.rt.isnull()]
    data['cue_coding'] = data.cue_validity.map({'valid':1, 'neutral':0, 'invalid':-1})

    def z_link_func(x, data=data):
        return 1 / (1 + np.exp(-(x.values.ravel())))

    if model == 'drift_super':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{}_stim + C(difficulty)*{}_stim + C(difficulty)*{}_stim'.format('STh_{}_A'.format(hemisphere),
                                                                                                                 'STh_{}_B'.format(hemisphere),
                                                                                                                 'STh_{}_C'.format(hemisphere),), 
                 'link_func': lambda x: x}
        
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'drift_errors_super':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim + {}_stim:correct + '
                          'C(difficulty)*{}_stim + {}_stim:correct + ' 
                          'C(difficulty)*{}_stim + {}_stim:correct').format('STh_{}_A'.format(hemisphere), 'STh_{}_A'.format(hemisphere),
                                                                            'STh_{}_B'.format(hemisphere), 'STh_{}_B'.format(hemisphere),
                                                                            'STh_{}_C'.format(hemisphere), 'STh_{}_C'.format(hemisphere)), 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'start_point_super':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)', 'link_func': lambda x: x}
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=True, group_only_nodes=['sv'])

    if model == 'all_super':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        #v_reg = {'model': ('v ~ 1 + '
                          #'C(difficulty)*{}_stim + {}_stim:correct + '
                          #'C(difficulty)*{}_stim + {}_stim:correct + ' 
                          #'C(difficulty)*{}_stim + {}_stim:correct').format('STh_{}_A'.format(hemisphere), 'STh_{}_A'.format(hemisphere),
                                                                            #'STh_{}_B'.format(hemisphere), 'STh_{}_B'.format(hemisphere),
                                                                            #'STh_{}_C'.format(hemisphere), 'STh_{}_C'.format(hemisphere)), 'link_func': lambda x: x}        
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  +'
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])

    if model == 'all_super_drift_bias_cue':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  +'
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim + '
                          'cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), 'STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        

        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])

    if model == 'all_super_drift_bias_stim':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  +'
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim + '
                          'cue_coding + cue_coding:{}_stim + cue_coding:{}_stim + cue_coding:{}_stim').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), 'STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        

        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv'), bias=True, group_only_regressors=False, group_only_nodes=['sv'])

    if model == 'all_super_drift_bias_stim_sv_sz_p_outlier':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': ('v ~ 1 + '
                          'C(difficulty)*{}_stim  +'
                          'C(difficulty)*{}_stim + ' 
                          'C(difficulty)*{}_stim + '
                          'cue_coding + cue_coding:{}_stim + cue_coding:{}_stim + cue_coding:{}_stim').format('STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), 'STh_{}_A'.format(hemisphere), 'STh_{}_B'.format(hemisphere), 'STh_{}_C'.format(hemisphere), ), 'link_func': lambda x: x}        
        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz'], p_outlier=0.05)

    if model == 'final_super_model_startpoint':
        z_reg = {'model': 'z ~ 0 + cue_coding + cue_coding:{}_cue + cue_coding:{}_cue + cue_coding:{}_cue'.format('STh_{}_A'.format(hemisphere),
                                                                                                                  'STh_{}_B'.format(hemisphere),
                                                                                                                  'STh_{}_C'.format(hemisphere),), 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty) + cue_coding', 'link_func': lambda x: x}        


        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz', 'st'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz', 'st'])
    
    if model == 'final_super_model_drift':
        z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
        v_reg = {'model': 'v ~ 1 + C(difficulty)*{}_stim + C(difficulty)*{}_stim +  C(difficulty)*{}_stim + cue_coding'.format('STh_{}_A'.format(hemisphere),
                                                                                                                               'STh_{}_B'.format(hemisphere),
                                                                                                                               'STh_{}_C'.format(hemisphere),), 'link_func': lambda x: x}        


        reg_descr = [z_reg, v_reg]
        hddm_model = hddm.HDDMRegressor(data, reg_descr, include=('sv', 'sz', 'st'), bias=True, group_only_regressors=False, group_only_nodes=['sv', 'sz', 'st'])


    hddm_model.find_starting_values()
    db_fn = os.path.join(path, 'traces_{}_{}_{}.pkl'.format(model, hemisphere, id))

    hddm_model.sample(5000, 1000, dbname=db_fn, db='pickle')
    return db_fn


from IPython.parallel import Client

json_path = os.path.join(os.environ['TMPDIR'], 'hddm'.format(model, id), 'security', 'ipcontroller-client.json')
v = Client(json_path)[:]

jobs = v.map(fit_model, range(15)) 
db_fns = jobs.get()

hddm_model = get_model(model, hemisphere)

models = [hddm_model.load_db(db_fn, db='pickle') for db_fn in db_fns]

gr =  gelman_rubin(models)
pandas.DataFrame(gr).to_csv(os.path.join(path, 'gelman_rubin_{}_{}.csv'.format(model, hemisphere))) 


# Create a new model that has all traces concatenated
# of individual models.
combined_model = kabuki.utils.concat_models(models)

combined_model.print_stats()

model_fn = os.path.join(path, 'model_{}_{}'.format(model, hemisphere))
combined_model.save(model_fn)

