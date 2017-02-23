import pandas
import hddm
import numpy as np

data = pandas.read_pickle('../../data/behavior/behavior.pandas')
data = data[['cue_validity', 'rt', 'correct', 'difficulty', 'subj_idx']]
data['response'] = data['correct']
# data['cued'] = data.cue_validity.map({'invalid':True, 'valid':True, 'neutral':False})
data['rt'] = data['rt'] / 1000.

data = data[(data.rt > .2) & (data.rt < 1.5)]

data = data[~data.rt.isnull()]
data['cue_coding'] = data.cue_validity.map({'valid':1, 'neutral':0, 'invalid':-1})

def z_link_func(x, data=data):
    return 1 / (1 + np.exp(-(x.values.ravel())))

z_reg = {'model': 'z ~ 0 + cue_coding', 'link_func': z_link_func}
v_reg = {'model': 'v ~ 1 + C(difficulty)', 'link_func': lambda x: x}

reg_descr = [z_reg, v_reg]
model = hddm.HDDMRegressor(data, reg_descr, include='all', group_only_regressors=False, group_only_nodes=['sv', 'st'])
model.find_starting_values()


model.sample(10000, 5000, dbname='../../data/behavior/hddm_fits/traces_start_point.pkl', db='pickle')
model.print_stats()
model.save('../../data/behavior/hddm_fits/model_start_point.pickle')
