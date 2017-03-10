import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
from nipype.workflows.fmri.fsl.estimate import create_modelfit_workflow, create_fixed_effects_flow
import os
import numpy as np
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl

def get_session_info(subject_id, run, shift=0, design='model0', nucleus_signal=None, regressor_file=None):
    import pandas
    import numpy as np
    from nipype.interfaces.base import Bunch
    
    subject_id = int(subject_id)
    
    df =pandas.read_pickle('/home/gdholla1/projects/bias/data/behavior/behavior.pandas')

    df = df[(df.subj_idx == subject_id) & (df.block == run)]

    df['onset_cue'] += shift
    df['onset_stim'] += shift

    if design == 'model0':
        onsets_cue = df.onset_cue.tolist()
        onsets_stim = df.onset_stim.tolist()

        conditions=['cue',
                    'left_response',
                    'right_response']
        
        onsets=[onsets_cue,
                df[df.response == 1].onset_stim.tolist(),
                df[df.response == 2].onset_stim.tolist()]

        info = Bunch(conditions=conditions,
                  onsets=onsets,
                  durations=[[1]] * len(conditions))
        
    elif design == 'model1a':
        conditions=['payoff_cue',
                    'neutral_cue',
                    'difficult_rdm',
                    'easy_rdm']
        
        onsets=[df[df.cue != 'neutral'].onset_cue.tolist(),
                df[df.cue == 'neutral'].onset_cue.tolist(),
                df[df.difficulty == 'easy'].onset_stim.tolist(),
                df[df.difficulty == 'hard'].onset_stim.tolist()]

        info = Bunch(conditions=conditions,
                  onsets=onsets,
                  durations=[[1]] * len(conditions))
        
    elif design == 'model1c':
        conditions=['payoff_cue',
                    'neutral_cue',
                    'hard_rdm (correct)',
                    'easy_rdm (correct)',                    
                    'hard_rdm (error)',                    
                    'easy_rdm (error)']
        
        onsets=[df[df.cue != 'neutral'].onset_cue.tolist(),
                df[df.cue == 'neutral'].onset_cue.tolist(),
                df[(df.difficulty == 'easy') & (df.correct == 1)].onset_stim.tolist(),
                df[(df.difficulty == 'hard') & (df.correct == 1)].onset_stim.tolist(),
                df[(df.difficulty == 'easy') & (df.correct == 0)].onset_stim.tolist(),
                df[(df.difficulty == 'hard') & (df.correct == 0)].onset_stim.tolist()]        

        info = Bunch(conditions=conditions,
                  onsets=onsets,
                  durations=[[1]] * len(conditions)) 
        
    if nucleus_signal and regressor_file:
        
        nucleus_signal = np.loadtxt(nucleus_signal)
        nucleus_signal -= nucleus_signal.mean()
        
        regressors = np.loadtxt(regressor_file, skiprows=5)[:, :len(conditions)]
        
        new_info = {'regressor_names':['nucleus_signal'] + ['%s * nucleus_signal' % s for s in conditions],
                     'regressors':[nucleus_signal] + [nucleus_signal * regressor for regressor in regressors.T]}
        
        new_info['regressors'] = [e - e.mean() for e in new_info['regressors']]
        
        info.update(new_info)
        
    return info

project_folder = '/home/gdholla1/projects/bias/'

workflow = pe.Workflow(name='bias_response_glm')
workflow.base_dir = os.path.join(project_folder, 'workflow_folders')

identity = pe.Node(util.IdentityInterface(fields=['subject_id', 'mask']), name='identity')

templates = {'highpassed_files':os.path.join(project_folder, 'data', 'processed', 'feat_preprocess', 'highpassed_files', '_subject_id_{subject_id}', '_fwhm_{fwhm}', '_addmean*', 'sub-{subject_id}*.nii.gz'),
             'realignment_parameters':os.path.join(project_folder, 'data', 'processed', 'feat_preprocess', 'motion_parameters', '_subject_id_{subject_id}', '_fwhm_0.0', '_realign*', 'sub-{subject_id}*.nii.gz.par'),
             'mask':os.path.join(project_folder, 'data', 'processed', 'feat_preprocess', 'mask', '_subject_id_{subject_id}', '_fwhm_{fwhm}', '_dilatemask0', 'sub-{subject_id}*.nii.gz'),}

subject_ids = ['%02d' % i for i in np.arange(1, 20)]

identity.iterables = [('subject_id', subject_ids),]

selector = pe.Node(nio.SelectFiles(templates), name='selector')
selector.iterables = [('fwhm', [5.0])]

workflow.connect(identity, 'subject_id', selector, 'subject_id')
workflow.connect(identity, 'mask', selector, 'mask')


contrasts = [('left response > right response', 'T', ['left_response', 'right_response'], [1.0, -1.0]),
             ('right response  > left response', 'T', ['right_response', 'left_response'], [-1.0, 1.0]),]


specifymodel1 = pe.Node(SpecifyModel(), name='specifymodel1')

specifymodel1.inputs.input_units = 'secs'
specifymodel1.inputs.time_repetition = 3.0
specifymodel1.inputs.high_pass_filter_cutoff = 128. / (2. * 3.)

workflow.connect(selector, 'realignment_parameters', specifymodel1, 'realignment_parameters')
workflow.connect(selector, 'highpassed_files', specifymodel1, 'functional_runs')


session_info_getter = pe.MapNode(util.Function(function=get_session_info,
                                     input_names=['subject_id', 'run', 'shift'],
                                     output_names=['session_info']),
                       iterfield=['run'],
                       name='session_info_getter')

session_info_getter.inputs.run = [1,2,3]
session_info_getter.inputs.shift = -3.

workflow.connect(identity, 'subject_id', session_info_getter, 'subject_id')
workflow.connect(session_info_getter, 'session_info', specifymodel1, 'subject_info')

level1design1 = pe.Node(interface=fsl.Level1Design(), name="level1design1")
workflow.connect(specifymodel1, 'session_info', level1design1, 'session_info')
level1design1.inputs.interscan_interval = 3.0
level1design1.inputs.bases = {'dgamma': {'derivs': False}}
level1design1.inputs.contrasts = contrasts
level1design1.inputs.model_serial_correlations = True

modelgen1 = pe.MapNode(interface=fsl.FEATModel(), iterfield=['ev_files', 'fsf_file'], name='modelgen1')

workflow.connect(level1design1, 'ev_files', modelgen1, 'ev_files')
workflow.connect(level1design1, 'fsf_files', modelgen1, 'fsf_file')


iterfield = ['design_file', 'in_file', 'tcon_file']
modelestimate = pe.MapNode(interface=fsl.FILMGLS(smooth_autocorr=True,
                                                         mask_size=5),
                       name='modelestimate',
                       iterfield=iterfield)

modelestimate.inputs.threshold = 1000
workflow.connect(selector, 'highpassed_files', modelestimate, 'in_file')
workflow.connect(modelgen1, 'design_file', modelestimate, 'design_file')
workflow.connect(modelgen1, 'con_file', modelestimate, 'tcon_file')


fixedfx = create_fixed_effects_flow()

def pickone(input):
    return input[0]


workflow.connect(selector, 'mask', fixedfx, 'flameo.mask_file')

def num_copes(files):
    return len(files)

def transpose_copes(copes):    
    import numpy as np
    return np.array(copes).T.tolist()


workflow.connect(modelestimate, ('copes', transpose_copes), fixedfx, 'inputspec.copes')
workflow.connect(modelestimate, ('varcopes', transpose_copes), fixedfx, 'inputspec.varcopes')
workflow.connect(modelestimate, 'dof_file', fixedfx, 'inputspec.dof_files')
workflow.connect(modelestimate, ('copes', num_copes), fixedfx, 'l2model.num_copes')


smoothestimate = pe.MapNode(fsl.SmoothEstimate(), iterfield=['zstat_file'], name='smoothestimate')
workflow.connect(selector, 'mask', smoothestimate, 'mask_file')
workflow.connect(fixedfx, 'outputspec.zstats', smoothestimate, 'zstat_file')


get_volume = pe.Node(fsl.ImageStats(op_string = '-V'), name='get_volume')
workflow.connect(selector, 'mask', get_volume, 'in_file')

grf_cluster = pe.MapNode(fsl.Cluster(), iterfield=['dlh', 'in_file'], name='grf_cluster')
grf_cluster.iterables = [("threshold", [3.1])] #, 2.3
workflow.connect(smoothestimate, 'dlh', grf_cluster, 'dlh')
workflow.connect(fixedfx, 'outputspec.zstats', grf_cluster, 'in_file')

def convert_volume(input):
    return int(input[1])

workflow.connect(get_volume, ('out_stat', convert_volume), grf_cluster, 'volume')

grf_cluster.inputs.out_threshold_file = True

ds = pe.Node(nio.DataSink(), name='datasink')
ds.inputs.base_directory = os.path.join(project_folder, 'data', 'derivatives', 'glm_fits_level2.responses')

workflow.connect(grf_cluster, 'threshold_file', ds, 'grf_thresholded_zstats_file')
workflow.connect(grf_cluster, 'localmax_txt_file', ds, 'grf_localmax_txt_file')

workflow.connect(fixedfx, 'outputspec.zstats', ds, 'zstats')
workflow.connect(fixedfx, 'outputspec.copes', ds, 'level2_copes')
workflow.connect(fixedfx, 'outputspec.varcopes', ds, 'level2_varcopes')
workflow.connect(fixedfx, 'flameo.tdof', ds, 'level2_tdof')
workflow.run(plugin='MultiProc', plugin_args={'n_procs':8})
