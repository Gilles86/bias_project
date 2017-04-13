import nipype.pipeline.engine as pe

from nipype.interfaces import io as nio
from nipype.interfaces import ants
from nipype.interfaces import fsl
from nipype.interfaces import utility as util
from nipype.workflows.fmri.fsl.estimate import create_overlay_workflow
import numpy as np

from nipype.workflows.fmri.fsl.estimate import create_fixed_effects_flow

subject_ids = ['%02d' % i for i in np.arange(1, 20)]

identity = pe.Node(util.IdentityInterface(fields=['subject_id']), name='identity')
identity.inputs.subject_id = subject_ids

templates = {'level2_cope':'/home/gdholla1/projects/bias/data/derivatives/glm_fits_level2.no_ppi/level2_copes/_model_{model}_subject_id_{subject_id}/_fwhm_{fwhm}/_shift_{shift}/_flameo{contrast}/cope1.nii.gz',
             'level2_varcope':'/home/gdholla1/projects/bias/data/derivatives/glm_fits_level2.no_ppi/level2_varcopes/_model_{model}_subject_id_{subject_id}/_fwhm_{fwhm}/_shift_{shift}/_flameo{contrast}/varcope1.nii.gz',
             'level2_tdof':'/home/gdholla1/projects/bias/data/derivatives/glm_fits_level2.no_ppi/level2_tdof/_model_{model}_subject_id_{subject_id}/_fwhm_{fwhm}/_shift_{shift}/_flameo{contrast}/tdof_t1.nii.gz',
             'epi2struct':'/home/gdholla1/projects/bias/data/derivatives/registration/epi2t1weighted/epi2structmat_ants/_subject_id_{subject_id}/transformComposite.h5',
             't1weighted':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz',
             'struct2mni':'/home/gdholla1/projects/bias/data/derivatives/registration/epi2t1weighted/struct2mnimat_ants/_subject_id_{subject_id}/transformComposite.h5',
             'mask':fsl.Info.standard_image('MNI152_T1_1mm_brain_mask.nii.gz')}

workflow = pe.Workflow(name='level3_model_parameters')
workflow.base_dir = '/home/gdholla1/projects/bias/workflow_folders/'

selector = pe.MapNode(nio.SelectFiles(templates), iterfield=['subject_id'], name='selector')
selector.iterables = [('model', ['model1a'][:]), ('contrast', [2,3]), ('shift', [-1.5]), ('fwhm', [8.0])]
# selector.inputs.subject_id = subject_ids
workflow.connect(identity, 'subject_id', selector, 'subject_id')

cope_transformer = pe.MapNode(ants.ApplyTransforms(), iterfield=['input_image', 'transforms'], name='transformer')
cope_transformer.inputs.reference_image = fsl.Info.standard_image('MNI152_T1_1mm.nii.gz')
varcope_transformer = cope_transformer.clone('varcope_transformer')
tdof_transformer = cope_transformer.clone('tdof_transformer')

workflow.connect(selector, 'level2_cope', cope_transformer, 'input_image')
workflow.connect(selector, 'level2_varcope', varcope_transformer, 'input_image')
workflow.connect(selector, 'level2_tdof', tdof_transformer, 'input_image')

merger = pe.MapNode(util.Merge(2), iterfield=['in1', 'in2'], name='transform_merger')

workflow.connect(selector, 'struct2mni', merger, 'in1')
workflow.connect(selector, 'epi2struct', merger, 'in2')

workflow.connect(merger, 'out', cope_transformer, 'transforms')
workflow.connect(merger, 'out', varcope_transformer, 'transforms')
workflow.connect(merger, 'out', tdof_transformer, 'transforms')



def get_regressors(subject_ids, variable='z_diff'):
    import pandas
    import numpy as np
    
    subject_ids = np.array(subject_ids, dtype='int')

    df = pandas.read_pickle('/home/gdholla1/projects/bias/data/derivatives/behavior/hddm/bias_shifts.pkl').set_index('subj_idx')

    if variable == 'z_diff':
        regressors = df.ix[subject_ids].z_cue_coding.values
        regressors = list(regressors - regressors.mean())
            
    if variable == 'v_diff':
        regressors = df.ix[subject_ids].v_difficulty.values
        regressors = list(regressors - regressors.mean())

            
    regressors = {'group_mean':[1]*len(subject_ids), variable:regressors}
    contrasts = [['group mean', 'T',['group_mean'],[1]],[variable, 'T',[variable],[1]]]
    
    
    return regressors, contrasts

get_regressors_node = pe.Node(util.Function(function=get_regressors,
                                            input_names=['subject_ids', 'variable'],
                                            output_names=['regressors', 'contrasts']),
                              name='get_regressors_node')


get_regressors_node.iterables = [('variable', ['v_diff'])]

workflow.connect(identity, 'subject_id', get_regressors_node, 'subject_ids')

copemerge = pe.Node(interface=fsl.Merge(dimension='t'), name="copemerge")
varcopemerge = pe.Node(interface=fsl.Merge(dimension='t'), name="varcopemerge")
tdof_merge =  pe.Node(interface=fsl.Merge(dimension='t'), name="tdof_merge")

flameo = pe.Node(fsl.FLAMEO(run_mode='flame1'), name='flameo')
flameo.inputs.mask_file = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'

workflow.connect(cope_transformer, 'output_image', copemerge, 'in_files')
workflow.connect(copemerge, 'merged_file', flameo, 'cope_file')

workflow.connect(varcope_transformer, 'output_image', varcopemerge, 'in_files')
workflow.connect(varcopemerge, 'merged_file', flameo, 'var_cope_file')

workflow.connect(tdof_transformer, 'output_image', tdof_merge, 'in_files')
workflow.connect(tdof_merge, 'merged_file', flameo, 'dof_var_cope_file')

multiple_regress = pe.Node(fsl.MultipleRegressDesign(), name='multiple_regress')
workflow.connect(get_regressors_node, 'contrasts', multiple_regress, 'contrasts')
workflow.connect(get_regressors_node, 'regressors', multiple_regress, 'regressors')

workflow.connect(multiple_regress, 'design_con', flameo, 't_con_file')
workflow.connect(multiple_regress, 'design_grp', flameo, 'cov_split_file')
workflow.connect(multiple_regress, 'design_mat', flameo, 'design_file')


ds = pe.Node(nio.DataSink(base_directory='/home/gdholla1/projects/bias/data/derivatives/level3_model_parameters'), name='datasink')
workflow.connect(flameo, 'zstats', ds, 'zstats')

smooth_est = pe.MapNode(fsl.SmoothEstimate(), iterfield=['zstat_file'], name='smooth_est')
smooth_est.inputs.mask_file = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'
workflow.connect(flameo, 'zstats', smooth_est, 'zstat_file')

grf_cluster = pe.MapNode(fsl.Cluster(), iterfield=['dlh', 'in_file'], name='grf_cluster')
grf_cluster.iterables = [("threshold", [2.3, 2.6, 3.1])] #, 2.3
grf_cluster.inputs.out_localmax_txt_file = True
grf_cluster.inputs.volume = 1827243
grf_cluster.inputs.pthreshold = 0.05
grf_cluster.inputs.out_index_file = True

workflow.connect(smooth_est, 'dlh', grf_cluster, 'dlh')
workflow.connect(flameo, 'zstats', grf_cluster, 'in_file')
grf_cluster.inputs.out_threshold_file = True

workflow.connect(grf_cluster, 'threshold_file', ds, 'grf_thresholded_zstats_file')
workflow.connect(grf_cluster, 'localmax_txt_file', ds, 'grf_localmax_txt_file')
workflow.connect(grf_cluster, 'index_file', ds, 'cluster_indices')
workflow.connect(cope_transformer, 'output_image', ds, 'transformed_copes')

workflow.write_graph()
workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 9})

