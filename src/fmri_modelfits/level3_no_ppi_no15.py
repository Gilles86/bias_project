import nipype.pipeline.engine as pe

from nipype.interfaces import io as nio
from nipype.interfaces import ants
from nipype.interfaces import fsl
from nipype.interfaces import utility as util
from nipype.workflows.fmri.fsl.estimate import create_overlay_workflow
import numpy as np

from nipype.workflows.fmri.fsl.estimate import create_fixed_effects_flow

subject_ids = ['%02d' % i for i in np.arange(1, 20)]
subject_ids.pop(subject_ids.index('15'))

templates = {'level2_cope':'/home/gdholla1/projects/bias/data/derivatives/glm_fits_level2.no_ppi/level2_copes/_model_{model}_subject_id_{subject_id}/_fwhm_5.0/_flameo{contrast}/cope1.nii.gz',
             'level2_varcope':'/home/gdholla1/projects/bias/data/derivatives/glm_fits_level2.no_ppi/level2_varcopes/_model_{model}_subject_id_{subject_id}/_fwhm_5.0/_flameo{contrast}/varcope1.nii.gz',
             'level2_tdof':'/home/gdholla1/projects/bias/data/derivatives/glm_fits_level2.no_ppi/level2_tdof/_model_{model}_subject_id_{subject_id}/_fwhm_5.0/_flameo{contrast}/tdof_t1.nii.gz',             
             'epi2struct':'/home/gdholla1/projects/bias/data/derivatives/registration/epi2t1weighted/epi2structmat_ants/_subject_id_{subject_id}/transformComposite.h5',
             't1weighted':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz',
             'struct2mni':'/home/gdholla1/projects/bias/data/derivatives/registration/epi2t1weighted/struct2mnimat_ants/_subject_id_{subject_id}/transformComposite.h5',
             'mask':fsl.Info.standard_image('MNI152_T1_1mm_brain_mask.nii.gz')}

fixedfx_flow = create_fixed_effects_flow('level3_no_ppi')
fixedfx_flow.base_dir = '/home/gdholla1/projects/bias/workflow_folders/'

selector = pe.MapNode(nio.SelectFiles(templates), iterfield=['subject_id'], name='selector')

selector.iterables = [('model', ['model2'][:]), ('contrast', np.arange(4))]
selector.inputs.subject_id = subject_ids

cope_transformer = pe.MapNode(ants.ApplyTransforms(), iterfield=['input_image', 'transforms'], name='transformer')
cope_transformer.inputs.reference_image = fsl.Info.standard_image('MNI152_T1_1mm.nii.gz')
varcope_transformer = cope_transformer.clone('varcope_transformer')
tdof_transformer = cope_transformer.clone('tdof_transformer')

fixedfx_flow.connect(selector, 'level2_cope', cope_transformer, 'input_image')
fixedfx_flow.connect(selector, 'level2_varcope', varcope_transformer, 'input_image')
fixedfx_flow.connect(selector, 'level2_tdof', tdof_transformer, 'input_image')

merger = pe.MapNode(util.Merge(2), iterfield=['in1', 'in2'], name='transform_merger')

fixedfx_flow.connect(selector, 'struct2mni', merger, 'in1')
fixedfx_flow.connect(selector, 'epi2struct', merger, 'in2')

fixedfx_flow.connect(merger, 'out', cope_transformer, 'transforms')
fixedfx_flow.connect(merger, 'out', varcope_transformer, 'transforms')
fixedfx_flow.connect(merger, 'out', tdof_transformer, 'transforms')



def num_copes(files):
    return len(files)

def listify(x):
    return [x]

def pickone(input):
    return input[0]


fixedfx_flow.connect(cope_transformer, ('output_image', listify), fixedfx_flow.get_node('inputspec'), 'copes')
fixedfx_flow.connect(varcope_transformer, ('output_image', listify), fixedfx_flow.get_node('inputspec'), 'varcopes')


fixedfx_flow.connect(cope_transformer, ('output_image', num_copes), fixedfx_flow.get_node('l2model'), 'num_copes')


fixedfx_flow.connect(selector, ('mask', pickone), fixedfx_flow.get_node('flameo'), 'mask_file')
fixedfx_flow.inputs.flameo.run_mode = 'flame1'


fixedfx_flow.disconnect([(fixedfx_flow.get_node('inputspec'), fixedfx_flow.get_node('gendofvolume'), [('dof_files', 'dof_files')]),
                         (fixedfx_flow.get_node('copemerge'), fixedfx_flow.get_node('gendofvolume'), [('merged_file', 'cope_files')]),
                         (fixedfx_flow.get_node('gendofvolume'), fixedfx_flow.get_node('flameo'), [('dof_volume', 'dof_var_cope_file')])])
fixedfx_flow.remove_nodes([fixedfx_flow.get_node('gendofvolume')])

tdof_merge =  pe.Node(interface=fsl.Merge(dimension='t'), name="tdof_merge")
fixedfx_flow.connect(tdof_transformer, 'output_image', tdof_merge, 'in_files')
fixedfx_flow.connect(tdof_merge, 'merged_file', fixedfx_flow.get_node('flameo'), 'dof_var_cope_file')


ds = pe.Node(nio.DataSink(), name='datasink')
ds.inputs.base_directory = '/home/gdholla1/projects/bias/data/derivatives/glm_fits_level3_no_ppi/'


def volume_convert(input):
    return int(input[0])


smoothestimate = pe.MapNode(fsl.SmoothEstimate(), iterfield=['zstat_file'], name='smoothestimate')
fixedfx_flow.connect(selector, ('mask', pickone), smoothestimate, 'mask_file')
fixedfx_flow.connect(fixedfx_flow.get_node('outputspec'), 'zstats', smoothestimate, 'zstat_file')

get_volume = pe.Node(fsl.ImageStats(op_string = '-V'), name='get_volume')
fixedfx_flow.connect(selector, ('mask', pickone), get_volume, 'in_file')

grf_cluster = pe.MapNode(fsl.Cluster(), iterfield=['dlh', 'in_file'], name='grf_cluster')
grf_cluster.iterables = [("threshold", [2.3, 2.6, 3.1])] #, 2.3
grf_cluster.inputs.out_localmax_txt_file = True
grf_cluster.inputs.volume = 1827243
grf_cluster.inputs.pthreshold = 0.05
grf_cluster.inputs.out_index_file = True

fixedfx_flow.connect(smoothestimate, 'dlh', grf_cluster, 'dlh')
fixedfx_flow.connect(fixedfx_flow.get_node('outputspec'), 'zstats', grf_cluster, 'in_file')

fixedfx_flow.connect(get_volume, ('out_stat', volume_convert), grf_cluster, 'volume')
grf_cluster.inputs.out_threshold_file = True

fixedfx_flow.connect(fixedfx_flow.get_node('outputspec'), 'zstats', ds, 'level3_zstats')
fixedfx_flow.connect(grf_cluster, 'threshold_file', ds, 'grf_thresholded_zstats_file')
fixedfx_flow.connect(grf_cluster, 'localmax_txt_file', ds, 'grf_localmax_txt_file')
fixedfx_flow.connect(grf_cluster, 'index_file', ds, 'cluster_indices')
fixedfx_flow.connect(cope_transformer, ('output_image', listify), ds, 'transformed_copes')

fixedfx_flow.run(plugin='MultiProc', plugin_args={'n_procs':6})
