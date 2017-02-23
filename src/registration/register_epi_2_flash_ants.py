import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import os
import numpy as np

project_dir = '/home/gdholla1/projects/bias'

workflow = pe.Workflow(name='register_epi2FLASH_ants')
workflow.base_dir = os.path.join(project_dir, 'workflow_folders')

templates = {'mean_epi':os.path.join(project_dir, 'data', 'processed', 'feat_preprocess', 'mean', '_subject_id_{subject_id}', '_fwhm_0.0', 'sub-{subject_id}_task-randomdotmotion_run-01_bold_unwarped_st_dtype_mcf_mask_gms_mean.nii.gz'),
             'FLASH':os.path.join(project_dir, 'data', 'derivatives', 'sub-{subject_id}', 'mean_flash', 'sub-{subject_id}_FLASH_echo_11.22_merged_mean_brain.nii.gz')}

selector = pe.MapNode(nio.SelectFiles(templates), iterfield='subject_id', name='selector')
subject_ids = ['%02d' % i for i in np.arange(1, 20)]

selector.iterables = [('subject_id', subject_ids[:1])]

from nipype.interfaces.c3 import C3dAffineTool
import nipype.interfaces.ants as ants

reg = pe.Node(ants.Registration(), name='antsRegister')
reg.inputs.transforms = ['Rigid', 'Affine']
reg.inputs.transform_parameters = [(0.1,), (0.1,)]
reg.inputs.number_of_iterations = [[1000,500,250,100]]*2
reg.inputs.dimension = 3
reg.inputs.write_composite_transform = True
reg.inputs.collapse_output_transforms = True
reg.inputs.metric = ['MI']*2
reg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
reg.inputs.radius_or_number_of_bins = [32]*2
reg.inputs.sampling_strategy = ['Regular']*2
reg.inputs.sampling_percentage = [0.25]*2
reg.inputs.convergence_threshold = [1.e-8]*2
reg.inputs.convergence_window_size = [10]*2
reg.inputs.smoothing_sigmas = [[3,2,1,0]]*2
reg.inputs.sigma_units = ['mm']*2
reg.inputs.shrink_factors = [[8,4,2,1]]*2
reg.inputs.use_estimate_learning_rate_once = [True, True, True]
reg.inputs.use_histogram_matching = [False]*2 # This is the default
reg.inputs.initial_moving_transform_com = True
reg.inputs.output_warped_image = True
reg.inputs.winsorize_lower_quantile = 0.01
reg.inputs.winsorize_upper_quantile = 0.99

workflow.connect(selector, 'mean_epi', reg, 'moving_image')
workflow.connect(selector, 'FLASH', reg, 'fixed_image')

ds = pe.Node(nio.DataSink(), name='datasink')
ds.inputs.base_directory = os.path.join(project_dir, 'data', 'derivatives', 'registration', 'epi2flash')

workflow.connect(reg, 'composite_transform', ds, 'epi2FLASH_transform')
workflow.connect(reg, 'inverse_composite_transform', ds, 'FLASH2epi_transform')
workflow.connect(reg, 'warped_image', ds, 'mean_epi_in_FLASH_space')

#workflow.run(plugin='MultiProc', plugin_args={'n_procs':8})
workflow.run()

