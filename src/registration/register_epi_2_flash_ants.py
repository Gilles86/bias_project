import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio


subject_ids = ['KCAT', 'WSFT', 'WW2T', 'TS6T', 'FMFT', 'HCBT', 'PF5T', 'LV2T', 'UM2T', 'MRCT', 'RSIT', 'KP6T', 'NM3T', 'BI3T', 'SC1T', 'SPGT', 'ZK4T', 'GAIT', 'DA9T', 'VL1T']

workflow = pe.Workflow(base_dir='/home/gdholla1/workflow_folders/', name='register_epi2FLASH_ants')

templates = {'mean_epi':'/home/gdholla1/data/bias_task/preprocessed/feat_preprocess/mean/_subject_id_{subject_id}/_fwhm_0.0/run1_unwarped_st_dtype_mcf_mask_gms_mean.nii.gz',
             'FLASH':'/home/gdholla1/data/bias_task/flash_data_std_mean/_subject_id_{subject_id}/e11.22_merged_mean.nii.gz'}

selector = pe.Node(nio.SelectFiles(templates), name='selector')

selector.iterables = [('subject_id', subject_ids)]

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

ds = pe.Node(nio.DataSink(base_directory='/home/gdholla1/data/bias_task/register_epi2flash_ants'), name='datasink')

workflow.connect(reg, 'composite_transform', ds, 'epi2FLASH_transform')
workflow.connect(reg, 'inverse_composite_transform', ds, 'FLASH2epi_transform')
workflow.connect(reg, 'warped_image', ds, 'mean_epi_in_FLASH_space')

#workflow.run(plugin='MultiProc', plugin_args={'n_procs':8})
workflow.run()

