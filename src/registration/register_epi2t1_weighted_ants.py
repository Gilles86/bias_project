import nipype.pipeline.engine as pe
from nipype.interfaces import ants
from nipype.interfaces import fsl
import nipype.interfaces.io as nio
import numpy as np
import os

project_folder = '/home/gdholla1/projects/bias'

workflow = pe.Workflow(name='register_epi_to_struct_ants')
workflow.base_dir = os.path.join(project_folder, 'workflow_folders')

templates = {'mean_epi':os.path.join(project_folder, 'data', 'processed', 'feat_preprocess', 'mean', '_subject_id_{subject_id}', '_fwhm_0.0', 'sub-{subject_id}_task-randomdotmotion_run-01_bold_unwarped_st_dtype_mcf_mask_gms_mean.nii.gz'),
             't1_weighted':os.path.join(project_folder, 'data', 'raw', 'sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz')}

selector = pe.Node(nio.SelectFiles(templates), name='selector')
subject_ids = ['%02d' % i for i in np.arange(1, 20)]
selector.iterables = [('subject_id', subject_ids)]

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
workflow.connect(selector, 't1_weighted', reg, 'fixed_image')

ds = pe.Node(nio.DataSink(), name='datasink')
#ds.inputs.base_directory = '../../data/derivatives/registration/epi2t1weighted'
ds.inputs.base_directory = os.path.join(project_folder, 'data', 'derivatives', 'registration', 'epi2t1weighted')

mni_reg = pe.Node(ants.Registration(args='--float',
                            collapse_output_transforms=True,
                            initial_moving_transform_com=True,
                            num_threads=1,
                            output_inverse_warped_image=True,
                            output_warped_image=True,
                            sigma_units=['vox']*3,
                            transforms=['Rigid', 'Affine', 'SyN'],
                            terminal_output='file',
                            winsorize_lower_quantile=0.005,
                            winsorize_upper_quantile=0.995,
                            convergence_threshold=[1e-06],
                            convergence_window_size=[10],
                            metric=['MI', 'MI', 'CC'],
                            metric_weight=[1.0]*3,
                            number_of_iterations=[[1000, 500, 250, 100],
                                                  [1000, 500, 250, 100],
                                                  [100, 70, 50, 20]],
                            radius_or_number_of_bins=[32, 32, 4],
                            sampling_percentage=[0.25, 0.25, 1],
                            sampling_strategy=['Regular',
                                               'Regular',
                                               'None'],
                            shrink_factors=[[8, 4, 2, 1]]*3,
                            smoothing_sigmas=[[3, 2, 1, 0]]*3,
                            transform_parameters=[(0.1,),
                                                  (0.1,),
                                                  (0.1, 3.0, 0.0)],
                            use_histogram_matching=True,
                            write_composite_transform=True),
               name='mni_reg')

mni_reg.inputs.fixed_image = fsl.Info.standard_image('MNI152_T1_1mm_brain.nii.gz')

workflow.connect(selector, 't1_weighted', mni_reg, 'moving_image')

workflow.connect(reg, 'composite_transform', ds, 'epi2structmat_ants')
workflow.connect(reg, 'inverse_composite_transform', ds, 'struct2epimat_ants')
workflow.connect(reg, 'warped_image', ds, 'epi_in_struct_ants')

workflow.connect(mni_reg, 'composite_transform', ds, 'struct2mnimat_ants')
workflow.connect(mni_reg, 'inverse_composite_transform', ds, 'mni2structmat_ants')
workflow.connect(mni_reg, 'warped_image', ds, 'struct_in_mni_ants')

workflow.run(plugin='MultiProc', plugin_args={'n_procs':8})
