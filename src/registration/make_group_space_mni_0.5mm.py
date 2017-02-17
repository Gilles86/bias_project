import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import os
import glob
import numpy as np

from nipype.interfaces.c3 import C3dAffineTool
import nipype.interfaces.ants as ants

reg = pe.MapNode(ants.Registration(), iterfield=['moving_image'], name='antsRegister')
reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
reg.inputs.number_of_iterations = [[1000,500,250,100]]*2 + [[100,100,70,20]]
reg.inputs.dimension = 3
reg.inputs.write_composite_transform = True
reg.inputs.collapse_output_transforms = True
reg.inputs.metric = ['MI']*2 + ['CC']
reg.inputs.metric_weight = [1]*3 # Default (value ignored currently by ANTs)
reg.inputs.radius_or_number_of_bins = [32]*2 + [4]
reg.inputs.sampling_strategy = ['Regular']*2 + [None]
reg.inputs.sampling_percentage = [0.25]*2 + [None]
reg.inputs.convergence_threshold = [1.e-8]*2 + [1e-9]
reg.inputs.convergence_window_size = [10]*2 + [15]
reg.inputs.smoothing_sigmas = [[3,2,1,0]]*3
reg.inputs.sigma_units = ['mm']*3
reg.inputs.shrink_factors = [[8,4,2,1]]*2 + [[6,4,2,1]]
reg.inputs.use_estimate_learning_rate_once = [True, True, True]
reg.inputs.use_histogram_matching = [False]*2 + [True] # This is the default
reg.inputs.initial_moving_transform_com = True
reg.inputs.output_warped_image = True
reg.inputs.winsorize_lower_quantile = 0.01
reg.inputs.winsorize_upper_quantile = 0.99
reg.inputs.fixed_image = '/home/gdholla1/data/MNI_T2_0.5mm_brain.nii.gz'


workflow = pe.Workflow(name='make_group_template_ants_mni', base_dir='/home/gdholla1/workflow_folders/')

input_node = pe.Node(util.IdentityInterface(fields=['FLASH']), name='input_node')

templates = {'FLASH1':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_T2_weighted_flash_echo_11.22.nii.gz',
             'FLASH2':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_T2_weighted_flash_echo_20.39.nii.gz',
             'FLASH3':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_T2_weighted_flash_echo_29.57.nii.gz'}



selector = pe.MapNode(nio.SelectFiles(templates), iterfield='subject_id', name='selector')
subject_ids = ['%02d' % i for i in np.arange(1, 20)]
selector.inputs.subject_id = subject_ids

flash_list_merger = pe.MapNode(util.Merge(numinputs=3), iterfield=['in1', 'in2', 'in3'], name='flash_list_merger')
workflow.connect(selector, 'FLASH1', flash_list_merger, 'in1')
workflow.connect(selector, 'FLASH2', flash_list_merger, 'in2')
workflow.connect(selector, 'FLASH3', flash_list_merger, 'in3')

echo_merger = pe.MapNode(fsl.Merge(dimension='t'), iterfield=['in_files'], name='echo_merger')
workflow.connect(flash_list_merger, 'out', echo_merger, 'in_files')


echo_meaner = pe.MapNode(fsl.MeanImage(dimension='T'), iterfield=['in_file'], name='echo_meaner')
workflow.connect(echo_merger, 'merged_file', echo_meaner, 'in_file')

better = pe.MapNode(fsl.BET(frac=0.3), iterfield=['in_file'], name='better')
better.inputs.padding = True

workflow.connect(echo_meaner, 'out_file', better, 'in_file')

# get_first = lambda x: x[0]

n_iterations = 5

ants_registers = []
mergers = []
meaners = []

ds = pe.Node(nio.DataSink(base_directory='/home/gdholla1/data/bias/mni_group_template'), name='datasink')

for i in xrange(n_iterations):
    ants_registers.append(reg.clone('register_%d' % (i + 1)))
    
    workflow.connect(better, 'out_file', ants_registers[-1], 'moving_image')
    
#     if i == 0:
#         workflow.connect(input_node, ('FLASH', get_first), ants_registers[-1], 'fixed_image')
#         ants_registers[0].inputs.fixed_image = '/home/gdholla1/data/MNI_T2_0.5mm_brain.nii.gz'
#     else:
    if i > 0:
        workflow.connect(meaners[-1], 'out_file', ants_registers[-1], 'fixed_image')        
    
    mergers.append(pe.Node(fsl.Merge(dimension='t'), name='merger%d' % (i+1)))
    meaners.append(pe.Node(fsl.MeanImage(dimension='T'), name='meaner%d' % (i+1)))
    
    workflow.connect(ants_registers[-1], 'warped_image', mergers[-1], 'in_files')
    workflow.connect(ants_registers[-1], 'warped_image', ds, 'warped_image_%d_iterations' % (i+1))

    workflow.connect(mergers[-1], 'merged_file', meaners[-1], 'in_file')
    
    workflow.connect(ants_registers[-1], 'composite_transform', ds, 'composite_transform_%d_iterations' % (i+1))
    
    workflow.connect(meaners[-1], 'out_file', ds, 'template_%d_iterations' % (i+1))
    
workflow.write_graph()

#workflow.run()
workflow.run(plugin='MultiProc', plugin_args={'n_procs':9})
