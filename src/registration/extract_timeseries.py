from nipype.interfaces import fsl
from nipype.interfaces import io as nio
import nipype.pipeline.engine as pe
import os
import numpy as np

project_folder = '/home/gdholla1/projects/bias'


templates = {'mask_epi':os.path.join(project_folder, 'data', 'derivatives', 'sub-{subject_id}', 'masks', 'epi_space', 'sub-{subject_id}_mask-{mask}_trans.nii.gz'),
             'highpassed_motion':os.path.join(project_folder, 'data', 'processed', 'feat_preprocess', 'motion_regressors_filtered_files', '_subject_id_{subject_id}', '_fwhm_0.0', '_addmean2*', 'sub-{subject_id}*.nii.gz'),}

workflow = pe.Workflow(name='extract_signal')
workflow.base_dir = os.path.join(project_folder, 'workflow_folders')

selector = pe.Node(nio.SelectFiles(templates), name='selector')

subject_ids = ['%02d' % i for i in np.arange(1, 20)]

selector.iterables = [('subject_id', subject_ids),
                      ('mask', ['STh_L', 'STh_R', 'STh_L_A', 'STh_L_B', 'STh_L_C', 'STh_R_A', 'STh_R_B', 'STh_R_C'])]


ds_extracted =  pe.Node(nio.DataSink(), name='datasink_extracted')
ds_extracted.inputs.base_directory = os.path.join(project_folder, 'data', 'derivatives', 'extracted_signal')


extracter_highpassed_motion = pe.MapNode(fsl.ImageMeants(), iterfield=['in_file'], name='extracter_highpassed_motion')
workflow.connect(selector, 'mask_epi', extracter_highpassed_motion, 'mask')
workflow.connect(selector, 'highpassed_motion', extracter_highpassed_motion, 'in_file')
workflow.connect(extracter_highpassed_motion, 'out_file', ds_extracted, 'highpassed_motion')

#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 6})
workflow.run()
