import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from nipype.interfaces import ants
from nipype.interfaces import fsl
import numpy as np
import os

project_folder = '/home/gdholla1/projects/bias'

templates = {'mask':os.path.join(project_folder, 'data', 'derivatives', 'sub-{subject_id}', 'masks', 'FLASH_space', 'sub-{subject_id}_mask-{mask}.nii.gz'),
        'flash2epi':os.path.join(project_folder, 'data', 'derivatives', 'registration', 'epi2flash', 'FLASH2epi_transform', '_subject_id_{subject_id}', 'transformInverseComposite.h5'),
         'mean_flash':os.path.join(project_folder, 'data', 'derivatives', 'sub-{subject_id}', 'mean_flash', 'sub-{subject_id}_FLASH_echo_11.22_merged_mean_brain.nii.gz'),
        'mean_epi':os.path.join(project_folder, 'data', 'processed', 'feat_preprocess', 'mean', '_subject_id_{subject_id}', '_fwhm_0.0', 'sub-{subject_id}_task-randomdotmotion_run-01_bold_unwarped_st_dtype_mcf_mask_gms_mean.nii.gz')}
        
workflow = pe.Workflow(name='transform_masks')
workflow.base_dir = '../../workflow_folders'

selector = pe.Node(nio.SelectFiles(templates), name='selector')
subject_ids = ['%02d' % i for i in np.arange(1, 20)]
selector.iterables = [('subject_id', subject_ids), ('mask', ['STh_L', 'STh_R'])]

# We need to copy the geometrical information of the FLASH image to the mask, because
# it was changed by the padded BET-procedure.
copygeom = pe.Node(fsl.CopyGeom(), name='copygeom')
workflow.connect(selector, 'mask', copygeom, 'dest_file')
workflow.connect(selector, 'mean_flash', copygeom, 'in_file')

transformer = pe.Node(ants.ApplyTransforms(interpolation='NearestNeighbor'), name='transformer')
workflow.connect(selector, 'flash2epi', transformer, 'transforms')
workflow.connect(selector, 'mean_epi', transformer, 'reference_image')
workflow.connect(copygeom, 'out_file', transformer, 'input_image')

ds =  pe.Node(nio.DataSink(), name='datasink')
ds.inputs.base_directory = os.path.join(project_folder, 'data', 'derivatives', 'masks', )
ds.inputs.regexp_substitutions = [('/masks/epi_space/_mask_.*_subject_id_([0-9]+)', '/sub-\\1/masks/epi_space/')]


workflow.connect(transformer, 'output_image', ds, 'epi_space')


workflow.run()


