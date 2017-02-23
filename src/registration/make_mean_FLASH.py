import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import os
import glob
import numpy as np

from nipype.interfaces.c3 import C3dAffineTool
import nipype.interfaces.ants as ants

projects_dir = '/home/gdholla1/projects/bias'

workflow = pe.Workflow(name='make_mean_flash')
workflow.base_dir = os.path.join(projects_dir, 'workflow_folders')

templates = {'FLASH1':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_FLASH_echo_11.22.nii.gz',
             'FLASH2':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_FLASH_echo_20.39.nii.gz',
             'FLASH3':'/home/gdholla1/projects/bias/data/raw/sub-{subject_id}/anat/sub-{subject_id}_FLASH_echo_29.57.nii.gz'}



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


ds = pe.Node(nio.DataSink(), name='datasink')
ds.inputs.regexp_substitutions = [('/_echo_meaner([0-9])+/', '/'),
                                  ('/_better[0-9]+/', '/'),
                                  ('/sub-([0-9]{2})', '/sub-\\1/mean_flash/sub-\\1')]
                                   
ds.inputs.base_directory = os.path.join(projects_dir, 'data')
workflow.connect(echo_meaner, 'out_file', ds, 'derivatives')
workflow.connect(better, 'out_file', ds, 'derivatives.@bet')

workflow.write_graph()
workflow.run(plugin='MultiProc', plugin_args={'n_procs':4})
