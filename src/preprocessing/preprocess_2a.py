from nipype.workflows.fmri.fsl import create_featreg_preproc
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from nipype.algorithms.misc import TSNR
from nipype.interfaces import utility as util
from nipype.interfaces import fsl
import os

workflow = create_featreg_preproc() #The default name is "featpreproc".
workflow.base_dir = '/home/gdholla1/workflow_folders'

#templates = {'functional_runs':'/home/gdholla1/data/bias_task/preprocessed/slice_time_corrected_file/_subject_id_{subject_id}/_slice_corrector*/run*_unwarped_st.nii.gz'}

project_folder = '/home/gdholla1/projects/bias/'

templates = {'functional_runs':os.path.join(project_folder, 'data', 'processed', 'slice_time_corrected_file', '_subject_id_{subject_id}', '_slice_corrector*', 'sub-{subject_id}*.nii.gz')}

subject_ids = ['%02d' % i for i in range(1, 20)]

selector = pe.Node(nio.SelectFiles(templates), name='selector')
selector.iterables = [('subject_id', subject_ids)]

workflow.connect(selector, 'functional_runs', workflow.get_node('inputspec'), 'func' )

#workflow.inputs.inputspec.fwhm = 5
#workflow.get_node('inputspec').iterables = [('fwhm', [0.0, 1.5, 5.0])]
workflow.get_node('inputspec').iterables = [('fwhm', [0.0, 1.5, 5.0])]
workflow.inputs.inputspec.highpass = True

ds = pe.Node(nio.DataSink(), name='datasink')
#ds.inputs.base_directory = '/home/gdholla1/data/bias_task/preprocessed/feat_preprocess/'
ds.inputs.base_directory = os.path.join(project_folder, 'data', 'processed', 'feat_preprocess')

workflow.connect(workflow.get_node('outputspec'), 'mean', ds, 'mean')
workflow.connect(workflow.get_node('outputspec'), 'highpassed_files', ds, 'highpassed_files')
workflow.connect(workflow.get_node('outputspec'), 'mask', ds, 'mask')
workflow.connect(workflow.get_node('outputspec'), 'motion_parameters', ds, 'motion_parameters')
workflow.connect(workflow.get_node('outputspec'), 'motion_plots', ds, 'motion_plots')

tsnr_node = pe.MapNode(TSNR(), iterfield=['in_file'], name='tsnr')
workflow.connect(workflow.get_node('outputspec'), 'realigned_files', tsnr_node, 'in_file')
workflow.connect(tsnr_node,  'tsnr_file', ds, 'tsnr')


def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative

    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    import os
    from nipype.utils.filemanip import filename_to_list
    import numpy as np
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor%02d.txt" % idx)
        np.savetxt(filename, out_params2, fmt=str("%.10f"))
        out_files.append(filename)
    return out_files


make_motion_regressors = pe.Node(util.Function(input_names=['motion_params'],
                                                  output_names=['motion_glm'],
                                                  function=motion_regressors),
                                    name='make_motion_regressors')

workflow.connect(workflow.get_node('outputspec'), 'motion_parameters', make_motion_regressors, 'motion_params')

filter_out_motion = pe.MapNode(fsl.FilterRegressor(filter_all=True),
                               iterfield=['in_file','design_file'],
                               name='filter_out_motion')
workflow.connect(make_motion_regressors, 'motion_glm', filter_out_motion, 'design_file')

workflow.connect(workflow.get_node('highpass'), 'out_file', filter_out_motion, 'in_file') 

addmean = pe.MapNode(interface=fsl.BinaryMaths(operation='add'),
                         iterfield=['in_file', 'operand_file'],
                             name='addmean2')

workflow.connect(filter_out_motion, 'out_file', addmean, 'in_file')
workflow.connect(workflow.get_node('meanfunc4'), 'out_file', addmean, 'operand_file')

workflow.connect(addmean, 'out_file', ds, 'motion_regressors_filtered_files')


def make_r2(original, residuals):
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import split_filename
    import os

    _, fname, ext = split_filename(residuals)
    
    original = nb.load(original)
    residuals = nb.load(residuals)

    original_var = np.var(original.get_data(), -1)
    residuals_var = np.var(residuals.get_data(), -1)

    r2 = (original_var - residuals_var) / original_var
    
    new_fn = os.path.abspath('{fname}_r2{ext}'.format(**locals()))
    
    nb.save(nb.Nifti1Image(r2, original.get_affine()), new_fn)
    
    return new_fn

r2_node = pe.MapNode(util.Function(input_names=['original', 'residuals'],
                                       output_names=['r2'],
                                       function=make_r2),
                        iterfield=['original', 'residuals'],
                        name='get_r2')

workflow.connect(filter_out_motion, 'out_file', r2_node, 'residuals')
workflow.connect(workflow.get_node('highpass'), 'out_file', r2_node, 'original') 

workflow.connect(r2_node, 'r2', ds, 'motion_regressors_r2')


workflow.write_graph()

from nipype import config
config.set('execution', 'remove_unnecessary_outputs', 'true')

workflow.run(plugin='MultiProc', plugin_args={'n_procs':6})
#workflow.run()



