import nipype.pipeline.engine as pe

from nipype.interfaces import fsl
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util

workflow = pe.Workflow(name='field_correction_bias', base_dir='/home/gdholla1/workflow_folders/')

templates = {'fieldmap_magnitude':'/home/gdholla1/data/bias_task/clean/{subject_id}/fieldmap_te7.02_magnitude.nii',
                     'fieldmap_phase':'/home/gdholla1/data/bias_task/clean/{subject_id}/fieldmap_te7.02_phase.nii',
                                  'functional_runs':'/home/gdholla1/data/bias_task/clean/{subject_id}/run*.nii'}

selector = pe.Node(nio.SelectFiles(templates), name='selector')

#subject_ids = ['PF5T', 'WW2T', 'WSFT', 'KP6T', 'LV2T', 'FMFT', 'HCBT', 'RSIT', 'TS6T', 'UM2T', 'MRCT', 'NM3T', 'SPGT', 'ZK4T', 'GAIT', 'DA9T', 'VL1T']
subject_ids = ['FMFT']

selector.iterables = [('subject_id', subject_ids)]

better = pe.Node(fsl.BET(), name='better')

# better.inputs.in_file = '/home/gdholla1/data/bias_task/clean/FMFT/fieldmap_te6.nii'
better.inputs.mask = True
better.inputs.frac = 0.5

workflow.connect(selector, 'fieldmap_magnitude', better, 'in_file')

def erode(in_file, iterations=2):
    import nibabel as nb
    import scipy as sp
    from scipy import ndimage
    import os

    image = nb.load(in_file)
    data = image.get_data()
    new_data = data.copy()

    new_mask = ndimage.binary_erosion(data > 0, iterations=3)
    new_data[new_mask == False] = 0

    from nipype.utils.filemanip import split_filename

    _, fn, ext = split_filename(in_file)
    fn = os.path.abspath('{fn}_eroded{iterations}{ext}'.format(fn=fn, iterations=iterations, ext=ext))

    nb.save(nb.Nifti1Image(new_data, image.get_affine()), fn)

    return fn

eroder = pe.Node(util.Function(function=erode,
                               input_names=['in_file'],
                               output_names=['eroded_image'],), name='eroder')

workflow.connect(better, 'out_file', eroder, 'in_file')

prepare_fieldmap = pe.Node(fsl.epi.PrepareFieldmap(), name='prepare_fieldmap')

prepare_fieldmap.inputs.delta_TE = 1.02
prepare_fieldmap.inputs.in_magnitude = '/home/gdholla1/notebooks/2015_leipzig_bias/fieldmap_te6_brain_eroded2.nii.gz'
prepare_fieldmap.inputs.in_phase = '/home/gdholla1/data/bias_task/clean/FMFT/fieldmap_te7.02_phase.nii'

workflow.connect(eroder, 'eroded_image', prepare_fieldmap, 'in_magnitude')
workflow.connect(selector, 'fieldmap_phase', prepare_fieldmap, 'in_phase')

def pickone(x):
    return x[0]

convert_fieldmap = pe.Node(fsl.ApplyXfm(), name='resample_fieldmap')

convert_fieldmap.inputs.in_matrix_file = '/usr/share/fsl/5.0/etc/flirtsch/ident.mat'


workflow.connect(prepare_fieldmap, 'out_fieldmap', convert_fieldmap, 'in_file')
workflow.connect(selector, ('functional_runs', pickone), convert_fieldmap, 'reference')

corrector = pe.MapNode(fsl.FUGUE(), iterfield=['in_file'], name='corrector')

corrector.inputs.smooth3d = 2.5
corrector.inputs.dwell_time = 1 / (29.29700089 * 128)
corrector.inputs.unwarp_direction = 'y-'

workflow.connect(convert_fieldmap, 'out_file', corrector, 'fmap_in_file')
workflow.connect(selector, 'functional_runs', corrector, 'in_file')

slice_corrector = pe.MapNode(fsl.SliceTimer(), iterfield=['in_file'], name='slice_corrector')
slice_corrector.inputs.interleaved = True
slice_corrector.inputs.index_dir = True

workflow.connect(corrector, 'unwarped_file', slice_corrector, 'in_file')

ds = pe.Node(nio.DataSink(base_directory='/home/gdholla1/data/bias_task/preprocessed'), name='datasink')

workflow.connect(corrector, 'unwarped_file', ds, 'field_corrected_file')
workflow.connect(slice_corrector, 'slice_time_corrected_file', ds, 'slice_time_corrected_file')

#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 6})
workflow.run()
