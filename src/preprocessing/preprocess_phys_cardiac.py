import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.utility as util
from nipype.algorithms import misc

from nipype.workflows.fmri.fsl import create_featreg_preproc

import glob, shutil, re, os


workflow = pe.Workflow('regress_out_phys_bias', base_dir='/home/gdholla1/workflow_folders/')

afni.base.AFNICommand.set_default_output_type('NIFTI_GZ')

templates = {'field_corrected_epi':'/home/gdholla1/data/bias_task/preprocessed/field_corrected_file/_subject_id_{subject_id}/_corrector*/run{run}_unwarped.nii.gz'}

selector = pe.Node(nio.SelectFiles(templates), name='selector')

#subject_ids = ['PF5T', 'WW2T', 'KP6T', 'LV2T', 'FMFT', 'HCBT', 'TS6T', 'UM2T', 'MRCT', 'NM3T', 'SPGT', 'ZK4T', 'GAIT', 'DA9T', 'VL1T']
subject_ids = ['FMFT']

identity = pe.Node(util.IdentityInterface(fields=['subject_id', 'run']), name='identity')
identity.iterables = [('subject_id', subject_ids)]
identity.inputs.run = [1, 2, 3]

workflow.connect(identity, 'subject_id', selector, 'subject_id')
workflow.connect(identity, 'run', selector, 'run')

def get_physiological_regressor(subject_id, run):
    import numpy as np
    import scipy as sp
    from scipy import signal
    import os
    
    tr = 3.0
    
    samples_per_second = float(open('/home/gdholla1/data/bias_task/physiological_recordings/%s/sample_rate.txt' % (subject_id)).read())


    card = np.load('/home/gdholla1/data/bias_task/physiological_recordings/%s/run%s_channel_pulse_waveform.npy' % (subject_id, run))
    trs = np.load('/home/gdholla1/data/bias_task/physiological_recordings/%s/run%s_channel_digital_input.npy' % (subject_id, run))
    


    resp_fn = '/home/gdholla1/data/bias_task/physiological_recordings/%s/run%s_channel_respiration_waveform.npy' % (subject_id, run)

    if not os.path.exists(resp_fn):
        resp_fn = '/home/gdholla1/data/bias_task/physiological_recordings/%s/run%s_channel_respiration_belt.npy' % (subject_id, run)

    if not os.path.exists(resp_fn):
        resp = np.zeros_like(card)
    else:
        resp = np.load(resp_fn)

    # Artificially include time triggers where they should be but there was an error
    trs[::tr*samples_per_second] = True

    if samples_per_second != 1000.0:
        
        resp = sp.signal.signaltools.resample(resp, resp.shape[0] * 1000.0 / samples_per_second)
        card = sp.signal.signaltools.resample(card, card.shape[0] * 1000.0 / samples_per_second)       
        trs = np.zeros_like(card)
        trs[::tr*1000.0] = 1
    
    print resp.shape, card.shape, trs.shape

    phys = np.concatenate((resp[:, np.newaxis], card[:, np.newaxis], trs[:, np.newaxis]), 1)

    np.savetxt('phys_%s_%s.txt' % (subject_id, run), phys)
        
    return os.path.abspath('phys_%s_%s.txt' % (subject_id, run))
        
get_phys_node = pe.MapNode(util.Function(function=get_physiological_regressor, 
                                      input_names=['subject_id', 'run'],
                                      output_names=['physiological_regressors']),
                        iterfield=['run'],
                        name='get_phys_node')

workflow.connect([(identity, get_phys_node, [('subject_id', 'subject_id'),
                                        ('run', 'run')])])

prepare_pnm = pe.MapNode(fsl.PreparePNM(), iterfield=['in_file'], name='prepare_pnm')
prepare_pnm.inputs.sampling_rate = 1000
prepare_pnm.inputs.tr = 3.0

workflow.connect(get_phys_node, 'physiological_regressors', prepare_pnm, 'in_file')


pnms_to_evs = pe.MapNode(fsl.PNMtoEVs(), iterfield=['cardiac', 'resp', 'functional_epi'], name='pnms_to_evs')
pnms_to_evs.inputs.verbose = False
pnms_to_evs.inputs.sliceorder = 'interleaved_down'
pnms_to_evs.inputs.order_cardiac = 4
pnms_to_evs.inputs.order_resp = 0
pnms_to_evs.inputs.order_resp_interact = 0
pnms_to_evs.inputs.order_cardiac_interact = 0
pnms_to_evs.inputs.tr = 3.0


def tile_nifti(in_files, tile_shape):
    import nibabel as nb
    import numpy as np
    import os
    from nipype.utils.filemanip import split_filename

    if len(tile_shape) == 3:
        tile_shape = tile_shape + (1,)

    fns =[]
        
    for in_file in in_files:
        image = nb.load(in_file)
        
        new_data = np.tile(image.get_data(), tile_shape)
        
        path, outname, ext = split_filename(in_file) 
        
        fn = os.path.abspath(outname + '_tiled' + ext)
        
        nb.save(nb.Nifti1Image(new_data, image.get_affine()), fn)
        fns.append(fn)

    return fns

get_mean = pe.MapNode(fsl.MeanImage(), iterfield=['in_file'], name='get_mean')

subtract = pe.MapNode(interface=fsl.ImageMaths(), iterfield=['in_file', 'in_file2'], name="subtract")
subtract.inputs.op_string = "-sub"


tiler = pe.MapNode(util.Function(function=tile_nifti, 
                              input_names=['in_files', 'tile_shape'],
                              output_names=['out_file']),
                   iterfield=['in_files'],
                   name='tiler')
tiler.inputs.tile_shape = (128, 128, 1)

workflow.connect(prepare_pnm, 'card', pnms_to_evs, 'cardiac')
workflow.connect(prepare_pnm, 'resp', pnms_to_evs, 'resp')

workflow.connect(selector, 'field_corrected_epi', get_mean, 'in_file')
workflow.connect(selector, 'field_corrected_epi', subtract, 'in_file')
workflow.connect(get_mean, 'out_file', subtract, 'in_file2')

workflow.connect(selector, 'field_corrected_epi', pnms_to_evs, 'functional_epi')

workflow.connect(pnms_to_evs, 'evs', tiler, 'in_files')

get_residuals = pe.MapNode(interface=fsl.ImageMaths(), iterfield=['in_file', 'in_file2'], name="get_residuals")
get_residuals.inputs.op_string = "-sub"



glm_fitter = pe.MapNode(afni.TFitter3D(), iterfield=['lhs', 'rhs'], name='glm_fitter')
glm_fitter.inputs.out_fitts = True
glm_fitter.inputs.out_errsum = True
glm_fitter.inputs.out_betas  = True


workflow.connect(glm_fitter, 'fitts', get_residuals, 'in_file2')
workflow.connect(selector, 'field_corrected_epi', get_residuals, 'in_file')

ds = pe.Node(nio.DataSink(base_directory='/home/gdholla1/data/bias_task/preprocessed/phys_filtering/cardiac_only'), name='datasink')

workflow.connect(get_residuals, 'out_file', ds, 'phys_correction.residuals')
workflow.connect(tiler, 'out_file', glm_fitter, 'lhs')
workflow.connect(subtract, 'out_file', glm_fitter, 'rhs')


workflow.connect(glm_fitter, 'errsum', ds, 'phys_correction.errsum')
workflow.connect(glm_fitter, 'betas', ds, 'phys.correction.betas')



from nipype import config
config.set('execution', 'remove_unnecessary_outputs', 'true')

workflow.write_graph()
workflow.run()
