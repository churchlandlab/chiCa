#!/usr/bin/env python

"""
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.
The demo demonstrates how to use the params, MotionCorrect and cnmf objects
for processing 1p microendoscopic data. The analysis pipeline is similar as in
the case of 2p data processing with core difference being the usage of the
CNMF-E algorithm for source extraction (as opposed to plain CNMF). Check
the companion paper for more details.

You can also run a large part of the pipeline with a single method
(cnmf.fit_file) See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""

import logging
import matplotlib.pyplot as plt
import numpy as np

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params

import os #For all the file path manipulations
import json #To read the miniscope metadata
from time import time # For time logging
import cv2
import sys #To apppend the python path for the selection GUI
sys.path.append('C:/Users/Anne/Documents/chiCa') #Include the path to the functions
import neuron_selection_GUI
#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)
    # filename="/tmp/caiman.log"

# %% start the cluster
try:
    cm.stop_server()  # stop it if it was running
except():
    pass

c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=12,  # number of process to use, if you go out of memory try to reduce this one
                                                 single_thread=False)

# %% First setup some parameters for motion correction
#To let user select files
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
Tk().withdraw() #Don't let the verification window pop up
fileSet = askopenfilenames(title = "Select your imaging movie files",
                        filetypes =[("AVI files","*.avi*")])
original_files = list(fileSet)

#%%--Spatial down sampling

scaling_factor = 0.5 #Shrink image size by factor 2
cropping_ROI = None #No cropping implemented so far

preproc_t = time()

#Retrieve path and create caiman directory if necessary
file_path = os.path.split(os.path.split(original_files[0])[0])[0]
target_path = os.path.join(file_path, 'caiman')
if not os.path.isdir(target_path):
    os.mkdir(target_path)

#Start the down-sampling
fnames = [] #Prepare to link to the down-sampled files
for single_video in original_files:
    cap = cv2.VideoCapture(single_video)
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames 
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    new_dims = (int(frame_dims[0]*scaling_factor), int(frame_dims[1]*scaling_factor))
    
    output_file = os.path.join(target_path, os.path.splitext(os.path.split(single_video)[1])[0] + f'_binned.avi')
    fnames.append(output_file)
    wrt = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, new_dims)

    for k in range(frame_number):
        success, f = cap.read()
        if not success:
            print(f'Something_happend at frame {k}. Check the movie!')
        frame = np.array(f[:,:,1],) #The video is gray-scale, convert to float to be able to record negative values
        binned = cv2.resize(frame, new_dims, interpolation = cv2.INTER_AREA)
        #Rescale the image and round to integer value
        wrt.write(np.stack((binned, binned, binned), axis=2)) #Reassemble as rgb for saving      
    cap.release()
    wrt.release()
    print(f'Processed: {single_video}')

preproc_dict = dict({'scaling_factor': scaling_factor, 'cropping_ROI': cropping_ROI})
np.save(os.path.join(target_path, 'cropping_binning.npy'), preproc_dict)
print(f'Video binning done in {round(time() - preproc_t)} seconds')
print('------------------------------------------------')


#%%---Set up some of the motion correction parameters

#Retrieve frame rate from metadata file
with open(os.path.join(file_path, 'miniscope', 'metaData.json')) as json_file:
    miniscope_metadata = json.load(json_file)

fr = miniscope_metadata['frameRate']   # movie frame rate

decay_time = 0.6                 # length of a typical transient in seconds

# motion correction parameters
motion_correct = True            # flag for motion correction
pw_rigid = False                 # flag for pw-rigid motion correction

gSig_filt = (3, 3)   # size of filter, in general gSig (see below),
#                      change this one if algorithm does not work
max_shifts = (5, 5)  # maximum allowed rigid shift
strides = (48, 48)   # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3
border_nan = 'copy'

mc_dict = {
    'fnames': fnames,
    'fr': fr,
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan
}

opts = params.CNMFParams(params_dict=mc_dict)

# %% MOTION CORRECTION
#  The pw_rigid flag set above, determines where to use rigid or pw-rigid
#  motion correction
if motion_correct:
    #Track the time to compute shifts
    mc_start_time = time()

    # do motion correction rigid
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')

    bord_px = 0 if border_nan == 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=bord_px)

    mc_end_time = time()
    #Display elapsed time
    print(f"Motion correcition finished in {round(mc_end_time - mc_start_time)} s.")

else:  # if no motion correction just memory map the file
    fname_new = cm.save_memmap(fnames[0], base_name='memmap_',
                               order='C', border_to_0=0, dview=dview)
#%%-- Save the shifts, the very basic implementation
rigid_shifts = np.array(mc.shifts_rig) # Retrieve shifts from mc object

outputPath = os.path.dirname(fnames[0]) #Assuming that you want the results in the same location
np.save(outputPath + '/rigid_shifts', rigid_shifts) #Save the np array to npy file

#%%
# load memory mappable file
Yr, dims, T = cm.load_memmap(fname_new, mode='r+')
images = Yr.T.reshape((T,) + dims, order='F')


# %% Parameters for source extraction and deconvolution (CNMF-E algorithm)

p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None for 1p data
gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .7      # merging threshold, max correlation allowed
rf = 80             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .7       # min peak value from correlation image
min_pnr = 8        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

opts.change_params(params_dict={'dims': dims,
                                'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': bord_px})                # number of pixels to not consider in the borders)

#%% Remove pixels with basically zero intensity but very few

medProj = np.median(images, axis=0, keepdims=True)
median_bool = np.squeeze(medProj < 1)
for k in range(images.shape[0]):
    temp = images[k,:,:]
    temp[median_bool] = 0.0001
    images[k,:,:] = temp

# %% compute some summary images (correlation and peak to noise)
# change swap dim if output looks weird, it is a problem with tiffile
corr_image_start = time()
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
#compute the correlation and pnr image on every frame. This takes longer but will yield
#the actual correlation image that can be used later to align other sessions to this session
corr_image_end = time()
print(f"Computed correlation- and pnr images in {corr_image_end - corr_image_start} s.")

np.save(outputPath + '/spatio_temporal_correlation_image', cn_filter)
np.save(outputPath + '/median_projection', medProj)
# if your images file is too long this computation will take unnecessarily
# long time and consume a lot of memory. Consider changing images[::1] to
# images[::5] or something similar to compute on a subset of the data

# inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter, pnr)
# print parameters set above, modify them if necessary based on summary images
print(min_corr) # min correlation of peak (from correlation image)
print(min_pnr)  # min peak to noise ratio

#%% Shut donw shut down parallel pool and restart if desired
dview.terminate()
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=4,  # number of process to use, if you go out of memory try to reduce this one
                                                 single_thread=False)
# %% RUN CNMF ON PATCHES
cnmfe_start_time = time()

cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
cnm.fit(images)

#Display elapsed time
cnmfe_end_time = time()
print(f"Ran initialization and fit cnmfe model in {round(cnmfe_end_time - cnmfe_start_time)} s.")

# Save first round of results
cnm.save(outputPath + '/firstRound.hdf5')
#%% Manual curation and selection of reasonable neruons
print(f'Detected {cnm.estimates.C.shape[0]} neurons')
print('--------------------------------------------')
keep_neuron, frame_slider = neuron_selection_GUI.run_neuron_selection(data_source = cnm)
# Do the selection here. Don't advance until the selection is completed since further executions are not blocked!

#Interrupt the further execution here until the curation is completed
_ = input('Press enter after selecting the neurons')
#%% Now convert the list of accepted components to indices
#keep_idx = [i for i, val in enumerate(neurons_to_keep) if val] #Returns the index of every true entry in the list

#Plug the indices in before re-running the fitting prodecure
cnm.estimates.select_components(idx_components = keep_neuron)

#%% Restart a new parallel pool
dview.terminate()
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=6,  # number of process to use, if you go out of memory try to reduce this one
                                                 single_thread=False)

#%% Run the fitting again with only the accepted components
#cnm.refit(images, dview = dview) #Currently the rejected components are not merged back into the background
cnm.estimates.detrend_df_f() #Also reconstruct the detrended non/denoised trace
cnm.save(outputPath + '/secondRound.hdf5')

cm.stop_server(dview=dview)

# #%%-------AUXILIARY FUNCTIONS AND VISUALIZATIONS

# ##############################################################################
# # %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
#     #   you can also perform the motion correction plus cnmf fitting steps
#     #   simultaneously after defining your parameters object using
# #    cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
# #    cnm1.fit_file(motion_correct=True)

# # %% DISCARD LOW QUALITY COMPONENTS
#     min_SNR = 2.5           # adaptive way to set threshold on the transient size
#     r_values_min = 0.85    # threshold on space consistency (if you lower more components
#     #                        will be accepted, potentially with worst quality)
#     cnm.params.set('quality', {'min_SNR': min_SNR,
#                                'rval_thr': r_values_min,
#                                'use_cnn': False})
#     cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

#     print(' ***** ')
#     print('Number of total components: ', len(cnm.estimates.C))
#     print('Number of accepted components: ', len(cnm.estimates.idx_components))

# # %% PLOT COMPONENTS
#     cnm.dims = dims
#     display_images = True           # Set to true to show movies and images
#     if display_images:
#         cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
#         cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)

# # %% MOVIES
#     display_images = False           # Set to true to show movies and images
#     if display_images:
#         # fully reconstructed movie
#         cnm.estimates.play_movie(images, q_max=99.5, magnification=2,
#                                  include_bck=True, gain_res=10, bpx=bord_px)
#         # movie without background
#         cnm.estimates.play_movie(images, q_max=99.9, magnification=2,
#                                  include_bck=False, gain_res=4, bpx=bord_px)

# # %% STOP SERVER
#     cm.stop_server(dview=dview)

# # %% This is to mask the differences between running this demo in Spyder
# # versus from the CLI
# if __name__ == "__main__":
#     main()
