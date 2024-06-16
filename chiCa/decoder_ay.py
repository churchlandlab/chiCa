#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:31:06 2024

@author: abbieyu
"""

import numpy as np
import pandas as pd
import glob
#from time import time
from os.path import splitext, join
# import sys
#sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa')
#import decoding_utils
#sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/analysis_sandbox')
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.ndimage import gaussian_filter1d
# sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/regbench')
# import regbench as rb
import chiCa
import matplotlib.pyplot as plt

#%% Parameter definitions

session_dir = '/Users/abbieyu/Documents/UCLA/Yr1/S24_churchland/data/LO032/20220923_135753/'

file_name = ''

k_folds = 10 #For regular random cross-validation

which_models = 'cognitive' #'group' # 'individual', 'timepoint'
#Determines what type of models should be fitted. 'group' will lump specified regressors
#into a group and fit models for the cvR2 and dR2 for each of the groups, 'individual'
#will assess the explained variance for each regressor alone and 'timepoint' will
#look at the task regressors collectively but evaluate the model performance at each
#trial time separately

add_complete_shuffles = 0 #Allows one to add models where all the regressors
#are shuffled idependently. This can be used to generate a null distribution for
#the beta weights of certain regressors

use_parallel = False #Whether to do parallel processing on the different shuffles
exclude_video_me = False #Do not use video me for this run

#%% Load data

trial_alignment_file = glob.glob(session_dir + '/analysis/*miniscope_data.npy')[0]
miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
frame_rate = miniscope_data['frame_rate']
trialdata = pd.read_hdf(glob.glob(session_dir + '/chipmunk/*.h5')[0], '/Data')

#Get video alignment
video_alignment_files = glob.glob(session_dir + '/analysis/*video_alignment.npy')
if len(video_alignment_files) > 1:
        print('More than one video is currently not supported')
video_alignment = np.load(video_alignment_files[0], allow_pickle = True).tolist()

#Retrieve dlc tracking
dlc_file = glob.glob(session_dir + '/dlc_analysis/*.h5')[-1] #Load the latest extraction! - loads as string instead of list
dlc_data = pd.read_hdf(dlc_file)
dlc_metadata = pd.read_pickle(splitext(dlc_file)[0] + '_meta.pickle')

#%% Alignments and construction of the design matrix

#SET UP TIMES
#aligned_to is a list of the states you want to look at 
#time_frame is a list of the same size as aligned_to that corresponds to the number of frames before and after the start of each state you want to include 
aligned_to = ['DemonInitFixation', 'PlayStimulus', 'DemonWaitForResponse', 'outcome_presentation']
time_frame = [np.array([round(-1*frame_rate), round(0*frame_rate)+1], dtype=int), np.array([0, round(1*frame_rate)+1], dtype=int),
              np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int), np.array([round(0*frame_rate), round(2*frame_rate)], dtype=int)]

#ASSEMBLE TASK VARIABLES FOR DESIGN MATRIX
choice = np.array(trialdata['response_side']) #what the animal chose: left=0, right=1
category = np.array(trialdata['correct_side']) #what was the correct side: left=0, right=1
prior_choice =  chiCa.determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
prior_category =  chiCa.determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')

outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
prior_outcome =  chiCa.determine_prior_variable(outcome, np.ones(len(trialdata)), 1, 'consecutive')

#FIND VALID TRIALS TO BE INCLUDED
#criteria are the following:
#there has to be a prior choice a current choice and the stimulus is one of the easy two
valid_trials_bool = (np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0)
valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]


#FIND STIMULUS STRENGTHS
tmp_stim_strengths = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
for k in range(trialdata.shape[0]):
    tmp_stim_strengths[k] = trialdata['stimulus_event_timestamps'][k].shape[0]

#GET CATEGORY BOUNDARY TO NORMALIZE THE STIM STRENGTHS - i think i dont need this 
# unique_freq = np.unique(tmp_stim_strengths)
# category_boundary = (np.min(unique_freq) + np.max(unique_freq))/2
# stim_strengths = (tmp_stim_strengths- category_boundary) / (np.max(unique_freq) - category_boundary)
# if trialdata['stimulus_modality'][0] == 'auditory':
#     stim_strengths = stim_strengths * -1


#EXTRACT A SET OF DLC LABELS AND STANDARDIZE THESE 
dlc_keys = dlc_data.keys().tolist()
specifier = dlc_keys[0][0] #Retrieve the session specifier that is needed as a keyword
body_part_name = dlc_metadata['data']['DLC-model-config file']['all_joints_names']
bp_name_duplicate=[]
for x in body_part_name:
    bp_name_duplicate.append(x)
    bp_name_duplicate.append(x)

temp_body_parts = []
part_likelihood_estimate = []
for bp in body_part_name:
    for axis in ['x', 'y']:
        temp_body_parts.append(np.array(dlc_data[(specifier, bp, axis)]))
    part_likelihood_estimate.append(np.array(dlc_data[(specifier, bp, 'likelihood')]))

body_parts = np.array(temp_body_parts).T #To array and transpose
part_likelihood_estimate = np.array(part_likelihood_estimate).T

head_orientation=np.vstack((miniscope_data['pitch'],miniscope_data['roll'],miniscope_data['yaw'])).T
head_orientation_names=['pitch','roll','yaw']

#GET NEURAL SIGNALS
signal_type='S'
if signal_type == 'S':
    signal = gaussian_filter1d(miniscope_data[signal_type].T,1, axis=0, mode='constant', cval=0, truncate=4)
    #Pad the first and last samples with zeros in this condition
else:
    signal = miniscope_data[signal_type].T #Transpose the original signal so that rows are timepoints and columns cells
#determine which neurons to keep
keep_neuron = np.arange(signal.shape[1])

#Align signal to the respsective state and retrieve the data
Y_test = []
x_analog_test = []
part_likelihood = []
#Also store the timestamps that are included into the trialized design matrix
trial_timestamps_imaging = []
trial_timestamps_video = []

for k in range(len(aligned_to)):
    state_start_frame, state_time_covered = chiCa.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
    zero_frame = np.array(state_start_frame[valid_trials] + time_frame[k][0], dtype=int) #The firts frame to consider
    
    for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
        matching_frames = []
        for q in range(zero_frame.shape[0]): #unfortunately need to loop through the trials, should be improved in the future...
                tmp = chiCa.match_video_to_imaging(np.array([zero_frame[q] + add_to]), miniscope_data['trial_starts'][valid_trials[q]],
                       miniscope_data['frame_interval'], video_alignment['trial_starts'][valid_trials[q]], video_alignment['frame_interval'])[0].astype(int)
                matching_frames.append(tmp)
        
        Y_test.append(signal[zero_frame + add_to,:][:, keep_neuron])
        x_analog_test.append(np.concatenate((head_orientation[zero_frame + add_to,:], body_parts[matching_frames,:]), axis=1))
        part_likelihood.append(part_likelihood_estimate[matching_frames,:])
        
        trial_timestamps_imaging.append(zero_frame + add_to)
        trial_timestamps_video.append(matching_frames)
        
#Back to array like, where columns are trials, rows time points and sheets cells   
Y_test = np.squeeze(Y_test)
x_analog_test = np.squeeze(x_analog_test)
#%% make my little plots

#STATE DEFINITIONS

state1frames=[0,31] #does not include 31
state2frames=[31,62]
state3frames=[62,78]
state4frames=[78,138]

#look at the third state and plot chest center for all trials where the mouse 
#went Left in blue and Right in Red 

test_frames_x_analog=x_analog_test[62:78,:,11:13] #63-79 are only third state and 11-13 are chest center

another_test=choice[valid_trials] == 0 #tells me where the choice is left
        
state2=np.arange(62,78)
chest=np.arange(11,13)

y_pos_chest_l=test_frames_x_analog[:,another_test,1]
x_pos_chest_l=test_frames_x_analog[:,another_test,0]
x_pos_chest_r=test_frames_x_analog[:,another_test==False,0]
y_pos_chest_r=test_frames_x_analog[:,another_test==False,1]

plt.plot(x_pos_chest_l,y_pos_chest_l,'b',linewidth=0.5)
plt.plot(x_pos_chest_r,y_pos_chest_r,'r',linewidth=0.5)
plt.xlim(0,640)
plt.ylim(0,512)
