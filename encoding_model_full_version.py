# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:58:07 2023

@author: Lukas Oesch
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
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter1d
# sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/regbench')
# import regbench as rb
import chiCa

#%%------Some parameter definitions--------------------
session_dir = 'C:/data/LO032/20220923_135753'

file_name = 'full_size_encoding_models'

signal_type = 'F'

k_folds = 10 #For regular random cross-validation

fit_regressor_group = True #Groups refers to a set of regressors obtained from one modality, like cognitive regressors from
#task events, or head orientation from miniscope gyro.
#If set to false the model will be fit to individual regressors, excluding
#the intercept regressors

add_complete_shuffles = 0 #Allows one to add models where all the regressors
#are shuffled idependently. This can be used to generate a null distribution for
#the beta weights of certain regressors

use_parallel = True #Whether to do parallel processing on the different shuffles

#%%------Loading the data--------------------
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
dlc_file = glob.glob(session_dir + '/dlc_analysis/*.h5')[-1] #Load the latest extraction!
dlc_data = pd.read_hdf(dlc_file)
dlc_metadata = pd.read_pickle(splitext(dlc_file)[0] + '_meta.pickle')

#Load the video components and the video motion energy components
video_svd = np.load(glob.glob(session_dir + '/analysis/*video_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components
me_svd = np.load(glob.glob(session_dir + '/analysis/*motion_energy_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components 

#%%--------Alignments and construction of the design matrix-----------------

# #Define the range of regularization strengths to search through
# #Find the one regularization for all the neurons
# regularization_strengths = (10**(np.arange(7))).tolist()

# #Determine the best penalty for each neuron individually
# regularization_strengths = [np.tile(x, (miniscope_data[signal_type].shape[0],1)) for x in (10**(np.arange(7)))]

#Set the times up
aligned_to = ['DemonInitFixation', 'PlayStimulus', 'DemonWaitForResponse', 'outcome_presentation']
time_frame = [np.array([round(-1*frame_rate), round(0*frame_rate)+1], dtype=int), np.array([0, round(1*frame_rate)+1], dtype=int),
              np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int), np.array([round(0*frame_rate), round(2*frame_rate)], dtype=int)]

# aligned_to = ['PlayStimulus', 'outcome_presentation']
# time_frame =  time_frame = [np.array([round(-0.2 * frame_rate), round(1*frame_rate)+1], dtype=int), np.array([round(-0.2*frame_rate), round(1*frame_rate)+1], dtype=int)]

#Assemble the task variable design matrix
choice = np.array(trialdata['response_side'])
category = np.array(trialdata['correct_side'])
prior_choice =  chiCa.determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
prior_category =  chiCa.determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')

outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
prior_outcome =  chiCa.determine_prior_variable(outcome, np.ones(len(trialdata)), 1, 'consecutive')

#Find stimulus strengths
tmp_stim_strengths = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
for k in range(trialdata.shape[0]):
    tmp_stim_strengths[k] = trialdata['stimulus_event_timestamps'][k].shape[0]

#Get category boundary to normalize the stim strengths
unique_freq = np.unique(tmp_stim_strengths)
category_boundary = (np.min(unique_freq) + np.max(unique_freq))/2
stim_strengths = (tmp_stim_strengths- category_boundary) / (np.max(unique_freq) - category_boundary)
if trialdata['stimulus_modality'][0] == 'auditory':
    stim_strengths = stim_strengths * -1

#Retrieve head direction and map to carthesian coordinates
head_orientation = np.stack((np.cos(miniscope_data['pitch']), np.sin(miniscope_data['pitch']),
                             np.cos(miniscope_data['roll']), np.sin(miniscope_data['roll']),
                             np.cos(miniscope_data['yaw']), np.sin(miniscope_data['yaw'])),axis=1)


#Extract a set of dlc labels and standardize these.
dlc_keys = dlc_data.keys().tolist()
specifier = dlc_keys[0][0] #Retrieve the session specifier that is needed as a keyword
body_part_name = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

temp_body_parts = []
part_likelihood_estimate = []
for bp in body_part_name:
    for axis in ['x', 'y']:
        temp_body_parts.append(np.array(dlc_data[(specifier, bp, axis)]))
    part_likelihood_estimate.append(np.array(dlc_data[(specifier, bp, 'likelihood')]))

body_parts = np.array(temp_body_parts).T #To array and transpose
part_likelihood_estimate = np.array(part_likelihood_estimate).T

#Find the valid trials to be included the criteria are the following:
#There has to be a prior choice a current choice and the stimulus is one of the easy two
valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]

#Now start settting up the regressors
individual_reg_idx = [] #Keep indices of individual regressors
reg_group_idx = [] #Keep indices of regressor groups

total_frames = []
for k in time_frame:
    total_frames.append(k[1] - k[0])
total_frames = np.sum(total_frames)

block = np.zeros([total_frames, total_frames])
for k in range(block.shape[0]):
    block[k,k] = 1

#Stack the blocks of cognitive regressors and multiply them by the respective value
time_reg = block
choice_x = block * choice[valid_trials[0]]
stimulus_x = block * stim_strengths[valid_trials[0]]
outcome_x = block * outcome[valid_trials[0]]
prior_choice_x = block * prior_choice[valid_trials[0]]
prior_outcome_x = block * prior_outcome[valid_trials[0]]
for k in range(1, valid_trials.shape[0]):
      time_reg = np.vstack((time_reg, block))
      choice_x = np.vstack((choice_x, block * choice[valid_trials[k]]))
      stimulus_x = np.vstack((stimulus_x, block * stim_strengths[valid_trials[k]]))
      outcome_x = np.vstack((outcome_x, block * outcome[valid_trials[k]]))
      prior_choice_x = np.vstack((prior_choice_x, block * prior_choice[valid_trials[k]]))
      prior_outcome_x = np.vstack((prior_outcome_x, block * prior_outcome[valid_trials[k]]))

# #With only a single column of cognitive regressors but trail-wise intercepts
# time_reg = block
# choice_x = np.ones([block.shape[0],1]) * choice[valid_trials[0]]
# stimulus_x = np.ones([block.shape[0],1])  * stim_strengths[valid_trials[0]]
# outcome_x = np.ones([block.shape[0],1])  * outcome[valid_trials[0]]
# prior_choice_x = np.ones([block.shape[0],1])  * prior_choice[valid_trials[0]]
# prior_outcome_x = np.ones([block.shape[0],1])  * prior_outcome[valid_trials[0]]
# for k in range(1, valid_trials.shape[0]):
#      time_reg = np.vstack((time_reg, block))
#      choice_x = np.vstack((choice_x, np.ones([block.shape[0],1]) * choice[valid_trials[k]]))
#      stimulus_x = np.vstack((stimulus_x, np.ones([block.shape[0],1])  * stim_strengths[valid_trials[k]]))
#      outcome_x = np.vstack((outcome_x, np.ones([block.shape[0],1])  * outcome[valid_trials[k]]))
#      prior_choice_x = np.vstack((prior_choice_x, np.ones([block.shape[0],1])  * prior_choice[valid_trials[k]]))
#      prior_outcome_x = np.vstack((prior_outcome_x, np.ones([block.shape[0],1])  * prior_outcome[valid_trials[k]]))

     
x_include = np.hstack((time_reg, choice_x, stimulus_x, outcome_x, prior_choice_x, prior_outcome_x))

#Track where the regressors and regressor groups live inside the design matrix
reg_group_idx.append(np.arange(block.shape[1],x_include.shape[1])) #Index to all the cognitive regressors at one
for k in range(int((x_include.shape[1] - block.shape[1]) / block.shape[1])):
    individual_reg_idx.append(np.arange(block.shape[1] + block.shape[1]*k, block.shape[1] + block.shape[1] * (k+1)))

regressor_labels = ['trial_time', 'choice', 'stim_strength', 'outcome', 'previous_choice', 'previous_outcome', 'pitch', 'roll', 'yaw']
regressor_size = [block.shape[0], block.shape[0], block.shape[0], block.shape[0], block.shape[0], block.shape[0], 1, 1, 1]
for k in body_part_name:
    regressor_labels = regressor_labels + [k + '_x', k + '_y']
    regressor_size = regressor_size + [2]
    
#Get the neural signals
if signal_type == 'S':
    signal = gaussian_filter1d(miniscope_data[signal_type].T,1, axis=0, mode='constant', cval=0, truncate=4)
    #Pad the first and last samples with zeros in this condition
else:
    signal = miniscope_data[signal_type].T #Transpose the original signal so that rows are timepoints and columns cells

#Determine which neurons to include in the analysis
keep_neuron = np.arange(signal.shape[1])

#Align signal to the respsective state and retrieve the data
Y = []
x_analog = []
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
        
        Y.append(signal[zero_frame + add_to,:][:, keep_neuron])
        x_analog.append(np.concatenate((head_orientation[zero_frame + add_to,:], body_parts[matching_frames,:], video_svd[matching_frames,:], me_svd[matching_frames,:]), axis=1))
        part_likelihood.append(part_likelihood_estimate[matching_frames,:])
        
        trial_timestamps_imaging.append(zero_frame + add_to)
        trial_timestamps_video.append(matching_frames)
        
#Back to array like, where columns are trials, rows time points and sheets cells   
Y = np.squeeze(Y)
x_analog = np.squeeze(x_analog)

#Reshape the arrays to match the design matrix
Y = np.reshape(Y, (x_include.shape[0], Y.shape[2]), order = 'F')
x_analog = np.reshape(x_analog, (x_include.shape[0], x_analog.shape[2]), order = 'F')

#Add the analog regressors to the cognitive regressor design matrix
X = np.hstack((x_include, x_analog))

#Update info about the regressor indices
reg_group_idx = []
reg_group_idx.append(np.arange(block.shape[1],x_include.shape[1])) #Index to all the cognitive regressors at one
reg_group_idx.append(np.arange(reg_group_idx[-1][-1]+1, reg_group_idx[-1][-1]+1+ head_orientation.shape[1]))
reg_group_idx.append(np.arange(reg_group_idx[-1][-1]+1, reg_group_idx[-1][-1]+1+ body_parts.shape[1]))
reg_group_idx.append(np.arange(reg_group_idx[-1][-1]+1, reg_group_idx[-1][-1]+1+ video_svd.shape[1] + me_svd.shape[1]))                   

#Determine which regressors are expressed as pairs
single_regressors = np.hstack((reg_group_idx[0],reg_group_idx[3])) #Cognitive and video + video me regressors come as singles
paired_regressors = np.hstack((reg_group_idx[1],reg_group_idx[2])) #Head-orientation angles and dlc labels as pairs

#Define which regressors are analog and which ones are not, use this info for z-scoring later in the code
cognitive = np.arange(x_include.shape[1])
#analog = np.arange(x_include.shape[1],x_include.shape[1] + x_analog.shape[1])
#Quick fix here, exclude head orientation angles because they are bounded between -1 and 1 (due to sin and cos)
analog = np.arange(x_include.shape[1] + 6 ,x_include.shape[1] + x_analog.shape[1]) # the + 6 are the 3 head orientation angles projected into x and y

part_likelihood = np.squeeze(part_likelihood)

#Transform to timepoint x (valid) trial matrix
trial_timestamps_imaging = np.squeeze(trial_timestamps_imaging)
trial_timestamps_video = np.squeeze(trial_timestamps_video)

#%%------------Determine which regressor to shuffle
shuffle_regressor = [None] #The full model with no shuffling

if fit_regressor_group: #Shuffle all the regressors belonging to one regressor group
    for k in reg_group_idx:
        tmp = np.arange(X.shape[1]).tolist()
        tmp = list(set(tmp) - set(k))
        shuffle_regressor.append(tmp) #The single variable group models
    
    for k in reg_group_idx:
        shuffle_regressor.append(k.tolist()) #The eliminate-one group models

else: #One regressor or a pair of coordinates are shuffled -> unique explained variance
    # shuffle_individual = np.arange(block.shape[1], x_include.shape[1] + head_orientation.shape[1]).tolist() #Exclude the time regressors from shuffling for now
    # shuffle_pairs = np.arange(shuffle_individual[-1] + 1, shuffle_individual[-1] + 1 + body_parts.shape[1], 2).tolist()
    for k in single_regressors:
        tmp = np.arange(block.shape[0], X.shape[1]).tolist()
        tmp.remove(k)
        shuffle_regressor.append(tmp) #The single variable models
    for k in paired_regressors:
        tmp = np.arange(block.shape[0], X.shape[1]).tolist()
        tmp.remove(k)
        tmp.remove(k+1)
        shuffle_regressor.append(tmp)
    
    for k in single_regressors:
        shuffle_regressor.append([k]) #The eliminate-one models
    for k in paired_regressors:
        shuffle_regressor.append([k, k+1])

#Add a set number of complete shuffles, except for the intercept term
for k in range(add_complete_shuffles):
    shuffle_regressor.append(np.arange(block.shape[0], X.shape[1]).tolist())
#%%-------Draw the training and testing splits-------------------

#First get the splits for training and testing sets. These will be constant throughout
kf = KFold(n_splits = k_folds, shuffle = True) 
k_fold_generator = kf.split(X, Y) #This returns a generator object that spits out a different split at each call
training = []
testing = []
for draw_num in range(k_folds):
    tr, te = k_fold_generator.__next__()
    training.append(tr)
    testing.append(te)

#%%---------Start the model fitting------------------------------

if not use_parallel:
    all_betas = []
    all_alphas = []
    all_rsquared = []
    all_corr = []
    
    #for shuffle in range(len(shuffle_regressor)):
    for shuffle in shuffle_regressor:    
        # start_fitting = time()
        
        # #Shuffle regressors independently for each time point
        # x_shuffle = np.array(X)
        # if shuffle_regressor[shuffle] is not None: #Only do the shuffling when required
        #     for k in shuffle_regressor[shuffle]:
        #          x_shuffle[:,k] = np.array(x_shuffle[np.random.permutation(x_shuffle.shape[0]), k]) 
        
        # betas = []
        # r_squared = []
        # corr = []
        # for fold in range(k_folds):      
            
        #     y_data = Y[training[fold],:]
        #     y_std = np.std(y_data, axis=0)
        #     y_mean = np.mean(y_data, axis=0)
            
        #     y_train = (y_data - y_mean) / y_std
        #     y_test = (Y[testing[fold],:] - y_mean) / y_std
            
        #     x_data = x_shuffle[training[fold],:]
        #     x_std_analog = np.std(x_data[:,analog], axis=0)
        #     x_mean_analog = np.mean(x_data[:,analog], axis=0)
            
        #     x_train = x_data
        #     x_train[:,analog] = (x_data[:,analog] - x_mean_analog) / x_std_analog
        #     x_test = x_shuffle[testing[fold]]
        #     x_test[:,analog] = (x_test[:,analog] - x_mean_analog) / x_std_analog
            
        #     if fold == 0: #Estiate the regularization strength on the first round
        #         alphas, tmp_betas = rb.ridge_MML(y_train, x_train, regress=True)
        #     else:
        #         tmp_betas = rb.ridge_MML(y_train, x_train, L = alphas)
        #     betas.append(tmp_betas)
            
        #     #Get the predictions from the model
        #     y_hat = np.matmul(x_test,tmp_betas) #Order matters here!
           
        #     #Calculate the coefficient of detemination
        #     ss_results = np.sum((y_test - y_hat)**2, axis = 0)
        #     ss_total = np.sum((y_test - np.mean(y_test, axis=0))**2, axis = 0)
        #     rsq = 1 - (ss_results / ss_total) 
        #     r_squared.append(rsq)
            
        #     #Also correlate pearon correlation
        #     tmp_corr = np.zeros([y_test.shape[1],1])*np.nan
        #     for q in range(y_test.shape[1]):
        #         tmp_corr[q] = np.corrcoef(y_test[:,q], y_hat[:,q])[0,1]**2 #It is the squared correlation coefficient here
        #     corr.append(tmp_corr)
        
        # #Store the averages across the folds
        # all_betas.append(np.mean(betas, axis=0))
        # all_alphas.append(alphas)
        # all_rsquared.append(np.mean(r_squared, axis=0))
        # all_corr.append(np.mean(corr, axis=0))
        
        # stop_fitting = time()
        # print(f'Finished run {shuffle} in {stop_fitting - start_fitting}')
        alphas, betas, r_squared, corr = chiCa.fit_encoding_model_shuffles(X, Y, shuffle, analog, training, testing)
        all_alphas.append(alphas)
        all_betas.append(betas)
        all_rsquared.append(r_squared)
        all_corr.append(corr)   
    
    # #Store results
    # out_dict = dict()
    # out_dict['betas'] = all_betas
    # out_dict['alphas'] = all_alphas
    # out_dict['r_squared'] = all_rsquared
    # out_dict['squred_correlation'] = all_corr
    # out_dict['regressor_labels'] = regressor_labels
    # out_dict['regressor_groups'] = reg_group_idx
    # out_dict['shuffle_regressor'] = shuffle_regressor
    # out_dict['k_fold'] = k_folds

#----
elif use_parallel(): #Run the loop in parallel
    #Do some importing that is not necessary when running on one worker
    import ipyparallel as ipp
    import os
    
    #Determine the number of engines that can be used on the local machine
    num_engines = os.cpu_count() - 1 #Spare one for the client!
    
    #Start cluster and set up a client
    cluster = ipp.Cluster(profile='myProfile', n=num_engines) #Specify something here, otherwise ipp will look for the default file
    cluster.start_cluster_sync() #Short hand to start the cluster
    rc = cluster.connect_client_sync() #Connect a client to the cluster
    dview = rc.direct_view() #Direct view to the engines?...
    rc.wait_for_engines() #When executing blocks of code sometimes the engines
    #seem to not be ready to receive a job yet.
    
    
    #Create a generator that returns itself after every iteration for x repeats.
    #This is useful as an input to map_sync below that takes lists of equal sizes
    #as arguments. With the generator one does not need to create a list with 
    #hundreds of copies of the heavy arrays X and Y.
    def regenerator(val, repeats):
        for _ in range(repeats):
            yield val
    
    #Prepare generator objects
    X_gen = regenerator(X, len(shuffle_regressor))
    Y_gen = regenerator(Y, len(shuffle_regressor))
    analog_gen = regenerator(analog, len(shuffle_regressor))
    training_gen = regenerator(training, len(shuffle_regressor))
    testing_gen = regenerator(testing, len(shuffle_regressor))
    
    print('Starting the fitting...')
    #Pre-allocate an output variable
    res = []
    res.append(dview.map_sync(chiCa.fit_encoding_model_shuffles,
                              X_gen, Y_gen, shuffle_regressor, analog_gen, training_gen, testing_gen))
    #First argument is the function to be executed and the following arguments are
    #inputs to the function
    cluster.stop_cluster() #Make sure to release the engines when the job is done!
    print('Fitting completed!')
    #Distribute the data into lists
    all_betas = []
    all_alphas = []
    all_rsquared = []
    all_corr = []
    
    for k in res[0]:
        all_alphas.append(k[0])
        all_betas.append(k[1])
        all_rsquared.append(k[2])
        all_corr.append(k[3])


#Store results
out_dict = dict()
out_dict['betas'] = all_betas
out_dict['alphas'] = all_alphas
out_dict['r_squared'] = all_rsquared
out_dict['squred_correlation'] = all_corr
out_dict['regressor_labels'] = regressor_labels
out_dict['regressor_groups'] = reg_group_idx
out_dict['shuffle_regressor'] = shuffle_regressor
out_dict['k_fold'] = k_folds
out_dict['frames_per_trial'] = block.shape[0]
       
np.save(join(session_dir,'analysis',file_name), out_dict)

