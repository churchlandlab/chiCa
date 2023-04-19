# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:25:23 2023

@author: Lukas Oesch
"""

import numpy as np
from numpy.matlib import repmat
import pandas as pd
import glob
import time
import itertools
import sys
sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa')
import decoding_utils
sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/analysis_sandbox')
from draft_newRidge_model import encoding_ridge_model
from time import time
from sklearn.model_selection import KFold

# session_list = ['C:/data/LO032/20220215_114758',
#                 'C:/data/LO032/20220923_135753',
#                 'C:/data/LO032/20220905_123313',
#                 'C:/data/LO032/20220907_150825',
#                 'C:/data/LO032/20220909_144008',
#                 'C:/data/LO037/20221005_162433',
#                 'C:/data/LO038/20221025_123300',
#                 'C:/data/LO028/20220209_153012',
#                 'C:/data/LO028/20220210_143922',
#                 'C:/data/LO028/20220214_172045',
#                 'C:/data/LO028/20220616_145438']   

file_name = 'dlc_regressor_encoding_models'


for session_dir in session_list:  
   
    signal_type = 'F'
    
    k_folds = 5 #For regular random cross-validation
    fit_intercept = True

    #Loading the data
    trial_alignment_file = glob.glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
    frame_rate = miniscope_data['frame_rate']
    trialdata = pd.read_hdf(glob.glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    
    dlc_data = pd.read_hdf(glob.glob(session_dir + '/dlc_analysis/*.h5')[-1]) #Load the latest extraction!
    video_alignment_files = glob.glob(session_dir + '/analysis/*video_alignment.npy')
    if len(video_alignment_files) > 1:
            print('More than one video is currently not supported')
    video_alignment = np.load(video_alignment_files[0], allow_pickle = True).tolist()
    
    #Define the range of regularization strengths to search through
    #Find the one regularization for all the neurons
    regularization_strengths = (10**(np.arange(7))).tolist()
    
    #Determine the best penalty for each neuron individually
    #regularization_strengths = [np.tile(x, (miniscope_data[signal_type].shape[0],1)) for x in (10**(np.arange(7)))]
    
    #Set the times up
    aligned_to = ['DemonInitFixation', 'PlayStimulus', 'DemonWaitForResponse', 'outcome_presentation']
    time_frame = [np.array([round(-1*frame_rate), round(0*frame_rate)+1], dtype=int), np.array([0, round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int), np.array([round(0*frame_rate), round(2*frame_rate)], dtype=int)]
    
    # aligned_to = ['PlayStimulus', 'outcome_presentation']
    # time_frame =  time_frame = [np.array([round(-0.2 * frame_rate), round(1*frame_rate)+1], dtype=int), np.array([round(-0.2*frame_rate), round(1*frame_rate)+1], dtype=int)]
    
    #Assemble the task variable design matrix
    choice = np.array(trialdata['response_side'])
    category = np.array(trialdata['correct_side'])
    prior_choice =  decoding_utils.determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
    prior_category =  decoding_utils.determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')
    
    outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
    outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
    prior_outcome =  decoding_utils.determine_prior_variable(outcome, np.ones(len(trialdata)), 1, 'consecutive')
    
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
    
    #Get head direction and standardize. 
    #TODO: Change this to include a pair of sine and cosines for the head angles to correctly express it in the regression model
    head_orientation = np.stack((miniscope_data['pitch'],miniscope_data['roll'],miniscope_data['yaw']),axis=1)
    
    #Extract a set of dlc labels and standardize these.
    dlc_keys = dlc_data.keys().tolist()
    specifier = dlc_keys[0][0] #Retrieve the session specifier that is needed as a keyword
    #TODO: Retrieve the label names automatically
    body_part_name = ['tail_tip', 'tail_center', 'rectum', 'penis', 'chest_center', 'nose_tip',
                      'hindpaw_left', 'hindpaw_right', 'frontpaw_left', 'frontpaw_right', 'ear_left', 'ear_right'] #Define the body parts to be looking at
    temp_body_parts = []
    part_likelihood_estimate = []
    for bp in body_part_name:
        for axis in ['x', 'y']:
            temp_body_parts.append(np.array(dlc_data[(specifier, bp, axis)]))
        part_likelihood_estimate.append(np.array(dlc_data[(specifier, bp, 'likelihood')]))
    temp_body_parts = np.array(temp_body_parts)
    body_parts = temp_body_parts.T
    #body_parts = decoding_utils.standardize_signal(temp_body_parts) #Standardize the body parts
    part_likelihood_estimate = np.array(part_likelihood_estimate).T
    
    #Find the valid trials to be included the criteria are the following:
    #There has to be a prior choice a current choice and the stimulus is one of the easy two
    valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]
    
    #Set up the array of cognitive regressors
    x_include = np.stack((choice[valid_trials], stim_strengths[valid_trials], outcome[valid_trials], prior_choice[valid_trials], prior_outcome[valid_trials]),axis=1)
    cognitive = np.arange(x_include.shape[1])
    analog = np.arange(x_include.shape[1], x_include.shape[1] + head_orientation.shape[1] + body_parts.shape[1])
    
    
    regressor_labels = ['choice', 'stim_strength', 'outcome', 'previous_choice', 'previous_outcome', 'pitch', 'roll', 'yaw']
    for k in body_part_name:
        regressor_labels = regressor_labels + [k + '_x', k + '_y']
        
    #Transpose the original signal so that rows are timepoints and columns cells
    signal = miniscope_data[signal_type].T

    #Determine which neurons to include in the analysis
    keep_neuron = np.arange(signal.shape[1])
    
    #Align signal to the respsective state and retrieve the data
    Y = []
    X = []
    part_likelihood = []
    for k in range(len(aligned_to)):
        state_start_frame, state_time_covered = decoding_utils.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
        zero_frame = np.array(state_start_frame[valid_trials] + time_frame[k][0], dtype=int) #The firts frame to consider
        
        for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
            matching_frames = []
            for q in range(zero_frame.shape[0]): #unfortunately need to loop through the trials, should be improved in the future...
                    tmp = decoding_utils.match_video_to_imaging(np.array([zero_frame[q] + add_to]), miniscope_data['trial_starts'][valid_trials[q]],
                           miniscope_data['frame_interval'], video_alignment['trial_starts'][valid_trials[q]], video_alignment['frame_interval'])[0].astype(int)
                    matching_frames.append(tmp)
            
            Y.append(signal[zero_frame + add_to,:][:, keep_neuron])
            X.append(np.concatenate((x_include, head_orientation[zero_frame + add_to,:], body_parts[matching_frames,:]), axis=1))
            part_likelihood.append(part_likelihood_estimate[matching_frames,:])
    #Back to array like, where columns are trials, rows time points and sheets cells   
    Y = np.squeeze(Y)
    #Similar dimensions here, with sheets being regressors
    X = np.squeeze(X)              
    
    part_likelihood = np.squeeze(part_likelihood)
    #Determine which regressor to shuffle
    shuffle_regressor = [None] #The full model with no shuffling
    
    #Only one regressor or a pair of coordinates are shuffled -> unique explained variance
    shuffle_individual = np.arange(x_include.shape[1] + head_orientation.shape[1]).tolist()
    shuffle_pairs = np.arange(len(shuffle_individual), len(shuffle_individual) + body_parts.shape[1], 2).tolist()
    for k in shuffle_individual:
        shuffle_regressor.append([k]) #The eliminate-one models
    for k in shuffle_pairs:
        shuffle_regressor.append([k, k+1])
        
    #Add the single variable models where all but one regressor are shuffled
    for k in shuffle_regressor[1:]:
        shuffle_regressor.append([x for x in np.arange(X.shape[2]).tolist() if x not in k]) #The eliminate-one models
   
    #-------------------------------------------------------------------------
    #%% --- Start the model fitting
    
    #First get the splits for training and testing sets. These will be constant throughout
    kf = KFold(n_splits = k_folds, shuffle = True) 
    k_fold_generator = kf.split(X[0], Y[0]) #This returns a generator object that spits out a different split at each call
    training = []
    testing = []
    for draw_num in range(k_folds):
        tr, te = k_fold_generator.__next__()
        training.append(tr)
        testing.append(te)
    
    #---------
    encoding_models = pd.DataFrame(index = np.arange(len(shuffle_regressor)), columns = ['models', 'regressor_labels', 'shuffle_regressor', 'training_splits', 'testing_splits',])
    
    keys_list = ['best_alpha', 'model_coefficients', 'model_intercept', 'neuron_r_squared',
                 'number_of_samples', 'shuffle_coefficients', 'shuffle_intercept',
                 'shuffle_neuron_r_squared', 'train_neuron_r_squared', 'train_shuffle_neuron_r_squared']
    timer = time()
    track_idx = 0
    for reg in shuffle_regressor: #Loop through all the regressor combinations
        output_models = []
        for time_t in range(Y.shape[0]): #Loop through all the diifferent timepoints
            
            #Shuffle regressors independently for each time point
            x_data = np.array(X[time_t, :, :])
            if reg is not None: #Only do the shuffling when required
                for k in reg:
                    # #Shuffle version 1: Disrupt trial structure for cognitive and all structure for analog regressors
                    # if k in cognitive: #Independently shuffle all the cognitive regressors
                    #     x_data[:,k] = np.array(X[time_t, np.random.permutation(X.shape[1]), k])
                    # elif k in analog: #Shuffle analog variables within and across trials!
                    #     x_data[:,k] = np.array(X[np.random.permutation(X.shape[0]), :, k][time_t, np.random.permutation(X.shape[1])])
                    # #Shuffle version 2: Equally disrupt trial structure for all regressors
                     x_data[:,k] = np.array(X[time_t, np.random.permutation(X.shape[1]), k])
                    
            for draw_num in range(k_folds):
                #Retrieve the indices for training and testing and get design matrix
                train_index = training[draw_num]
                test_index = testing[draw_num]
                x_train, x_test = x_data[train_index,:], x_data[test_index,:]
                
                #Standardize the analog regressors independently for training and testing...
                x_train[:,analog] = zscore(x_train[:,analog])
                x_test[:,analog] = zscore(x_test[:,analog])
                # x_train = zscore(x_train)
                # x_test = zscore(x_test)
                
                #Standardize the neural signals for the training and the testing data independently
                y_train = decoding_utils.standardize_signal(Y[time_t,:,:][train_index,:][:,keep_neuron].T)
                y_test = decoding_utils.standardize_signal(Y[time_t,:,:][test_index,:][:,keep_neuron].T)             
                             
                #Start the model fitting
                raw_model = decoding_utils.train_ridge_model(x_train, y_train,
                                                                 1, alpha = regularization_strengths, fit_intercept = fit_intercept)
                # raw_model = decoding_utils.train_lasso_model(x_train, y_train,
                #                                                   1, alpha = regularization_strengths, fit_intercept = fit_intercept)
                #raw_model = encoding_ridge_model(x_train, y_train, 1, alpha = regularization_strengths, fit_intercept = fit_intercept)
                
                #Assess model performance
                rsq = np.zeros(y_test.shape[1]) * np.nan
                shuffle_rsq = np.zeros(y_test.shape[1]) * np.nan
                y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
                shuffle_y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
                
                residuals = np.zeros(y_test.shape) * np.nan
                shuffle_residuals = np.zeros(y_test.shape) * np.nan
                
                #Reconstruct the signal, q are trials
                for q in range(y_test.shape[0]):
                    y_hat[:,q] = np.sum(x_test[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
                    shuffle_y_hat[:,q] = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
               
                #Asses the fit for each neuron individually
                for k in range(y_test.shape[1]):
                      rsq[k] = 1 - (np.sum((y_test[:,k] - y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
                      shuffle_rsq[k] = 1 - (np.sum((y_test[:,k] - shuffle_y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
                      
                      residuals[:,k] = y_test[:,k] - y_hat[k,:]
                      shuffle_residuals[:,k] = y_test[:,k] - shuffle_y_hat[k,:]
                      
                #For validation also reconstruct the training R2 for all the neurons
                train_rsq = np.zeros(y_train.shape[1]) * np.nan
                train_shuffle_rsq = np.zeros(y_train.shape[1]) * np.nan
                train_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
                train_shuffle_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
                
                for q in range(y_train.shape[0]):
                    train_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
                    train_shuffle_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
               
                #Asses the fit for each neuron individually
                for k in range(y_train.shape[1]):
                      train_rsq[k] = 1 - (np.sum((y_train[:,k] - train_y_hat[k,:])**2) / (np.sum((y_train[:,k] - np.mean(y_train[:,k]))**2)))
                      train_shuffle_rsq[k] = 1 - (np.sum((y_train[:,k] - train_shuffle_y_hat[k,:])**2) / (np.sum((y_train[:,k] - np.mean(y_train[:,k]))**2)))
                      
                      
                #Fill into data frame
                raw_model.insert(0, 'neuron_r_squared', [rsq])
                raw_model.insert(0,'shuffle_neuron_r_squared', [shuffle_rsq])
                
                raw_model.insert(0, 'train_neuron_r_squared', [train_rsq])
                raw_model.insert(0,'train_shuffle_neuron_r_squared', [train_shuffle_rsq])
             
                #concatenate the dataframes to achieve similar structure to other models
                if draw_num == 0:
                    model_df = raw_model
                else: 
                    model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
    
            tmp = model_df.to_dict()
            tmp_d = dict()
            for k in keys_list:
                tmp_d[k] = np.mean(model_df[k], axis=0)
            output_models.append(tmp_d)
                #Append so that the list elements represent the different timestamps
              
        encoding_models['models'][track_idx] = pd.DataFrame(output_models)
        encoding_models['shuffle_regressor'][track_idx] = reg
        encoding_models['training_splits'][track_idx] = training
        encoding_models['testing_splits'][track_idx] = testing
        
        track_idx = track_idx + 1
        
        print(f'Run {track_idx} completed, current time: {time() - timer}')


    encoding_models.to_pickle(session_dir + '/analysis/' + file_name +  f'_{k_folds}fold_cv.pkl')
    #Use pickle to store the data because Hdf seems to reach the limits of the data set size...
    
    print(f'Ran through <{session_dir}> in {time() - timer} seconds.')
  