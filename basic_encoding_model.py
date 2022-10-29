# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:41:14 2022

@author: Lukas Oesch
"""

if __name__ == '__main__':
    
    import numpy as np
    from numpy.matlib import repmat
    import pandas as pd
    import glob
    import time
    import multiprocessing as mp
    import itertools
    import sys
    sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa')
    import decoding_utils
    
    #Set up the desired states and time windows
    #session_dir = 'C:/data/LO032/20220215_114758'
    #session_dir = 'C:/data/LO032/20220923_135753'
    #session_dir = 'C:/data/LO037/20221005_162433'
    signal_type = 'F'
    
    k_folds = 3 #For regular random cross-validation
    fit_intercept = True
    
    cv_mode = 'manual' #Set the method to use for cross validtaion
    #Can be 'auto' for sklearn built-in method score on folds or it can be 'manual'
    #to be performed on held out data inside this script or it can be 
    #'stratified' to check for trials of all kinds of combinations of task
    #variables.
    
    draws = 10 #Set how many times to re-draw a new training and test set from the data
    num_cognitive_reg = 4 #Number of "cognitive regressors"
    easiest = [4, 20] #The easiest stim strength
    
    #Create lists with individual regressors removed to check for explained variance
    include_regressor = [np.arange(num_cognitive_reg).tolist()] #The full model
    for k in range(num_cognitive_reg):
        include_regressor.append([k]) #The single variable models
    for k in range(num_cognitive_reg):
        include_regressor.append([x for x in include_regressor[0] if x != k]) #The eliminate-one models

    #Loading the data
    trial_alignment_file = glob.glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()   
    trialdata = pd.read_hdf(glob.glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    try:
        video_svd = np.load(glob.glob(session_dir + '/chipmunk/*video_SVD.npy')[0], allow_pickle = True).tolist()[1] #Only load temporal components
        me_svd = np.load(glob.glob(session_dir + '/chipmunk/*motion_energy_SVD.npy')[0], allow_pickle = True).tolist()[1] #Only load temporal components
    except:
        print('No video svd found')
        
    frame_rate = miniscope_data['frame_rate']
    
    #Set the times up
    aligned_to = ['DemonInitFixation', 'PlayStimulus', 'DemonWaitForResponse', 'outcome_presentation']
    time_frame = [np.array([round(-1*frame_rate), round(0*frame_rate)+1], dtype=int), np.array([0, round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.4*frame_rate)+1], dtype=int), np.array([round(0*frame_rate), round(2*frame_rate)], dtype=int)]
    
    #Assemble the task variable design matrix
    choice = np.array(trialdata['response_side'])
    category = np.array(trialdata['correct_side'])
    prior_choice =  decoding_utils.determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
    prior_category =  decoding_utils.determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')
    
    outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
    outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
    prior_outcome =  decoding_utils.determine_prior_variable(outcome, np.ones(len(trialdata)), 1, 'consecutive')
    
    #Find the trials that can be included into the analysis and clean up task variable matrix
    easiest_stim = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
    for k in range(trialdata.shape[0]):
        if trialdata['stimulus_event_timestamps'][k].shape[0] == easiest[0] or trialdata['stimulus_event_timestamps'][k].shape[0] == easiest[1]:
            easiest_stim[k] = 1          
    
    #Find the valid trials to be included the criteria are the following:
        #There has to be a prior choice a current choice and the stimulus is one of the easy two
    valid_trials = np.where(((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0)) & easiest_stim)[0]
    
    
    #Initialize the task variable design matrix
    #x_all = np.stack((choice, category, outcome, prior_choice, prior_category, prior_outcome),axis=1)
    x_all = np.stack((choice, outcome, prior_choice, prior_outcome),axis=1)
    #x_all = np.stack((choice, outcome), axis=1)
    #valid_trials = np.where(np.isnan(choice) == 0)[0]
    #x_all = np.stack(((choice[valid_trials]==0) & (outcome[valid_trials]==1), (choice[valid_trials]==0) & (outcome[valid_trials]==0),
                      (choice[valid_trials]==1) & (outcome[valid_trials]==1), (choice[valid_trials]==1) & (outcome[valid_trials]==0))).T
    
    
    #Find the trials that can be included into the analysis and clean up task variable matrix
    easiest_stim = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
    for k in range(trialdata.shape[0]):
        if trialdata['stimulus_event_timestamps'][k].shape[0] == easiest[0] or trialdata['stimulus_event_timestamps'][k].shape[0] == easiest[1]:
            easiest_stim[k] = 1
            
    #valid_trials = np.where((np.isnan(np.sum(x_all, axis=1)) == 0) & (easiest_stim == 1))[0]
    x_include = x_all[valid_trials,:]
    #x_include = x_all #Interaction models
    
    #Standardize the data
    signal = decoding_utils.standardize_signal(miniscope_data[signal_type])
    
    #Determine which neurons to include in the analysis
    keep_neuron = np.arange(signal.shape[1])
    
    #Align signal to the respsective state and retrieve the data
    Y = []
    X = []
    for k in range(len(aligned_to)):
        state_start_frame, state_time_covered = decoding_utils.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
        zero_frame = np.array(state_start_frame[valid_trials] + time_frame[k][0], dtype=int) #The firts frame to consider
        for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
            Y.append(signal[zero_frame + add_to,:][:, keep_neuron])
            X.append(x_include)
   
    #-------------------------------------------------------------------------                       
    #%%---To fit models with the stratified method
    if cv_mode == 'stratified':
        k_folds = 1
        #Generate all the possible combinations of task variables
        combos = [np.reshape(np.array(i), (1, num_cognitive_reg)) for i in itertools.product([0, 1], repeat = num_cognitive_reg*1)]
        
        #Find the respective combination of task variables inside the design matrix
        combo_index = []
        observation_num = []
        for pattern in combos:
            if np.sum((X[0][:,:num_cognitive_reg] == pattern[0]).all(-1)) > 0: #Only include actually observed combinations
                combo_index.append(np.where((X[0][:,:num_cognitive_reg] == pattern[0]).all(-1))[0]) 
                observation_num.append(combo_index[-1].shape[0]) #Store the number of trials with this combination of task variables      
        
        #Loop through the number of splits
        temp_idx = np.arange(valid_trials.shape[0]).tolist() #These are increasing number until the amout of valid trials
        train_indices_list = []
        test_indices_list = []
        for n in range(draws):
            test_idx = []
            for k in range(len(combo_index)):
                if combo_index[k].shape[0] > 1: #Require more than one observation for testing
                    test_idx.append(combo_index[k][np.random.permutation(combo_index[k].shape[0])[0]])
                else:
                    print('Detected only a single trial with a particular task variable mix.')
                    print('This trial will not be represented in the test data.')
            #Remove all the test trials from the original list
            train_idx = [i for i in temp_idx if i not in test_idx]
            train_indices_list.append(train_idx)
            test_indices_list.append(test_idx)
        
        encoding_models = [] #Initialize a list whose elements are data frames
        #of models with different included regressors. The dataframes are the
        #fits for all the draws
        for reg in include_regressor: #Loop through all the regressor combinations
            output_models = []
            for time_t in range(len(Y)): #Loop through all the diifferent timepoints
                for draw_num in range(len(train_indices_list)):
                    #Retrieve the training data
                    x_train = X[time_t][train_indices_list[draw_num],:][:,reg] #Step-wise indexing here
                    y_train = Y[time_t][train_indices_list[draw_num],:]
                    #Start the model fitting
                    raw_model = decoding_utils.train_ridge_model(x_train, y_train,
                                                                 k_folds, alpha = None, fit_intercept = fit_intercept)
                   
                    
                    #Asses model performance
                    x_test = X[time_t][test_indices_list[draw_num],:][:,reg]
                    y_test = Y[time_t][test_indices_list[draw_num],:]                   
                    rsq = np.zeros(len(test_indices_list[draw_num])) * np.nan
                    shuffle_rsq = np.zeros(len(test_indices_list[draw_num])) * np.nan
                    
                    for q in range(len(test_indices_list[draw_num])):
                        rec = np.sum(x_test[q,:] * raw_model['model_coefficients'][0],axis=1) + raw_model['model_intercept'][0] #multiply the weights with the variables
                        rsq[q] = np.corrcoef(rec, y_test[q,:])[0,1]**2
                   
                        rec = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0],axis=1) + raw_model['shuffle_intercept'][0]
                        shuffle_rsq[q] = np.corrcoef(rec, y_test[q,:])[0,1]**2
                    
                    #Fill into data frame
                    raw_model['model_r_squared'] = np.mean(rsq)
                    raw_model['shuffle_r_squared'] = np.mean(shuffle_rsq)
                 
                    #concatenate the dataframes to achieve similar structure to other models
                    if draw_num == 0:
                        model_df = raw_model
                    else: 
                        model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
                 
                output_models.append(model_df)
                #Append so that the list elements represent the different timestamps
                
            encoding_models.append(output_models)
            #Append so that the list elements represent the different regressors
            
    #-----------------------------------------------------------------
    #%%----Do cross validation outside of the ridge model
    elif cv_mode == 'manual':
        from sklearn.model_selection import KFold
        
        #Generate the splits for all the times inside the trial and all the different models
        kf = KFold(n_splits = k_folds, shuffle = True) #Use stratified cross-validation to make sure 
        kf.get_n_splits(X[0], Y[0])
        k_fold_generator = kf.split(X[0], Y[0]) #This returns a generator object that spits out a different split at each call
        training = []
        testing = []
        for draw_num in range(k_folds):
            tr, te = k_fold_generator.__next__()
            training.append(tr)
            testing.append(te)
        
        encoding_models = []
        
        for reg in include_regressor: #Loop through all the regressor combinations
            output_models = []
            for time_t in range(len(Y)): #Loop through all the diifferent timepoints
                
                # kf = KFold(n_splits = k_folds, shuffle = True) #Use stratified cross-validation to make sure 
                # kf.get_n_splits(X[time_t][:,reg], Y[time_t][:,reg])
                # k_fold_generator = kf.split(X[time_t][:,reg], Y[time_t][:,reg]) #This returns a generator object that spits out a different split at each call
                for draw_num in range(k_folds):
                    # train_index, test_index = k_fold_generator.__next__() 
                    train_index = training[draw_num]
                    test_index = testing[draw_num]
                    #Retrieve training and testing data from splits
                    x_train, x_test = X[time_t][train_index,:][:,reg], X[time_t][test_index,:][:,reg]
                    y_train, y_test = Y[time_t][train_index,:], Y[time_t][test_index,:]
                    
                    #For the single variable models the arrays have to be reshaped into 2d
                    if x_train.ndim == 1:
                        x_train = x_train.reshape(-1,1)
                        x_test = x_test.reshape(-1,1)
                    
                    #Start the model fitting
                    raw_model = decoding_utils.train_ridge_model(x_train, y_train,
                                                                   1, alpha = None, fit_intercept = fit_intercept)
                    
                    #Assess model performance
                    rsq = np.zeros(y_test.shape[1]) * np.nan
                    shuffle_rsq = np.zeros(y_test.shape[1]) * np.nan
                    y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
                    shuffle_y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
                    
                    #Reconstruct the signal, q are trials
                    for q in range(y_test.shape[0]):
                        y_hat[:,q] = np.sum(x_test[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
                        shuffle_y_hat[:,q] = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
                   
                    #Asses the fit for each neuron individually
                    for k in range(y_test.shape[1]):
                          rsq[k] = 1 - (np.sum((y_test[:,k] - y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
                          shuffle_rsq[k] = 1 - (np.sum((y_test[:,k] - shuffle_y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
                  
                    #Fill into data frame
                    raw_model.insert(0, 'neuron_r_squared', [rsq])
                    raw_model.insert(0,'shuffle_neuron_r_squared', [shuffle_rsq])
                 
                    #concatenate the dataframes to achieve similar structure to other models
                    if draw_num == 0:
                        model_df = raw_model
                    else: 
                        model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
      
                output_models.append(model_df)
                    #Append so that the list elements represent the different timestamps
                  
            encoding_models.append(output_models)
              #Append so that the list elements represent the different regressors
              

    #%%----In case regular cross-validation is desired
    elif cv_mode == 'auto':
        encoding_models = [] #Initialize a list whose elements are data frames
        #of models with different included regressors. The dataframes are the
        #fits for all the draws
        for reg in include_regressor: #Loop through all the regressor combinations
            output_models = []    
            for time_t in range(len(Y)): #Loop through all the diifferent timepoints
               #Retrieve the training data
                    x_train = X[time_t][:,reg] 
                    y_train = Y[time_t][:,reg] 
                    #Start the model fitting
                    output_models.append(decoding_utils.train_ridge_model(x_train, y_train,
                                                                 k_folds, alpha = None, fit_intercept = fit_intercept))
            
            encoding_models.append(output_models)
                
        
        
        
        
        
                 
#                 #Initialite reconstruction array of dimensions: time point x trial x cell number
#                 reconstructed_y = np.zeros([len(output_models), test_indices_list[draw_num], test_indices_list[draw_num]]) * np.nan
#                 #Convert the test data to a similar array format as the reconstructed one           
#                 test_traces_array = np.zeros(reconstructed_y.shape) * np.nan
                
#                 #Initialize array for reconstruction error for: time_points x trial_type
#                 cv_R_squared = np.zeros([len(output_models), test_indices_list[draw_num]]) * np.nan
#                 shuffle_cv_R_squared = np.zeros([len(output_models), test_indices_list[draw_num]]) * np.nan
#                 for n in range(len(output_models)): # -> n tracks the time points
#                     for k in range(x_test.shape[0]): # -> k tracks the trials for testing
#                         #First reconstruct the cell activity vector from the coefficients
#                         #and the held out trials.
#                         #Do this by multiplying the coefficients with the task variables for that trial 
#                         rec = np.sum(X[time_t][train_indices_list[draw_num],reg] * output_models[n]['model_coefficients'][0],axis=1) + output_models[n]['model_intercept'][0] #multiply the weights with the variables
#                         rec_s = np.sum(x_test[k] * output_models[n]['model_coefficients'][0],axis=1) + output_models[n]['model_intercept'][0] #multiply the weights with the variables
                        
#                         #Calculate the R squared of the model by computing the squared correlation coefficient between the vectors
#                         rsq = (np.corrcoef(rec, y_test_list[n][k])[1,0])**2
                        
#                         #Store the values
#                         reconstructed_y[n,k,:] = rec
#                         R_squared[n,k] = rsq
#                         test_traces_array[n,k,:] = y_test_list[n][k]
                
                
                
                
                
                
                
        
#     else:
#         train_idx = np.arange(valid_trials.shape[0]).tolist() #Take all the trials if conventional n-fold cross-validation is needed
#         test_idx = None
    
#     #Align signal to the respsective state and retrieve the data
#     y_train_list = []
#     y_test_list = []
#     for k in range(len(aligned_to)):
#         state_start_frame, state_time_covered = decoding_utils.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
#         zero_frame = np.array(state_start_frame[valid_trials] + time_frame[k][0], dtype=int) #The firts frame to consider
#         for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
#             y_train_list.append(signal[zero_frame[train_idx] + add_to,:])
                                          
#             if test_idx is not None:
#                 y_test_list.append(signal[zero_frame[test_idx] + add_to,:])
                
#     x_train = X[train_idx,:]
#     x_test = X[test_idx,:]
#      #%%Start the model fitting
    
#     # start_parallel = time.time() #Measure the time
#     # par_pool = mp.Pool(mp.cpu_count())
#     # output_models = par_pool.starmap(decoding_utils.train_linear_regression,
#     #                              [(X, data , k_folds, fit_intercept) for data in data_list])
    
#     # par_pool.close()
#     # stop_parallel = time.time()
#     # print('-------------------------------------------')
#     # print(f'Done fitting the models in {round(stop_parallel - start_parallel)} seconds')        
#     output_models = []
#     for y_train in y_train_list:
#         #output_models.append(decoding_utils.train_linear_regression(x_train, y_train, k_folds, fit_intercept))
#         output_models.append(decoding_utils.train_ridge_model(x_train, y_train, k_folds, alpha = None, fit_intercept = fit_intercept))
        
        
# #%%---Evaluate the model performance on held-out data if specified

#     if stratified_hold_one_out:
#         #Initialite reconstruction array of dimensions: time point x trial x cell number
#         reconstructed_y = np.zeros([len(output_models), y_test_list[0].shape[0], y_test_list[0].shape[1]]) * np.nan
#         #Convert the test data to a similar array format as the reconstructed one           
#         test_traces_array = np.zeros(reconstructed_y.shape) * np.nan
        
#         #Initialize array for reconstruction error for: time_points x trial_type
#         R_squared = np.zeros([len(output_models), y_test_list[0].shape[0]]) * np.nan
#         for n in range(len(output_models)): # -> n tracks the time points
#             for k in range(x_test.shape[0]): # -> k tracks the trials for testing
#                 #First reconstruct the cell activity vector from the coefficients
#                 #and the held out trials.
#                 #Do this by multiplying the coefficients with the task variables for that trial 
#                 rec = np.sum(x_test[k] * output_models[n]['model_coefficients'][0],axis=1) + output_models[n]['model_intercept'][0] #multiply the weights with the variables
                
#                 #Calculate the R squared of the model by computing the squared correlation coefficient between the vectors
#                 rsq = (np.corrcoef(rec, y_test_list[n][k])[1,0])**2
                
#                 #Store the values
#                 reconstructed_y[n,k,:] = rec
#                 R_squared[n,k] = rsq
#                 test_traces_array[n,k,:] = y_test_list[n][k]
       
        