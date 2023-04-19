# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:13:44 2022

@author: Lukas Oesch
"""

if __name__ == "__main__":
    
    import numpy as np
    import pandas as pd
    import glob
    from time import time
    import multiprocessing as mp
    import sys
    sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa')
    import decoding_utils
    import encoding_utils
    
    from sklearn.model_selection import KFold
    
    
    #Set up the desired states and time windows
    #session_dir = 'C:/data/LO032/20220215_114758'
    session_dir = 'C:/data/LO032/20220923_135753'
    #session_dir =  'C:/data/LO032/20220905_123313'
    #session_dir = 'C:/data/LO028/20220209_153012'
    #session_dir = 'C:/data/LO037/20221005_162433'
    signal_type = 'F'
    
    k_folds = 5 #For regular random cross-validation
    fit_intercept = True
    
    cv_mode = 'manual' #Set the method to use for cross validtaion
    #Can be 'auto' for sklearn built-in method score on folds or it can be 'manual'
    #to be performed on held out data inside this script or it can be 
    #'stratified' to check for trials of all kinds of combinations of task
    #variables.
    
    split_num = 10**3 #Tries to find a split, for which both, the training and 
    #the testing sets most closely match the actual average of the entire dataset
    #This is supposed to fight over-fitting the model to noise.
    
    
    regularization_strengths = [1,10,100]
    
    #regularization_strengths =  [1, 10, 100, 10**3, 10**4, 10**5, 10**6, 10**7]
    # draws = 10 #Set how many times to re-draw a new training and test set from the data
    # num_cognitive_reg = 4 #Number of "cognitive regressors"
    easiest = [4, 20] #The easiest stim strength
    
    # #Create lists with individual regressors removed to check for explained variance
    # include_regressor = [np.arange(num_cognitive_reg).tolist()] #The full model
    # for k in range(num_cognitive_reg):
    #     include_regressor.append([k]) #The single variable models
    # for k in range(num_cognitive_reg):
    #     include_regressor.append([x for x in include_regressor[0] if x != k]) #The eliminate-one models
    
    #Loading the data
    trial_alignment_file = glob.glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()   
    trialdata = pd.read_hdf(glob.glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    try:
        video_alignment_files = glob.glob(session_dir + '/analysis/*video_alignment.npy')
        if len(video_alignment_files) > 1:
            print('More than one video is currently not supported')
        video_alignment = np.load(video_alignment_files[0], allow_pickle = True).tolist()
        video_svd = np.load(glob.glob(session_dir + '/analysis/*video_SVD.npy')[0], allow_pickle = True).tolist()[1] #Only load temporal components
        me_svd = np.load(glob.glob(session_dir + '/analysis/*motion_energy_SVD.npy')[0], allow_pickle = True).tolist()[1] #Only load temporal components 
        
        #Standardize the svd components to have them vary on the same scale 
        video_svd = decoding_utils.standardize_signal(video_svd).T
        me_svd = decoding_utils.standardize_signal(me_svd).T
    except:
        print('No video svd found')
        
    frame_rate = miniscope_data['frame_rate']
    
    #Set the times up
    aligned_to = ['DemonInitFixation', 'PlayStimulus', 'DemonWaitForResponse', 'outcome_presentation']
    time_frame = [np.array([round(-1*frame_rate), round(0*frame_rate)+1], dtype=int), np.array([0, round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int), np.array([round(0*frame_rate), round(2*frame_rate)], dtype=int)]
    
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
    # #x_all = np.stack((choice, category, outcome, prior_choice, prior_category, prior_outcome),axis=1)
    x_all = np.stack((choice, outcome, prior_choice, prior_outcome),axis=1)
    # #x_all = np.stack((choice, outcome), axis=1)
    # #valid_trials = np.where(np.isnan(choice) == 0)[0]
    
    #------------
    # #Construct the regressors as combinations of conditions
    # reg_stack = np.stack((choice, outcome),axis=1)
    # reg_stack_prior = np.stack((prior_choice, prior_outcome),axis=1)
    # num_cognitive_reg = reg_stack.shape[1]
    # #Create all posible combinations of prior choices and outcomes etc.
    # combos = [np.reshape(np.array(i), (1, num_cognitive_reg)) for i in itertools.product([0, 1], repeat = num_cognitive_reg*1)]
    # x_all_current = np.zeros([trialdata.shape[0], len(combos)]) * np.nan
    # x_all_prior = np.zeros([trialdata.shape[0], len(combos)]) * np.nan
    # for k in range(len(combos)):
    #     x_all_current[:,k] = (reg_stack == combos[k][0]).all(-1)
    #     x_all_prior[:,k] = (reg_stack_prior == combos[k][0]).all(-1)
    # #Drop the case 0,0,0,0 and have it go to the intercept
    # #x_all = x_all[:,1:]
    # x_all = np.concatenate((x_all_current[:,1:], x_all_prior[:,1:]),axis=1)
        
    #--------
    # reg_stack = np.stack((choice, outcome, prior_choice, prior_outcome),axis=1)
    # num_cognitive_reg = reg_stack.shape[1]
    # combos = [np.reshape(np.array(i), (1, num_cognitive_reg)) for i in itertools.product([0, 1], repeat = num_cognitive_reg*1)]
    # x_all = np.zeros([trialdata.shape[0], len(combos)]) * np.nan
    # for k in range(len(combos)):
    #     x_all[:,k] = (reg_stack == combos[k][0]).all(-1)
    # x_all = x_all[:,1:] #Drop one of the categories to be part of the intercept
    #--------
    
    #valid_trials = np.where((np.isnan(np.sum(x_all, axis=1)) == 0) & (easiest_stim == 1))[0]
    x_include = x_all[valid_trials,:]
    #x_include = x_all #Interaction models
    
    #Standardize the data
    signal = decoding_utils.standardize_signal(miniscope_data[signal_type], scale_only = True)
    head_orientation = decoding_utils.standardize_signal(miniscope_data['head_orientation'].T).T
       
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
            
            if 'video_svd' in locals():
                matching_frames = []
                #Loop through all the trials to retrieve the respective frame index
                for q in range(zero_frame.shape[0]): #unfortunately need to loop through the trials, should be improved in the future...
                    tmp = decoding_utils.match_video_to_imaging(np.array([zero_frame[q] + add_to]), miniscope_data['trial_starts'][valid_trials[q]],
                           miniscope_data['frame_interval'], video_alignment['trial_starts'][valid_trials[q]], video_alignment['frame_interval'])[0].astype(int)
                    matching_frames.append(tmp)
                
                tmp_vid = video_svd[:,matching_frames].T
                tmp_me = me_svd[:,matching_frames].T
                X.append(np.concatenate((x_include, miniscope_data['head_orientation'][zero_frame + add_to,:], tmp_vid, tmp_me), axis=1))
            else:
                 X.append(np.concatenate((x_include, miniscope_data['head_orientation'][zero_frame + add_to,:]), axis=1))
      
                
    #Now determine which regressors to include and exclude
    #Include the full model
    include_regressor = [np.arange(X[0].shape[1])]
    #Include all the "cognitive regressors"
    for k in range(x_all.shape[1]):
        include_regressor.append([k]) #The single variable models
    #Head direction
    include_regressor.append(np.arange(x_all.shape[1], x_all.shape[1]+4))    
    #Video svd components
    include_regressor.append(np.arange(x_all.shape[1]+4, x_all.shape[1]+4 + video_svd.shape[0]))
    #Motion energy components
    include_regressor.append(np.arange(x_all.shape[1]+4 + video_svd.shape[0], x_all.shape[1]+ 4 + video_svd.shape[0] + me_svd.shape[0]))
    
    #Now loop thorugh all the single model cases and exclude the indices that were included before
    for k in include_regressor[1:]:
        include_regressor.append([x for x in include_regressor[0] if x not in k]) #The eliminate-one models
    
    include_regressor.append([0,1,2,3])
    #-------------------------------------------------------------------------
    # #%%------Make a function that parallelizes the fold generation
    # def find_best_splits(all_train_splits, all_test_splits, Yt):
    #     '''Returns the split of the provided inputs that minimizes the
    #     distance of the means of the training and testing sets to the overall
    #     mean. This helps to mitigate effects of inadequate sampling inside the 
    #     folds for cross-validation
        
    #     Parameters
    #     ----------
    #     all_train_splits: list of lists with arrays holding the sample indices for
    #                       selecting a training dataset. The dimensions are: 
    #                       number of splits -> folds -> number of indices
    #                       dataset.
    #     all_test_splits: same as above for test set.
    #     Yt: numpy array of shape trials x number of neurons holding the calcium data
    #         for the timepoint t.
            
    #     Returns
    #     -------
    #     training: list containing the best training splits for each neuron
    #     testing: same as above for test
    #     minimum_deviation: the distance of the selected training and testing set
    #                        from the overall average.
    #     '''
    
    
    #     training = [] #Initialite the splits for each time point
    #     testing = []
    #     minimum_deviation = []
        
    #     timer = time()
    #     for n in range(Yt.shape[1]):
    #         all_training = []
    #         all_testing = []
    #         abs_deviation = []
    #         for k in range(len(all_train_splits)):
                    
    #                 tmp_dev = []
    #                 tmp_train = []
    #                 tmp_test = []
    #                 for draw_num in range(len(all_train_splits[k])):
    #                     average = np.mean(Yt[:,n])
    #                     y_train, y_test = Yt[all_train_splits[k][draw_num],n], Yt[all_test_splits[k][draw_num],n]
    #                     mean_train = np.mean(y_train, axis=0)
    #                     mean_test = np.mean(y_test, axis=0)
                        
    #                     tmp_dev.append(np.mean([np.abs(average - mean_train), np.abs(average - mean_test)]))
    #                     tmp_train.append(tr)
    #                     tmp_test.append(te)
                        
    #                 all_training.append(tmp_train)
    #                 all_testing.append(tmp_test)
    #                 abs_deviation.append(tmp_dev)
            
           
    #         split_deviation = np.mean(abs_deviation)
    #         min_dev = np.min(split_deviation)
    #         split_idx = np.where(split_deviation == min_dev)[0][0]
            
    #         training.append(all_training[split_idx])
    #         testing.append(all_testing[split_idx])
    #         minimum_deviation.append(min_dev)
        
    #     print(f'Found best splits in {time() - timer} seconds')
    #     print('---')
        
    #     training, testing, minimum_deviation
        
    # #%%--------Define the function that muliprocessing is going to need to use
    # def encoding_model_individual_cell(Xt, Yt, reg, training_t, testing_t, keys_list):
    #     '''xxxx'''
        
    #     from decoding_utils import train_ridge_model 
        
    #     #Only add to the arrays that are different for individual neurons
    #     append_keys =[x for x in keys_list if not ((x == 'best_alpha') or (x == 'number_of_samples'))]  
        
    #     for n in range(Yt.shape[1]):
    #         for  draw_num in range(len(training_t[n])):
    #             train_index = training_t[n][draw_num] #Get the training indices for that once cell
    #             test_index = testing_t[n][draw_num]
                
    #             #Retrieve training and testing data from splits
    #             x_train, x_test = Xt[train_index,:][:,reg], Xt[test_index,:][:,reg]
    #             y_train, y_test = Yt[train_index,:][:,n], Yt[test_index,:][:,n]
                
    #             #For the single variable models the arrays have to be reshaped into 2d
    #             if x_train.ndim == 1:
    #                 x_train = x_train.reshape(-1,1)
    #                 x_test = x_test.reshape(-1,1)
                
    #             #Start the model fitting
    #             raw_model = train_ridge_model(x_train, y_train, 1, alpha = regularization_strengths, fit_intercept = fit_intercept)      
                
    #             #---Assess model performance
    #             y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
    #             shuffle_y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
                
    #             #Reconstruct the signal, q are trials
    #             for q in range(y_test.shape[0]):
    #                 y_hat[:,q] = np.sum(x_test[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
    #                 shuffle_y_hat[:,q] = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
               
    #             #Asses the fit for each neuron individually ---- redun
    #             rsq = 1 - (np.sum((y_test[:,0] - y_hat[0,:])**2) / (np.sum((y_test[:,0] - np.mean(y_test[:,0]))**2)))
    #             shuffle_rsq = 1 - (np.sum((y_test[:,0] - shuffle_y_hat[0,:])**2) / (np.sum((y_test[:,0] - np.mean(y_test[:,0]))**2)))
                      
    #             #For validation also reconstruct the training R2 for all the neurons
    #             train_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
    #             train_shuffle_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
                
    #             for q in range(y_train.shape[0]):
    #                 train_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
    #                 train_shuffle_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
               
    #             #Asses the fit for each neuron individually
    #             train_rsq = 1 - (np.sum((y_train[:,0] - train_y_hat[0,:])**2) / (np.sum((y_train[:,0] - np.mean(y_train[:,0]))**2)))
    #             train_shuffle_rsq = 1 - (np.sum((y_train[:,0] - train_shuffle_y_hat[0,:])**2) / (np.sum((y_train[:,0] - np.mean(y_train[:,0]))**2)))
                      
    #             #Fill into data frame
    #             raw_model.insert(0, 'neuron_r_squared', [rsq])
    #             raw_model.insert(0,'shuffle_neuron_r_squared', [shuffle_rsq])
                
    #             raw_model.insert(0, 'train_neuron_r_squared', [train_rsq])
    #             raw_model.insert(0,'train_shuffle_neuron_r_squared', [train_shuffle_rsq])
             
    #             #concatenate the dataframes to achieve similar structure to other models
    #             if draw_num == 0:
    #                 model_df = raw_model
    #             else: 
    #                 model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
                
    #         #Now the weirdness: Assing a temporary data frame to append to.
    #         if n==0:
    #             neuron_df = model_df
    #         else:
    #             for k in append_keys():
    #                 neuron_df[k] = np.stack((neuron_df[k], model_df[k]),axis=1)
        
    #     output_dict = dict()
    #     for k in keys_list:
    #         output_dict[k] = neuron_df[k]
        
    #     return output_dict
    
    # #----------------------------------------------------------------------
    #%%-----Compute a set of splits, for 100,000 this should not take longer than ~20 s
    timer = time() #Start the overall timer of the fitting
    
    all_train_splits = []
    all_test_splits = []
    kf = KFold(n_splits = k_folds, shuffle = True) #Initialize object here already and use in the following
    for k in range(split_num): #10**3 splits seems to yield good results for the splitting
            k_fold_generator = kf.split(X[0], Y[0]) #This returns a generator object that spits out a different split at each call
            tmp_tr = []
            tmp_te = []
            for draw_num in range(k_folds):
                tr, te = k_fold_generator.__next__()
                tmp_tr.append(tr)
                tmp_te.append(tr)
            all_train_splits.append(tmp_tr)
            all_test_splits.append(tmp_te)
    print(f'Made {k+1} splits in {time() - timer} seconds')
    print('---')
    
    #%%--------Select the best split for each cell at each timepoint
    
    par_pool = mp.Pool(mp.cpu_count())
    
    training = []
    testing = []
    minimum_deviation = []
    
    split_timer = time()
    tr, te, mi = par_pool.starmap(encoding_utils.find_best_splits,
                                 [(all_train_splits, all_test_splits, Yt) for Yt in Y])
    
    training.append(tr)
    testing.append(te)
    minimum_deviation.append(mi)
    print(f'Found best splits in {time() - split_timer}')

    # #%%----Do cross validation outside of the ridge model
    # if cv_mode == 'manual':
    #     from sklearn.model_selection import KFold
        
    #     if find_splits:
    #     #This is important: One issue that came up with the encoding models
    #     #was that neurons would have very negative r squared values for the
    #     #shuffle and often also for the actual model. This effect can be seen
    #     #in instances where a cross-validated model is trained on pure noise.
    #     #The reason for this is that especially the testing set would not repre
    #     #represent the mean of the full data very well, especially when testing
    #     #is performed on rather few samples. The solution implemented here to 
    #     #attenuate this effect is to generate a series of different random splits
    #     #and selecting the one split where the absolute difference of the mean 
    #     #in the folds is closest to the mean of the entire data
        
    #         timer = time() #Check how long it takes to find the best split
    #         kf = KFold(n_splits = k_folds, shuffle = True) #Initialize object here already and use in the following
            
    #         training = [] #Initialite the splits for each time point
    #         testing = []
    #         minimum_deviance = []
            
    #         for time_t in range(len(Y)):
    #             all_training = []
    #             all_testing = []
    #             abs_deviance = []
    #             for k in range(10**0): #10**3 splits seems to yield good results for the splitting
    #                     k_fold_generator = kf.split(X[0], Y[0]) #This returns a generator object that spits out a different split at each call
    #                     tmp_train = []
    #                     tmp_test = []
    #                     tmp_dev = []
    #                     for draw_num in range(k_folds):
    #                         tr, te = k_fold_generator.__next__()
    #                         y_train, y_test = Y[time_t][tr,:], Y[time_t][te,:]
    #                         mean_train = np.mean(y_train, axis=0)
    #                         mean_test = np.mean(y_test, axis=0)
                            
    #                         tmp_dev.append(np.mean(np.abs(mean_train - mean_test)))
    #                         tmp_train.append(tr)
    #                         tmp_test.append(te)
                            
    #                     all_training.append(tmp_train)
    #                     all_testing.append(tmp_test)
    #                     abs_deviance.append(tmp_dev)
                
    #             abs_deviance = np.squeeze(abs_deviance)
    #             if abs_deviance.ndim == 1:
    #                 split_deviance = np.mean(abs_deviance)
    #             else:
    #                 split_deviance = np.mean(abs_deviance, axis=1)
                
    #             min_dev = np.min(split_deviance)
    #             split_idx = np.where(split_deviance == min_dev)[0][0]
                
    #             training.append(all_training[split_idx])
    #             testing.append(all_testing[split_idx])
    #             minimum_deviance.append(min_dev)
            
    #         print(f'Found best splits in {time() - timer} seconds')
    #         print('-------------------------------------')
    #         #-----------------------------------------------------------------
    #         #Now start the model training
    #         encoding_models = pd.DataFrame(index = np.arange(len(include_regressor)),
    #                                        columns = ['models', 'include_regressor', 'training_splits', 'testing_splits', 'minimum_deviation'])
        
    #         keys_list = ['best_alpha', 'model_coefficients', 'model_intercept', 'neuron_r_squared',
    #                  'number_of_samples', 'shuffle_coefficients', 'shuffle_intercept',
    #                  'shuffle_neuron_r_squared', 'train_neuron_r_squared', 'train_shuffle_neuron_r_squared']
    #         timer = time()
    #         track_idx = 0
    #         for reg in include_regressor: #Loop through all the regressor combinations
    #             output_models = []
    #             for time_t in range(len(Y)): #Loop through all the diifferent timepoints
    #                 for draw_num in range(k_folds): #Loop throgh the folds
    #                     train_index = training[time_t][draw_num]
    #                     test_index = testing[time_t][draw_num]
                        
    #                     #Retrieve training and testing data from splits
    #                     x_train, x_test = X[time_t][train_index,:][:,reg], X[time_t][test_index,:][:,reg]
    #                     y_train, y_test = Y[time_t][train_index,:], Y[time_t][test_index,:]
                        
    #                     #For the single variable models the arrays have to be reshaped into 2d
    #                     if x_train.ndim == 1:
    #                         x_train = x_train.reshape(-1,1)
    #                         x_test = x_test.reshape(-1,1)
                        
    #                     #Start the model fitting
    #                     raw_model = decoding_utils.train_ridge_model(x_train, y_train,
    #                                                                       1, alpha = regularization_strengths, fit_intercept = fit_intercept)      
    #                     #Assess model performance
    #                     rsq = np.zeros(y_test.shape[1]) * np.nan
    #                     shuffle_rsq = np.zeros(y_test.shape[1]) * np.nan
    #                     y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
    #                     shuffle_y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
                        
    #                     residuals = np.zeros(y_test.shape) * np.nan
    #                     shuffle_residuals = np.zeros(y_test.shape) * np.nan
                        
    #                     #Reconstruct the signal, q are trials
    #                     for q in range(y_test.shape[0]):
    #                         y_hat[:,q] = np.sum(x_test[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
    #                         shuffle_y_hat[:,q] = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
                       
    #                     #Asses the fit for each neuron individually
    #                     for k in range(y_test.shape[1]):
    #                           rsq[k] = 1 - (np.sum((y_test[:,k] - y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
    #                           shuffle_rsq[k] = 1 - (np.sum((y_test[:,k] - shuffle_y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
                              
    #                           residuals[:,k] = y_test[:,k] - y_hat[k,:]
    #                           shuffle_residuals[:,k] = y_test[:,k] - shuffle_y_hat[k,:]
                              
    #                     #For validation also reconstruct the training R2 for all the neurons
    #                     train_rsq = np.zeros(y_train.shape[1]) * np.nan
    #                     train_shuffle_rsq = np.zeros(y_train.shape[1]) * np.nan
    #                     train_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
    #                     train_shuffle_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
                        
    #                     for q in range(y_train.shape[0]):
    #                         train_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
    #                         train_shuffle_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
                       
    #                     #Asses the fit for each neuron individually
    #                     for k in range(y_train.shape[1]):
    #                           train_rsq[k] = 1 - (np.sum((y_train[:,k] - train_y_hat[k,:])**2) / (np.sum((y_train[:,k] - np.mean(y_train[:,k]))**2)))
    #                           train_shuffle_rsq[k] = 1 - (np.sum((y_train[:,k] - train_shuffle_y_hat[k,:])**2) / (np.sum((y_train[:,k] - np.mean(y_train[:,k]))**2)))
                              
                              
    #                     #Fill into data frame
    #                     raw_model.insert(0, 'neuron_r_squared', [rsq])
    #                     raw_model.insert(0,'shuffle_neuron_r_squared', [shuffle_rsq])
                        
    #                     raw_model.insert(0, 'train_neuron_r_squared', [train_rsq])
    #                     raw_model.insert(0,'train_shuffle_neuron_r_squared', [train_shuffle_rsq])
                     
#                         #concatenate the dataframes to achieve similar structure to other models
#                         if draw_num == 0:
#                             model_df = raw_model
#                         else: 
#                             model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
            
#                     tmp = model_df.to_dict()
#                     tmp_d = dict()
#                     for k in keys_list:
#                         tmp_d[k] = np.mean(model_df[k], axis=0)
#                     output_models.append(tmp_d)
#                         #Append so that the list elements represent the different timestamps
                      
#                   #Append so that the list elements represent the different regressors
#                 encoding_models['models'][track_idx] = pd.DataFrame(output_models)
#                 encoding_models['include_regressor'][track_idx] = reg
#                 encoding_models['training_splits'][track_idx] = training
#                 encoding_models['testing_splits'][track_idx] = testing
#                 encoding_models['minimum_deviation'][track_idx] = minimum_deviance #Abuse of naming... 
#                 track_idx = track_idx + 1
                
#                 print(f'Run {track_idx} completed, current time: {time() - timer}')
#             encoding_models.to_pickle(session_dir + '/analysis/' + f'encoding_models_{k_folds}fold_split_optimized.pkl')
#             #Use pickle to store the data because Hdf seems to reach the limits of the data set size...
            
#             print(f'Ran through all the models in {time() - timer} seconds.')
#             print('-----------------------------------------------------------')
        
#         else:
#             #Generate the splits for all the times inside the trial and all the different models
#             kf = KFold(n_splits = k_folds, shuffle = True) #Use stratified cross-validation to make sure 
#             #kf.get_n_splits(X[0], Y[0])
#             k_fold_generator = kf.split(X[0], Y[0]) #This returns a generator object that spits out a different split at each call
#             training = []
#             testing = []
#             for draw_num in range(k_folds):
#                 tr, te = k_fold_generator.__next__()
#                 training.append(tr)
#                 testing.append(te)
            
#             #---------
#             encoding_models = pd.DataFrame(index = np.arange(len(include_regressor)), columns = ['models', 'include_regressor'])
            
#             keys_list = ['best_alpha', 'model_coefficients', 'model_intercept', 'neuron_r_squared',
#                          'number_of_samples', 'shuffle_coefficients', 'shuffle_intercept',
#                          'shuffle_neuron_r_squared', 'train_neuron_r_squared', 'train_shuffle_neuron_r_squared']
#             timer = time()
#             track_idx = 0
#             for reg in include_regressor: #Loop through all the regressor combinations
#                 output_models = []
#                 for time_t in range(len(Y)): #Loop through all the diifferent timepoints
                    
#                     # kf = KFold(n_splits = k_folds, shuffle = True) #Use stratified cross-validation to make sure 
#                     # kf.get_n_splits(X[time_t][:,reg], Y[time_t][:,reg])
#                     # k_fold_generator = kf.split(X[time_t][:,reg], Y[time_t][:,reg]) #This returns a generator object that spits out a different split at each call
#                     for draw_num in range(k_folds):
#                         # train_index, test_index = k_fold_generator.__next__() 
#                         train_index = training[draw_num]
#                         test_index = testing[draw_num]
#                         #Retrieve training and testing data from splits
#                         x_train, x_test = X[time_t][train_index,:][:,reg], X[time_t][test_index,:][:,reg]
#                         y_train, y_test = Y[time_t][train_index,:][:,keep_neuron], Y[time_t][test_index,:][:,keep_neuron]
                        
#                         #For the single variable models the arrays have to be reshaped into 2d
#                         if x_train.ndim == 1:
#                             x_train = x_train.reshape(-1,1)
#                             x_test = x_test.reshape(-1,1)
                        
#                         #Start the model fitting
#                         raw_model = decoding_utils.train_ridge_model(x_train, y_train,
#                                                                           1, alpha = [0, 1, 10, 100, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8], fit_intercept = fit_intercept)
#                         #raw_model = decoding_utils.train_linear_regression(x_train, y_train, 1, fit_intercept = fit_intercept)
                        
                        
#                         #Assess model performance
#                         rsq = np.zeros(y_test.shape[1]) * np.nan
#                         shuffle_rsq = np.zeros(y_test.shape[1]) * np.nan
#                         y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
#                         shuffle_y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
                        
#                         residuals = np.zeros(y_test.shape) * np.nan
#                         shuffle_residuals = np.zeros(y_test.shape) * np.nan
                        
#                         #Reconstruct the signal, q are trials
#                         for q in range(y_test.shape[0]):
#                             y_hat[:,q] = np.sum(x_test[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
#                             shuffle_y_hat[:,q] = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
                       
#                         #Asses the fit for each neuron individually
#                         for k in range(y_test.shape[1]):
#                               rsq[k] = 1 - (np.sum((y_test[:,k] - y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
#                               shuffle_rsq[k] = 1 - (np.sum((y_test[:,k] - shuffle_y_hat[k,:])**2) / (np.sum((y_test[:,k] - np.mean(y_test[:,k]))**2)))
                              
#                               residuals[:,k] = y_test[:,k] - y_hat[k,:]
#                               shuffle_residuals[:,k] = y_test[:,k] - shuffle_y_hat[k,:]
                              
#                         #For validation also reconstruct the training R2 for all the neurons
#                         train_rsq = np.zeros(y_train.shape[1]) * np.nan
#                         train_shuffle_rsq = np.zeros(y_train.shape[1]) * np.nan
#                         train_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
#                         train_shuffle_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
                        
#                         for q in range(y_train.shape[0]):
#                             train_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
#                             train_shuffle_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
                       
#                         #Asses the fit for each neuron individually
#                         for k in range(y_train.shape[1]):
#                               train_rsq[k] = 1 - (np.sum((y_train[:,k] - train_y_hat[k,:])**2) / (np.sum((y_train[:,k] - np.mean(y_train[:,k]))**2)))
#                               train_shuffle_rsq[k] = 1 - (np.sum((y_train[:,k] - train_shuffle_y_hat[k,:])**2) / (np.sum((y_train[:,k] - np.mean(y_train[:,k]))**2)))
                              
                              
#                         #Fill into data frame
#                         raw_model.insert(0, 'neuron_r_squared', [rsq])
#                         raw_model.insert(0,'shuffle_neuron_r_squared', [shuffle_rsq])
                        
#                         raw_model.insert(0, 'train_neuron_r_squared', [train_rsq])
#                         raw_model.insert(0,'train_shuffle_neuron_r_squared', [train_shuffle_rsq])
                     
#                         #concatenate the dataframes to achieve similar structure to other models
#                         if draw_num == 0:
#                             model_df = raw_model
#                         else: 
#                             model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
            
#                     tmp = model_df.to_dict()
#                     tmp_d = dict()
#                     for k in keys_list:
#                         tmp_d[k] = np.mean(model_df[k], axis=0)
#                     output_models.append(tmp_d)
#                         #Append so that the list elements represent the different timestamps
                      
#                 # encoding_models.append(output_models)
#                   #Append so that the list elements represent the different regressors
#                 encoding_models['models'][track_idx] = pd.DataFrame(output_models)
#                 encoding_models['include_regressor'][track_idx] = reg
#                 track_idx = track_idx + 1
                
#         # np.save(session_dir + '/analysis/' + f'encoding_models_{k_folds}fold_cv.npy', encoding_models)
#         encoding_models.to_pickle(session_dir + '/analysis/' + f'encoding_models_{k_folds}fold_cv.pkl')
#         #Use pickle to store the data because Hdf seems to reach the limits of the data set size...
        
#         print(f'Ran through all the models in {time() - timer} seconds.')
       
# #     #%%---To fit models with the stratified method
# #     elif cv_mode == 'stratified':
# #         k_folds = 1
# #         #Generate all the possible combinations of task variables
# #         combos = [np.reshape(np.array(i), (1, num_cognitive_reg)) for i in itertools.product([0, 1], repeat = num_cognitive_reg*1)]
        
# #         #Find the respective combination of task variables inside the design matrix
# #         combo_index = []
# #         observation_num = []
# #         for pattern in combos:
# #             if np.sum((X[0][:,:num_cognitive_reg] == pattern[0]).all(-1)) > 0: #Only include actually observed combinations
# #                 combo_index.append(np.where((X[0][:,:num_cognitive_reg] == pattern[0]).all(-1))[0]) 
# #                 observation_num.append(combo_index[-1].shape[0]) #Store the number of trials with this combination of task variables      
        
# #         #Loop through the number of splits
# #         temp_idx = np.arange(valid_trials.shape[0]).tolist() #These are increasing number until the amout of valid trials
# #         train_indices_list = []
# #         test_indices_list = []
# #         for n in range(draws):
# #             test_idx = []
# #             for k in range(len(combo_index)):
# #                 if combo_index[k].shape[0] > 1: #Require more than one observation for testing
# #                     test_idx.append(combo_index[k][np.random.permutation(combo_index[k].shape[0])[0]])
# #                 else:
# #                     print('Detected only a single trial with a particular task variable mix.')
# #                     print('This trial will not be represented in the test data.')
# #             #Remove all the test trials from the original list
# #             train_idx = [i for i in temp_idx if i not in test_idx]
#             train_indices_list.append(train_idx)
#             test_indices_list.append(test_idx)
        
#         encoding_models = [] #Initialize a list whose elements are data frames
#         #of models with different included regressors. The dataframes are the
#         #fits for all the draws
#         for reg in include_regressor: #Loop through all the regressor combinations
#             output_models = []
#             for time_t in range(len(Y)): #Loop through all the diifferent timepoints
#                 for draw_num in range(len(train_indices_list)):
#                     #Retrieve the training data
#                     x_train = X[time_t][train_indices_list[draw_num],:][:,reg] #Step-wise indexing here
#                     y_train = Y[time_t][train_indices_list[draw_num],:]
#                     #Start the model fitting
#                     raw_model = decoding_utils.train_ridge_model(x_train, y_train,
#                                                                  k_folds, alpha = None, fit_intercept = fit_intercept)
                   
                    
#                     #Asses model performance
#                     x_test = X[time_t][test_indices_list[draw_num],:][:,reg]
#                     y_test = Y[time_t][test_indices_list[draw_num],:]                   
#                     rsq = np.zeros(len(test_indices_list[draw_num])) * np.nan
#                     shuffle_rsq = np.zeros(len(test_indices_list[draw_num])) * np.nan
                    
#                     for q in range(len(test_indices_list[draw_num])):
#                         rec = np.sum(x_test[q,:] * raw_model['model_coefficients'][0],axis=1) + raw_model['model_intercept'][0] #multiply the weights with the variables
#                         rsq[q] = np.corrcoef(rec, y_test[q,:])[0,1]**2
                   
#                         rec = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0],axis=1) + raw_model['shuffle_intercept'][0]
#                         shuffle_rsq[q] = np.corrcoef(rec, y_test[q,:])[0,1]**2
                    
#                     #Fill into data frame
#                     raw_model['model_r_squared'] = np.mean(rsq)
#                     raw_model['shuffle_r_squared'] = np.mean(shuffle_rsq)
                 
#                     #concatenate the dataframes to achieve similar structure to other models
#                     if draw_num == 0:
#                         model_df = raw_model
#                     else: 
#                         model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
                 
#                 output_models.append(model_df)
#                 #Append so that the list elements represent the different timestamps
                
#             encoding_models.append(output_models)
#             #Append so that the list elements represent the different regressors
            
#     #-----------------------------------------------------------------
#     #%%----In case regular cross-validation is desired
#     elif cv_mode == 'auto':
#         encoding_models = [] #Initialize a list whose elements are data frames
#         #of models with different included regressors. The dataframes are the
#         #fits for all the draws
#         for reg in include_regressor: #Loop through all the regressor combinations
#             output_models = []    
#             for time_t in range(len(Y)): #Loop through all the diifferent timepoints
#                #Retrieve the training data
#                     x_train = X[time_t][:,reg] 
#                     y_train = Y[time_t][:,reg] 
#                     #Start the model fitting
#                     output_models.append(decoding_utils.train_ridge_model(x_train, y_train,
#                                                                  k_folds, alpha = None, fit_intercept = fit_intercept))
            
#             encoding_models.append(output_models)
                
        
        
        
        
        
                 
# #                 #Initialite reconstruction array of dimensions: time point x trial x cell number
# #                 reconstructed_y = np.zeros([len(output_models), test_indices_list[draw_num], test_indices_list[draw_num]]) * np.nan
# #                 #Convert the test data to a similar array format as the reconstructed one           
# #                 test_traces_array = np.zeros(reconstructed_y.shape) * np.nan
                
# #                 #Initialize array for reconstruction error for: time_points x trial_type
# #                 cv_R_squared = np.zeros([len(output_models), test_indices_list[draw_num]]) * np.nan
# #                 shuffle_cv_R_squared = np.zeros([len(output_models), test_indices_list[draw_num]]) * np.nan
# #                 for n in range(len(output_models)): # -> n tracks the time points
# #                     for k in range(x_test.shape[0]): # -> k tracks the trials for testing
# #                         #First reconstruct the cell activity vector from the coefficients
# #                         #and the held out trials.
# #                         #Do this by multiplying the coefficients with the task variables for that trial 
# #                         rec = np.sum(X[time_t][train_indices_list[draw_num],reg] * output_models[n]['model_coefficients'][0],axis=1) + output_models[n]['model_intercept'][0] #multiply the weights with the variables
# #                         rec_s = np.sum(x_test[k] * output_models[n]['model_coefficients'][0],axis=1) + output_models[n]['model_intercept'][0] #multiply the weights with the variables
                        
# #                         #Calculate the R squared of the model by computing the squared correlation coefficient between the vectors
# #                         rsq = (np.corrcoef(rec, y_test_list[n][k])[1,0])**2
                        
# #                         #Store the values
# #                         reconstructed_y[n,k,:] = rec
# #                         R_squared[n,k] = rsq
# #                         test_traces_array[n,k,:] = y_test_list[n][k]
                
                
                
                
                
                
                
        
# #     else:
# #         train_idx = np.arange(valid_trials.shape[0]).tolist() #Take all the trials if conventional n-fold cross-validation is needed
# #         test_idx = None
    
# #     #Align signal to the respsective state and retrieve the data
# #     y_train_list = []
# #     y_test_list = []
# #     for k in range(len(aligned_to)):
# #         state_start_frame, state_time_covered = decoding_utils.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
# #         zero_frame = np.array(state_start_frame[valid_trials] + time_frame[k][0], dtype=int) #The firts frame to consider
# #         for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
# #             y_train_list.append(signal[zero_frame[train_idx] + add_to,:])
                                          
# #             if test_idx is not None:
# #                 y_test_list.append(signal[zero_frame[test_idx] + add_to,:])
                
# #     x_train = X[train_idx,:]
# #     x_test = X[test_idx,:]
# #      #%%Start the model fitting
    
# #     # start_parallel = time.time() #Measure the time
# #     # par_pool = mp.Pool(mp.cpu_count())
# #     # output_models = par_pool.starmap(decoding_utils.train_linear_regression,
# #     #                              [(X, data , k_folds, fit_intercept) for data in data_list])
    
# #     # par_pool.close()
# #     # stop_parallel = time.time()
# #     # print('-------------------------------------------')
# #     # print(f'Done fitting the models in {round(stop_parallel - start_parallel)} seconds')        
# #     output_models = []
# #     for y_train in y_train_list:
# #         #output_models.append(decoding_utils.train_linear_regression(x_train, y_train, k_folds, fit_intercept))
# #         output_models.append(decoding_utils.train_ridge_model(x_train, y_train, k_folds, alpha = None, fit_intercept = fit_intercept))
        
        
# # #%%---Evaluate the model performance on held-out data if specified

# #     if stratified_hold_one_out:
# #         #Initialite reconstruction array of dimensions: time point x trial x cell number
# #         reconstructed_y = np.zeros([len(output_models), y_test_list[0].shape[0], y_test_list[0].shape[1]]) * np.nan
# #         #Convert the test data to a similar array format as the reconstructed one           
# #         test_traces_array = np.zeros(reconstructed_y.shape) * np.nan
        
# #         #Initialize array for reconstruction error for: time_points x trial_type
# #         R_squared = np.zeros([len(output_models), y_test_list[0].shape[0]]) * np.nan
# #         for n in range(len(output_models)): # -> n tracks the time points
# #             for k in range(x_test.shape[0]): # -> k tracks the trials for testing
# #                 #First reconstruct the cell activity vector from the coefficients
# #                 #and the held out trials.
# #                 #Do this by multiplying the coefficients with the task variables for that trial 
# #                 rec = np.sum(x_test[k] * output_models[n]['model_coefficients'][0],axis=1) + output_models[n]['model_intercept'][0] #multiply the weights with the variables
                
# #                 #Calculate the R squared of the model by computing the squared correlation coefficient between the vectors
# #                 rsq = (np.corrcoef(rec, y_test_list[n][k])[1,0])**2
                
# #                 #Store the values
# #                 reconstructed_y[n,k,:] = rec
# #                 R_squared[n,k] = rsq
# #                 test_traces_array[n,k,:] = y_test_list[n][k]
       
        