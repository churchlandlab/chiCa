# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:45:26 2022

@author: Lukas Oesch
"""

#%%--------------Do z-Scoring on the calcium time series
def standardize_signal(signal):
    '''Apply Z-scoring on the provided input signal. Make sure that cells (features)
    are rows and samples are columns.'''
    
    from scipy.stats import zscore
    #import numpy as np #Don't need to import numpy because the signal is already a numpy array object that can be transposed
    
    st_signal = zscore(signal, axis = 1) #This is where the orientation is specified!
    st_signal = st_signal.transpose() #Now flip the orientation to be able to feed it into the model
    return st_signal


#%%-------------Retrieve the time stamps for the occurence of the task state of interest
def find_state_start_frame_imaging(state_name, trialdata, average_interval, trial_start_time_covered):
    '''Locate the frame during which a certain state in the chipmunk task has
    started. The function also returns the time after state onset that was 
    covered by the frame acquisition. This may be helpful when interpreting 
    onset responses and their variability.
    
    Parameters
    ----------
    state_name: string, the state machine's state to align the events to.
    trialdata: pandas dataframe, a trial number by X dataframe, where X are different
               task states and the calium imaging indices that indicate the 
               start of the respective trial.
    average_interval: float, the mean interval in ms between imaging frames, used
                      to calculate the aligned frame indices from the trial start 
                      frame index and the state occurence timers.
    trial_start_time_covered: numpy array, vector of time within the trial that 
                              was captured within the imaging trial start frame.
    
    Retunrs
    -------
    state_start_frame: list, the frame index when the respective state started
                       in every trial.
    state_time_covered: numpy array, the time after state onset that is covered
                        by the imaging frame.
                        
    Examples
    --------
    state_start_frame, state_time_covered = find_state_start_frame_imaging(state_name, trialdata, average_interval, trial_start_time_covered)
    '''
    import numpy as np
                                                
    state_start_frame = np.zeros([len(trialdata)]) #The frame that covers the start of
    state_time_covered = np.zeros([len(trialdata)]) #The of the side that has been covered by the frame
    
    for n in range(len(trialdata)): #Subtract one here because the last trial is unfinished
          if np.isnan(trialdata[state_name][n][0]) == 0: #The state has been visited
              try:    
                  frame_time = np.arange(trial_start_time_covered[n]/1000, trialdata['FinishTrial'][n][0] - trialdata['Sync'][n][0], average_interval/1000)
                  #Generate frame times starting the first frame at the end of its coverage of trial inforamtion
              except:
                  frame_time = np.arange(trial_start_time_covered[n]/1000, trialdata['ObsInitFixation'][n][0] - trialdata['Sync'][n][0], average_interval/1000)
                  #If this is the previous implementation of chipmunk
                  
              tmp = frame_time - trialdata[state_name][n][0] #Calculate the time difference
              state_start_frame[n] = int(np.where(tmp > 0)[0][0] + trialdata["trial_start_frame_index"][n])
              #np.where returns a tuple where the first element are the indices that fulfill the condition.
              #Inside the array of indices retrieve the first one that is positive, therefore the first
              #frame that caputres some information.
         
              state_time_covered[n] =  tmp[tmp > 0][0] #Retrieve the time that was covered by the frame
          else:
              state_start_frame[n] = np.nan
              state_time_covered[n] = np.nan
          
    return state_start_frame, state_time_covered


#%%------------Track task variable in past trials
def determine_prior_variable(tracked_variable, valid_trials, consecutive_trials_back=1):
    '''Get the label of a task variable or action on a preceeding trial. This 
    function ensures that all the trials between the one we look for and the current
    one are valid, meaning there was a response.
    
    Parameters
    ----------
    tracked_variable: a numpy array of the variable to be tracked
    valid_trials: numpy array of trials with a choice
    consecutive_trials_back: numpy array that sets at what trial in the past
                             the prior variable should be assesed if all the
                             trialsup to the current have been valid.
                             
    Returns
    -------
    prior_label: numpy array of the label of the tracked variable at trial = 
                 consecutive_trials_back.
                 
    Examples
    --------
    prior_label = determine_prior_variable(tracked_variable, valid_trials, consecutive_trials_back)
    prior_label = determine_prior_variable(tracked_variable, valid_trials)
    '''
    
    import numpy as np
    
    prior_label = np.zeros([tracked_variable.shape[0]]) * np.nan
    
    for n in range(consecutive_trials_back, tracked_variable.shape[0]): #First time this can be detected is after consecutive_trials_back
            if np.isnan(tracked_variable[n - consecutive_trials_back]) == 0: #Make sure variable to be tracked is not Nan for the trial in question, might be redundant
                if np.nansum(valid_trials[n - consecutive_trials_back : n+1]) == consecutive_trials_back + 1: #Only include if all consecutive trials are valid.
                    prior_label[n] = tracked_variable[n - consecutive_trials_back]
    
    return prior_label

#%%-----------Balance the data sets for top level variable and subvariable
def balance_dataset(labels, secondary_labels = None):
    '''Balance the number of observaitions for each class in the dataset by
    subsampling from the majority classes while retaining all observations of 
    the minority class. This function offers the option to balance for two
    types of classes. This is useful when for decoding stimulus category 
    independent of animal choice as they are highly correlated in expert animals
    and therefore unbalanced. Note that all the returned indices are permuted
    within class and indices for classes are stacked on to of each other.
    
    Parameters
    ----------
    labels: numpy array, vector of the class labels of the data. Nans will be
           ignored.
    secondary_labels: numpy array, vector of additional class labels, default = None
    
    Returns
    -------
    pick_to_balance: numpy array, vector of indices for samples to be icluded
                     to obtain a balanced dataset.
    sample_num = int, the number of samples per condition. For simple balancing
                 pick_to_balance = 2*sample_num, for balancing with a secondary
                 variable pick_to_balance = 2*2*sample_num.
                     
    Examples
    --------
    pick_to_balance, sample_num = balance_dataset(labels, secondary_labels) #Balance by the two variables
    pick_to_balance, sample_num = balance_dataset(labels) #Balance for classes in label only
    '''
    
    import numpy as np
    
    classes = np.unique(labels)
    classes = classes[np.isnan(classes)==0] #Determine the classes in the labels and exclude nans
    
    if secondary_labels is None:    
        class_counts = np.array([np.sum(labels==classes[n]) for n in range(classes.shape[0])])
        #Convoluted code: sum up all the labels falling under one class for the two different classes
        sample_num = int(np.min(class_counts)) #Define how many sample per class to retain
        
        #Start assembling the indices for balancing the classes
        pick_to_balance = np.array([], dtype=int)
        for n in range(0, classes.shape[0]):
            temp_permuted = np.random.permutation(np.where(labels==classes[n])[0]) #Also permute minority class to make it more general
            pick_to_balance = np.hstack((pick_to_balance, temp_permuted[0:sample_num]))
        
              
    elif secondary_labels is not None:
        #If one needs to balance a subclass inside the labels do the same thing again
        subclasses = np.unique(secondary_labels)
        subclasses = subclasses[np.isnan(subclasses)==0] #Determine the classes in the labels and exclude nans
    
        subclass_counts = np.zeros([subclasses.shape[0], classes.shape[0]]) #Generate a matrix with subclass counts (in rows) by class labels (in columns)
        for k in range(classes.shape[0]):
            for n in range(subclasses.shape[0]):
                subclass_counts[n,k] = np.sum(secondary_labels[labels==classes[k]]==subclasses[n])
                #Within the specified class sum up all the secondary_labels that correspond to a particular subclass
                
        sample_num = int(np.min(subclass_counts))
        
        #Start assembling the indices for balancing subclasses within balanced classes
        pick_to_balance = np.array([], dtype=int)
        for k in range(0, classes.shape[0]):
            for n in range(0,subclasses.shape[0]):
                temp_permuted = np.random.permutation(np.where((labels==classes[k]) & (secondary_labels==subclasses[n]))[0])             
                pick_to_balance = np.hstack((pick_to_balance, temp_permuted[0:sample_num]))

    return pick_to_balance, sample_num

#%%--------Train cross-validated logistic regression model
def train_logistic_regression(data, labels, k_folds, model_params=None):
    '''Train regularized, corss-validated logistic regression models and perform
    a shuffled control. The corss-validation is performed in a stratified way,
    to maintain similar proportions of the input lables. As a control, the labels
    of the training samples are shuffled and a model then trained on these shuffled
    labels. The shuffled model performance is evaluated on the correct testing
    data.
    
    Parameters
    ----------
    data: numpy array, rows are observations and columns are features.
    labels: numpy array, vector of lables for the corresponding observations.
    k_folds: int, number of folds to perform cross-validation on
    model_params: dict, specifies model parameters. The keys are: penalty 
                  (the type of regularization to be applied),
                  inverse_regularization_strength, solver 
                  
    Returns
    -------
    models: pandas dataframe, results of the model fitting.
    
    Examples
    --------
    models = train_logistic_regression(data, labels, 10, model_params)
        '''

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    #First, get the model parameters
    if model_params is None:
        penalty='l1' 
        inverse_regularization_strength = 1 
        solver='liblinear'
        model_params = {'penalty': penalty, 'inverse_regularization_strength': inverse_regularization_strength, 'solver': solver}
        #Re-create the model_params to store the defaults 
        
    elif model_params is not None:
        penalty = model_params['penalty']
        inverse_regularization_strength = model_params['inverse_regularization_strength']
        solver = model_params['solver']
        # Mysteriously this loop does not work inside the function
        # for key,val in model_params.items():
        #     exec(key + '=val')
       
    #Set up the dataframe to store the training outputs 
    models = pd.DataFrame(columns=['model_accuracy', 'model_coefficients', 'model_intercept', 'model_n_iter', 
                                   'shuffle_accuracy', 'shuffle_coefficients', 'shuffle_intercept', 'shuffle_n_iter',
                                   'parameters', 'fold_number','number_of_samples'],
                          index=range(0, k_folds))
            
    skf = StratifiedKFold(n_splits = k_folds, shuffle = True) #Use stratified cross-validation to make sure 
    #that the folds are balanced themselves and that the training and validation is 
    #stable.
    skf.get_n_splits(data, labels)
    k_fold_generator = skf.split(data, labels) #This returns a generator object that spits out a different split at each call
    
    #Train the decoder
    for n in range(k_folds):
        train_index, test_index = k_fold_generator.__next__() #This is how the generator has to be called if not directly looped over
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_train_shuffled = np.random.permutation(y_train) #Shuffle the labels of the training data for the control
        #The actual model
        log_reg = LogisticRegression(penalty = penalty, C = inverse_regularization_strength, solver = solver).fit(X_train,y_train)
        #The shuffled control
        log_reg_shuffled = LogisticRegression(penalty = penalty, C = inverse_regularization_strength, solver = solver).fit(X_train,y_train_shuffled)
        
        models['model_accuracy'][n] = log_reg.score(X_test, y_test)
        models['model_coefficients'][n] = log_reg.coef_
        models['model_intercept'][n] = log_reg.intercept_[0]
        models['model_n_iter'][n] = log_reg.n_iter_[0]
        
        models['shuffle_accuracy'][n] = log_reg_shuffled.score(X_test, y_test)
        models['shuffle_coefficients'][n] = log_reg_shuffled.coef_
        models['shuffle_intercept'][n] = log_reg_shuffled.intercept_[0]
        models['shuffle_n_iter'][n] = log_reg_shuffled.n_iter_[0]

        models['parameters'][n] = model_params
        models['fold_number'][n] = n
        models['number_of_samples'][n]= X_train.shape[0]
    
    return models

#%%---Pipeline for training a set of balanced models with multiple rounds of subsampling
def balanced_logistic_model_training(data, labels, k_folds, subsampling_rounds, secondary_labels, model_params):
    '''Pipeline for training a set of logistic regression models and performing
    shuffles on data that has to be balanced. The models are fit for a set 
    amount of repetitions of the subsampling procedure used to balance the class
    labels.
    
    Parameters
    ----------
    data: numpy array, rows are observations and columns are features.
    labels: numpy array, vector of lables for the corresponding observations.
    k_folds: int, number of folds to perform cross-validation on
    subsampling_rounds: int, specifies the number of times the subsampling 
                        procedure will be repeated.
    secondary_lables: numpy array, vector of additional class labels
    model_params: dict, specifies model parameters. The keys are: penalty 
                  (the type of regularization to be applied),
                  inverse_regularization_strength, solver 
                  
    Returns
    -------
    log_reg_models: pandas dataframe, results of the model fitting.
    
    Examples
    --------
    log_reg_models = balanced_logistic_model_training(data, labels, k_folds, subsampling_rounds, secondary_labels = secondary_labels, model_params = model_params)
    log_reg_models = balanced_logistic_model_training(data, labels, k_folds, subsampling_rounds)
    
    '''
    
    import numpy as np
    import pandas as pd
    import time
    
    # #Start setting the defaults
    # secondary_labels = None 
    # model_params = None 
    
    # #Now check for these defaults in the kwargs and overwrite if necessary
    # for key,val in kwargs.items():
    #         exec(key + '=val')

    #Get a time estimate for all the fitting
    exe_start = time.time()
    
    log_reg_models = pd.DataFrame()
    
    for s in range(subsampling_rounds):
        pick_to_balance, _ = balance_dataset(labels, secondary_labels) #If the secondary label is None it will not be considered
        models = train_logistic_regression(data[pick_to_balance,:], labels[pick_to_balance], k_folds, model_params)
        
        models['subsampling_round'] = np.ones(models.shape[0]) * s #Add a column with the run of subsampling performed
        log_reg_models = pd.concat([log_reg_models, models])
    
    #Stop the time and print
    exe_stop = time.time()
    print(f'Did logistic model fits with {k_folds} folds, subsampling from majority class {subsampling_rounds} times in {round(exe_stop - exe_start)} seconds.')
    print('-------------------------------------------------------------------')
    
    
    return log_reg_models
    
    
