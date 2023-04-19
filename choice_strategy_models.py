# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:47:08 2022

@author: Lukas Oesch
"""
def dPrime_2AFC(correct_side, response_side):
    '''Calculate the d prime value for a 2AFC task.
    
    Parameters
    ----------
    correct_side: numpy array, vector of the correct trial sides
    response_side: numpy array, vector of the chosen side by the animal, nan is handled.
    
    Returns
    -------
    dPrime: float, the d prime value
    side_code: float, the side that the hit_rate and the false_alarm rate are calculated on
    hit_rate: float, the rate of correctly choosing the specified side when it was actually correct
    false_alarm_rate: float, the rate of incorrectly choosing the specified side when the other side was correct
    
    Usage
    -----
    dPrime, side_code, hit_rate, false_alarm_rate = dPrime_2AFC(correct_side, response_side)
    '''
    
    
    import numpy as np
    from scipy.stats import norm
    
    #Check validity
    correct_side = correct_side[np.isnan(response_side) == 0] #Find all the instances where the animal made a choice and exclude incomplete trials
    response_side = response_side[np.isnan(response_side) == 0]    
   
    #Extract one of the side codes regardless of its encoding, e.g. can ne 0 and 1 or -1 and 1, or ...
    side_code = np.unique(response_side)[0]
    
    hit_rate = np.sum((correct_side == side_code) & (response_side == side_code)) / np.sum(correct_side == side_code)
    #Is defined as the proportion the animal responded correctly on the specified side (side_code) when the side was actually correct -> p(response = left | left = correct)
    false_alarm_rate = np.sum(np.not_equal(correct_side, side_code) & (response_side == side_code)) / np.sum(np.not_equal(correct_side, side_code))
    #Is defined as the proportion the animal responded incorrectly with specified side when the other side was actually correct -> p(response = left | right = correct)

    dPrime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
   
    return dPrime, side_code, hit_rate, false_alarm_rate
    

############################################################################
#%%

class fit_learning_curve:
    '''Fit a multi-parameter learning curve to an animal's performance data
    
    Attributes
    ----------
    parameter_names: list, the names of the parameters to estimate
    params: numpy array, the estimates for the respective parameters
    pcov: numpy array, the covarinace matrix between the different parameters
    estimate_std: numpy array, the standard deviation of the parameter estimate
    
    Methods
    -------
    estimate_params: Find the parameters that best fit the provided data
    reconstruct: Reconstruct the curve from a set of x values
    sigmoid: Only internal use, is passed as callable to estimate_params
    
    Usage (step-wise)
    -----
    learning_curve = fit_learning_curve() # initialize the object
    learning_curve.estimate_params(training_x, training_y) #get the parm estimate
    y = learning_curve.reconstruct(test_x) #reconstruct the curve to plot later
    '''
    
    def __init__(self, parameter_names = ['inflection_point', 'slope', 'maximum', 'minimum']):
                 self.parameter_names = parameter_names
      
    # #-----
    # def four_parameter_sigmoid(self, x, inflection_point, slope, maximum, minimum):
    #     '''Define the function for the fitting procedure.
        
    #     Parameters
    #     ---------
    #     x: numpy array, vector of x-vlaues for the fit.
    #     inflection_point: float, parameter to be estimated
    #     slope: float, parameter to be estimated
    #     maximum: float, parameter to be estimated
    #     minimum: float, parameter to be estimated
        
    #     Returns
    #     -------
    #     y: float, unused here
        
    #     Usage
    #     .....
    #     -> Pass as callable to curve_fit
        
    #     '''
    #---    
    def four_parameter_sigmoid(self, x, inflection_point, slope, maximum, minimum):
        '''Define the function for the fitting procedure.
        
        Parameters
        ---------
        x: numpy array, vector of x-vlaues for the fit.
        inflection_point: float, parameter to be estimated
        slope: float, parameter to be estimated
        maximum: float, parameter to be estimated
        
        Returns
        -------
        y: float, unused here
        
        Usage
        .....
        -> Pass as callable to curve_fit
        
        '''
        
        import numpy as np
        y = minimum + maximum / (1 + np.exp(-slope * (x - inflection_point)))
        return y
                 
    def estimate_params(self, training_x, training_y, init_values = None):
        '''Estimate the unknown parapmeters of the defined learning function.
        
        Parameters
        ----------
        training_x: numpy array, vector of x data to fit the relationship
        training_y: numpy array, corresponding y values.
        
        Returns
        -------
        
        
        Usage
        -----
        learning_curve.estimate_params(training_x, training_y)
         
        '''
        import numpy as np
        from scipy.optimize import curve_fit
        
        if init_values is not None:
        #Use some heuristic here to pass a more reasonable starting point for the parameter search
            dummy_infl = training_x.shape[0] / 2 #Assume that the animal learn in the middle of the passed trials
            dummy_slope = (np.mean(training_y[int(training_x.shape[0]/2):training_x.shape[0]]) - np.mean(training_y[0:int(training_x.shape[0]/2)])) / training_x.shape[0]
        #Assume the difference between the first half and the second half divided by total number of samples 
            dummy_max = np.mean(training_y[int(training_x.shape[0]/2):training_x.shape[0]]) #Assume average of second half
            dummy_min = np.mean(training_y[0:int(training_x.shape[0]/2)]) #Assume average of first half
        
            init_values = np.array([dummy_infl, dummy_slope, dummy_max, dummy_min]) #Assemble to array-like structure
        
        popt, pcov = curve_fit(self.four_parameter_sigmoid, training_x, training_y, init_values)
        self.parameters = popt # ptions contain the parameter estimates in the order of parameter_names
        self.pcov = pcov # Covariance matrix between the parameter estimates
        self.estimate_std = np.sqrt(np.diag(pcov)) # The standard deviation of the parameter estimate

    def reconstruct(self, test_x):
        '''Reconstruct the fitted learning curve on a specified sample of x values
        using the estimated parameters from the training data.
        
        Parameters
        ----------
        test_x: numpy array, set of x-values for reconstruction of the learning
                curve.
                
        Returns
        -------
        y: numpy array,reconstructed y values for the input data provided.
        
        Usage
        -----
        y = learning_curve.reconstruct(test_x)
        
        '''
        import numpy as np
        y = self.parameters[3] + self.parameters[2] / (1 + np.exp(-self.parameters[1] * (test_x - self.parameters[0])))
        return y
   
    
    
##############################################################################  
#%%
def choice_strategy(response_side, correct_side, stim_rates,
                    contingency_multiplier = 1, category_boundary = 12, window = None,
                    k_folds = 5, subsampling_rounds = 100, model_params = None):
    '''Fit a logistic regression model to predict the animal's behavior'''
    
    
    
    import numpy as np 
    import pandas as pd
    import multiprocessing as mp #To run the model fitting in parallel   
    import decoding_utils
    import time
    
    #---Input check
    if not model_params:
        penalty='none' #Do not regularize the model, allow all the features to shine!
        inverse_regularization_strength = 1 
        solver='newton-cg'
        fit_intercept = False
        model_params = {'penalty': penalty, 'inverse_regularization_strength': inverse_regularization_strength, 'solver': solver, 'fit_intercept': fit_intercept}
   
    secondary_labels = None #No balancing for another class required here 
   
    #---Prepare the data
    valid_trials = np.isnan(response_side)==0
    outcome = np.squeeze(np.array([response_side == correct_side], dtype=float))
    outcome[np.isnan(response_side)] = np.nan
    
    #---Find choices and outcomes from previous trial
    prior_response = decoding_utils.determine_prior_variable(response_side, valid_trials, 1)
    prior_outcome = decoding_utils.determine_prior_variable(outcome, valid_trials, 1)
    prior_category = decoding_utils.determine_prior_variable(correct_side, valid_trials, 1)
    
    #Find the prior trial's successes and failures and the relative signed stimulus strength
    prior_right_success = np.zeros([response_side.shape[0]]) * np.nan
    prior_right_failure = np.zeros([response_side.shape[0]]) * np.nan
    signed_stim_strength = np.zeros([response_side.shape[0]]) * np.nan
    
    for k in range(response_side.shape[0]):
        if np.isnan(prior_response[k]) == 0:
            if prior_response[k] == 1:
                if prior_outcome[k] == 1:
                    prior_right_success[k] = 1
                    prior_right_failure[k] = 0
                elif prior_outcome[k] == 0:
                    prior_right_success[k] = 0
                    prior_right_failure[k] = 1
            elif prior_response[k] == 0:
                if prior_outcome[k] == 1:
                    prior_right_success[k] = -1
                    prior_right_failure[k] = 0
                elif prior_outcome[k] == 0:
                    prior_right_success[k] = 0
                    prior_right_failure[k] = -1
            signed_stim_strength[k] = contingency_multiplier * (stim_rates[k] - category_boundary)
    
    intercept = np.ones([response_side.shape[0]]) #Explicitly fit an intercept term to check for general right side biases
    include_trials = np.where((np.isnan(response_side)==0) & (np.isnan(prior_right_success)==0))[0] #Only include trials with choice info in sequence
    
    if not window:
        #Build the design matrix with, in order: bias term, signed stim strength, prior success (1 = on right, -1 = on left, 0 = failure), prior failure
        data = np.transpose(np.vstack((intercept[include_trials], signed_stim_strength[include_trials], prior_right_success[include_trials], prior_right_failure[include_trials])))
        labels = response_side[include_trials]
        #Fit the model
        choice_strategy_models = decoding_utils.balanced_logistic_model_training(data, labels, k_folds, subsampling_rounds, secondary_labels, model_params)
        
    else: #When the evolution should be tracekd 
        #Build the design matrices and label vectors
        data_list = []
        label_list = []
        performance = [] #Average performance over the time window
        trial_index = [] #The trial in the center of the window
        for k in range(int(window/2), int(include_trials.shape[0]- window/2)):
            #Build the design matrix with, in order: bias term, signed stim strength, prior success (1 = on right, -1 = on left, 0 = failure), prior failure
            data_list.append(np.transpose(np.vstack((intercept[include_trials[int(k-window/2) : int(k+window/2)]],
                                                     signed_stim_strength[include_trials[int(k-window/2) : int(k+window/2)]],
                                                     prior_right_success[include_trials[int(k-window/2) : int(k+window/2)]],
                                                     prior_right_failure[include_trials[int(k-window/2) : int(k+window/2)]]))))
            label_list.append(response_side[include_trials[int(k-window/2) : int(k+window/2)]])
            performance.append(np.mean(response_side[include_trials[int(k-window/2) : int(k+window/2)]] == correct_side[include_trials[int(k-window/2) : int(k+window/2)]]))
            trial_index.append(k)
        
        if __name__ == '__main__': #This part is required for using multiprocessing within this script, check for the main process?
            import numpy as np 
            import pandas as pd
            import multiprocessing as mp #To run the model fitting in parallel   
            import time
            import sys
            sys.path.append('C:/Users/Anne/Documents/chiCa') #Include the path to the functions
            import decoding_utils
            
            
            start_parallel = time.time() #Measure the time
            steps = len(data_list) #Check the number of steps in the data
            par_pool = mp.Pool(mp.cpu_count())
            output_models = par_pool.starmap(decoding_utils.balanced_logistic_model_training,
                                     [(data_list[k], label_list[k], k_folds, subsampling_rounds, secondary_labels, model_params
                                       ) for k in range(steps)])

            par_pool.close()
            stop_parallel = time.time()
            print('-------------------------------------------')
            print(f'Done fitting the models in {round(stop_parallel - start_parallel)} seconds')
        
            #Create a data frame that holds all the model data frames
            choice_strategy_models = pd.DataFrame([output_models, performance, trial_index]).T
            choice_strategy_models.columns = ['models', 'animal_performance', 'trial_index']
   
    return choice_strategy_models


#%%
#if __name__ == '__main__': #This part is required for using multiprocessing within this script. Why?
def choice_strategy_draft(session_dir):
    '''xxx'''
    import numpy as np
    import pandas as pd
    import multiprocessing as mp #To run the model fitting in parallel
    from tkinter import Tk #For interactive selection, this part is only used to withdraw() the little selection window once the selection is done.
    import tkinter.filedialog as filedialog
    import glob #To pick files of a specified type
    import time
    from os.path import isdir
    from os import mkdir    
    import decoding_utils
    
    #%%-----Load the data etc.
    
    #session_dir = 'C:/data/LO032/20220218_113904' #Can also be None
    #Select the directory
    if session_dir is None:
        Tk().withdraw() #Don't show the tiny confirmation window
        session_dir = filedialog.askdirectory()
        
    #Load the dataframe
    trialdata = pd.read_hdf(glob.glob(session_dir + '/trial_alignment/*.h5')[0], '/Data')
    #This might need to be adapted if more hdf5 files are added to the alignment directory
    #or if the trial alignment changes its location...
    
    #%%-----Prepare the data
    
    #Extract choice and correct side information and compute outcome
    response_side = np.array(trialdata['response_side'])
    correct_side = np.array(trialdata['correct_side'])
    valid_trials = np.isnan(response_side)==0
        
    outcome = np.squeeze(np.array([response_side == correct_side], dtype=float))
    outcome[np.isnan(response_side)] = np.nan
    
    #Find choices and outcomes from previous trial
    prior_response = decoding_utils.determine_prior_variable(response_side, valid_trials, 1)
    prior_outcome = decoding_utils.determine_prior_variable(outcome, valid_trials, 1)
    prior_category = decoding_utils.determine_prior_variable(correct_side, valid_trials, 1)
    
    #Task parameters, should be automatized eventually
    contingency_multiplier = 1 #Use this one to switch the stimulus sign when the left side is the high rate side
    category_boundary = 12
    
    #Find the prior trial's successes and failures and the relative signed stimulus strength
    prior_right_success = np.zeros([trialdata.shape[0]]) * np.nan
    prior_right_failure = np.zeros([trialdata.shape[0]]) * np.nan
    signed_stim_strength = np.zeros([trialdata.shape[0]]) * np.nan
    
    for k in range(trialdata.shape[0]):
        if np.isnan(prior_response[k]) == 0:
            if prior_response[k] == 1:
                if prior_outcome[k] == 1:
                    prior_right_success[k] = 1
                    prior_right_failure[k] = 0
                elif prior_outcome[k] == 0:
                    prior_right_success[k] = 0
                    prior_right_failure[k] = 1
            elif prior_response[k] == 0:
                if prior_outcome[k] == 1:
                    prior_right_success[k] = -1
                    prior_right_failure[k] = 0
                elif prior_outcome[k] == 0:
                    prior_right_success[k] = 0
                    prior_right_failure[k] = -1
            signed_stim_strength[k] = contingency_multiplier * (trialdata['stimulus_event_timestamps'][k].shape[0] - category_boundary)
    
    intercept = np.ones([trialdata.shape[0]]) #Explicitly fit an intercept term to check for general right side biases
    
    valid_trials = np.isnan(prior_right_success)==0
    
    #Build the design matrix with, in order: bias term, signed stim strength, prior success (1 = on right, -1 = on left, 0 = failure), prior failure
    data = np.transpose(np.vstack((intercept[valid_trials], signed_stim_strength[valid_trials], prior_right_success[valid_trials], prior_right_failure[valid_trials])))
    labels = response_side[valid_trials]
    
    #%%-------------------Fit the model
    
    #Define the model parameters
    penalty='none' 
    inverse_regularization_strength = 1 
    solver='newton-cg'
    model_params = {'penalty': penalty, 'inverse_regularization_strength': inverse_regularization_strength, 'solver': solver}
    
    k_folds = 10
    secondary_labels = None 
    subsampling_rounds = 100 
    
    #Fit the model
    choice_strategy_models = decoding_utils.balanced_logistic_model_training(data, labels, k_folds, subsampling_rounds, secondary_labels, model_params)
    
    
    #%%-----Separately compute stay probability
    
    stay_probability = (np.sum(response_side[valid_trials] == prior_response[valid_trials]))/np.sum(valid_trials)
    
    match_prev_cat = (np.sum(response_side[valid_trials] == prior_category[valid_trials]))/np.sum(valid_trials)
    
    
    #%%----Calculate the probability that the trial following a completed trial will
    # be an early withdrawal if it is aligned with a win-stay or if it is the opposite strategy
    
    early_withdrawal_if_opposite = np.zeros(response_side.shape[0])* np.nan
    early_withdrawal_if_same = np.zeros(response_side.shape[0])* np.nan
    
    for k in range(1,early_withdrawal_if_opposite.shape[0]):
        if (np.isnan(response_side[k-1]) ==0) and (correct_side[k] == correct_side[k-1]) and (np.isnan(trialdata['DemonEarlyWithdrawal'][k][0]) == 1 ):
            early_withdrawal_if_same[k] = 0
        elif (np.isnan(response_side[k-1]) ==0)  and (correct_side[k] == correct_side[k-1]) and (np.isnan(trialdata['DemonEarlyWithdrawal'][k][0]) == 0 ):
            early_withdrawal_if_same[k] = 1
            
        if (np.isnan(response_side[k-1])==0) and (correct_side[k] != correct_side[k-1]) and (np.isnan(trialdata['DemonEarlyWithdrawal'][k][0]) == 1 ):
            early_withdrawal_if_opposite[k] = 0
        elif (np.isnan(response_side[k-1]) ==0) and (correct_side[k] != correct_side[k-1]) and (np.isnan(trialdata['DemonEarlyWithdrawal'][k][0]) == 0 ):
            early_withdrawal_if_opposite[k] = 1
    


#%%-----check whether the mouse goes to the other side after errors and how frequently for each side

def post_outcome_side_switch(trialdata):
    '''Assess whether an animal switched the side after outcome presentation 
    and tried to poke in the other port. This can often be observed after error
    trials. Switching the side after the outcome can change the heading direction
    of the mouse and may be a strategy to retain explicit information about 
    the previously correct side.
    
    Parameters
    ----------
    trialdata: pandas data frame containing the task information for each trial
    
    Returns
    -------
    side_switch: numpy array indicating whether an animal switched to the other
                 side before initiating the next trial, 0 = No, 1 = yes, nan = No outcome 
                 during the current trial
    heading_direction: numpy array encoding the putative side the animal approaches
                       the center poke from on the next trial based on the behavioral data and 
                       the detected switching. 0 = heading rightward (coming 
                       from the left), 1 = heading leftward (coming from the 
                       right), nan = incomplete prior trial.
    poke_on_previous: numpy array, flag indicating whether the additional poke happens
                      during the trial time after the switch (then it is 0) or 
                      during the time of the previous trial (1)
    poke_timing: numpy array, time in seconds at which the switching poke happend.
                 The corresponding trial is: side_switch[k] - poke_on_previous[k]                                           
                                                            
    Examples
    --------
    side_switch, heading_direction, poke_on_previous, poke_timing = post_outcome_side_switch(trialdata)
    '''
    import numpy as np
    import pandas as pd
    
    response_side = np.array(trialdata['response_side'])
    correct_side = np.array(trialdata['correct_side'])
    outcome = np.squeeze(np.array([response_side == correct_side], dtype=float)) #Include this in the trialdata in the future
    outcome[np.isnan(response_side)] = np.nan
    
    #Preprocess the trial events to have each entry as an ndarray inside a list. 
    #This is helpful to refrence later on, but it is soooo annoying!
    #Consider changing the trial event extraction to make this happen already.
    left_port_in = [] #Consider port in on the old trial
    right_port_in = []
    left_port_out = [] #Consider out on the new trial
    right_port_out = []
    for k in range(trialdata.shape[0]):
        if trialdata['Port1In'][k] is not None:
            if isinstance(trialdata['Port1In'][k], np.ndarray): 
                left_port_in.append(trialdata['Port1In'][k])
            else:
                left_port_in.append(np.array([trialdata['Port1In'][k]]))
        else:
            left_port_in.append(np.array([np.nan]))
            
        if trialdata['Port3In'][k] is not None:
            if isinstance(trialdata['Port3In'][k], np.ndarray): 
                right_port_in.append(trialdata['Port3In'][k])
            else:
                right_port_in.append(np.array([trialdata['Port3In'][k]]))
        else:
            right_port_in.append(np.array([np.nan]))
            
        if trialdata['Port1Out'][k] is not None:
            if isinstance(trialdata['Port1Out'][k], np.ndarray): 
                left_port_out.append(trialdata['Port1Out'][k])
            else:
                left_port_out.append(np.array([trialdata['Port1Out'][k]]))
        else:
            left_port_out.append(np.array([np.nan]))
            
        if trialdata['Port3Out'][k] is not None:
            if isinstance(trialdata['Port3Out'][k], np.ndarray): 
                right_port_out.append(trialdata['Port3Out'][k])
            else:
                right_port_out.append(np.array([trialdata['Port3Out'][k]]))
        else:
            right_port_out.append(np.array([np.nan]))
       
            
       
    #----Find the switching-----------------------------------
    side_switch = np.zeros([outcome.shape[0]]) * np.nan
    heading_direction = np.zeros([outcome.shape[0]]) * np.nan
    poke_on_previous = np.zeros([outcome.shape[0]]) * np.nan
    poke_timing = np.zeros([outcome.shape[0]]) * np.nan     
     
    #Use the upcoming trial as the reference trial and look back on the previous one
    for k in range(1,response_side.shape[0]):
        if np.isnan(response_side[k-1])==0: #The previous trial actually contained a choice
        
            if response_side[k-1] == 0: #When the left was chosen
                #Try first to locate the switch in the previous trial...
                if np.isnan(right_port_in[k-1][0]) == 0: #Were pokes on the right detected?
                    tmp = np.where(right_port_in[k-1] > trialdata['DemonWaitForResponse'][k-1][1])[0].astype(int) # Did these pokes happen after the choice?
                    if tmp.shape[0] > 0:
                        side_switch[k] = 1
                        poke_on_previous[k] = 1
                        poke_timing[k] = right_port_in[k-1][tmp[0]]
                        heading_direction[k] = 1
                #Check whether the mouse poked in the current trial...
                elif np.isnan(right_port_in[k][0]) == 0: #Were pokes on the right detected?
                    tmp = np.where(right_port_in[k] < trialdata['DemonInitFixation'][k][0])[0].astype(int) # Did these pokes happen before initiating at the center?
                    if tmp.shape[0] > 0:
                        side_switch[k] = 1
                        poke_on_previous[k] = 0
                        poke_timing[k] = right_port_in[k][tmp[0]]
                        heading_direction[k] = 1
                #Nothing detected... 
                else:
                    side_switch[k] = 0
                    heading_direction[k] = response_side[k-1]        
                        
                        
            elif response_side[k-1] == 1: #When the right was chosen
                #Try first to locate the switch in the previous trial...
                if np.isnan(left_port_in[k-1][0]) == 0: #Were pokes on the right detected?
                    tmp = np.where(left_port_in[k-1] > trialdata['DemonWaitForResponse'][k-1][1])[0].astype(int) # Did these pokes happen after the choice?
                    if tmp.shape[0] > 0:
                        side_switch[k] = 1
                        poke_on_previous[k] = 1
                        poke_timing[k] = left_port_in[k-1][tmp[0]]
                        heading_direction[k] = 0
                #Check whether the mouse poked in the current trial...
                elif np.isnan(left_port_in[k][0]) == 0: #Were pokes on the right detected?
                    tmp = np.where(left_port_in[k] < trialdata['DemonInitFixation'][k][0])[0].astype(int) # Did these pokes happen before initiating at the center?
                    if tmp.shape[0] > 0:
                        side_switch[k] = 1
                        poke_on_previous[k] = 0
                        poke_timing[k] = left_port_in[k][tmp[0]]
                        heading_direction[k] = 0
                #Nothing detected... 
                else:
                    side_switch[k] = 0
                    heading_direction[k] = response_side[k-1]        
                        

                    
    return side_switch, heading_direction, poke_on_previous, poke_timing
  
#------------------------------------------------------------------------------  
#%% ---- Look more closely at what happends after early withdrawals. Does the
#        mouse still go to one side, does it wait?
def early_withdrawal_action(trialdata):
    '''xxx'''
    
    import numpy as np
    import pandas as pd
    
    #Retrieve time of early withdrawal and subsequent poke on one of the side pokes
    early_withdrawal_time = np.zeros([trialdata.shape[0]]) * np.nan
    side_poke_time = np.zeros([trialdata.shape[0]]) * np.nan
    chosen_side = np.zeros([trialdata.shape[0]]) * np.nan
   
    for k in range(trialdata.shape[0]):
        if np.isnan(trialdata['DemonEarlyWithdrawal'][k][0]) == 0:
            early_withdrawal_time[k] = trialdata['DemonEarlyWithdrawal'][k][0] - trialdata['DemonInitFixation'][k][0]
            
            #Check for left and right port entries
            tmp_l = trialdata['Port1In'][k][trialdata['Port1In'][k] > trialdata['DemonEarlyWithdrawal'][k][0]]
            tmp_r = trialdata['Port3In'][k][trialdata['Port3In'][k] > trialdata['DemonEarlyWithdrawal'][k][0]]
            if (tmp_l.shape[0] == 1) & (tmp_r.shape[0] == 0):
                side_poke_time[k] = tmp_l[0]
                chosen_side[k] = 0
            elif (tmp_r.shape[0] == 1) & (tmp_l.shape[0] == 0):
                 side_poke_time[k] = tmp_r[0]
                 chosen_side[k] = 1
            elif (tmp_r.shape[0] == 1) & (tmp_l.shape[0] == 1):
                 if tmp_l[0] < tmp_r[0]:
                     side_poke_time[k] = tmp_l[k]
                     chosen_side[k] = 0
                 else:
                     side_poke_time[k] = tmp_r[0]
                     chosen_side[k] = 1

    return early_withdrawal_time, side_poke_time, chosen_side