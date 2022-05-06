# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:47:08 2022

@author: Lukas Oesch
"""


#if __name__ == '__main__': #This part is required for using multiprocessing within this script. Why?
def choice_strategy(session_dir):
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
                 side in the previous trial, 0 = No, 1 = yes, nan = No outcome 
                 in previous trial
    heading_direction: numpy array encoding the putative side the animal approaches
                       the center poke from based on the behavioral data and 
                       the detected switching. 0 = heading rightward (coming 
                       from the left), 1 = heading leftward (coming from the 
                       right), nan = incomplete prior trial.
                                                            
    Examples
    --------
    side_switch, heading_direction = post_outcome_side_switch(trialdata)
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
            
            
    #Find the times when the mouse poke in the port of the opposite side after
    #after outcome presentation but still during the current trial
    poke_other_side = [np.array([np.nan])]
    for k in range(outcome.shape[0]-1):
        if (np.isnan(outcome[k]) == 0):
            if response_side[k] == 0:
                if np.isnan(right_port_in[k][0]):
                    poke_other_side.append(np.array([0]))
                else:
                    tmp = np.where(right_port_in[k] > trialdata['DemonInitFixation'][k][0])[0].astype(int)
                    poke_other_side.append(right_port_in[k][tmp])
            elif response_side[k] == 1:
                if np.isnan(left_port_in[k][0]):
                    poke_other_side.append(np.array([0]))
                else:
                    tmp = np.where(left_port_in[k] > trialdata['DemonInitFixation'][k][0])[0].astype(int)
                    poke_other_side.append(left_port_in[k][tmp])
        else:
           poke_other_side.append(np.array([np.nan])) 
    poke_other_side = [np.array([0]) if n.shape[0]==0 else n for n in poke_other_side]
    #The line above is to account for cases where an empty array is returned when
    #the animal pokes into the respective side port before actually initiating the trial
    
    #Often the other side poke only happens when a new trial could actually already
    #be initiated. Check on the next trial's timestamps for side pokes before center poke
    #CAUTION: Here the relevant event is PortXout!               
    poke_other_side_next = [np.array([np.nan])]
    for k in range(outcome.shape[0]-1):
        if (np.isnan(outcome[k]) == 0):
            if response_side[k] == 0:
                if np.isnan(right_port_out[k+1][0]):
                    poke_other_side_next.append(np.array([0]))
                else:
                    tmp = np.where(right_port_out[k+1] < trialdata['DemonInitFixation'][k][0])[0].astype(int)
                    poke_other_side_next.append(right_port_out[k+1][tmp])
            elif response_side[k] == 1:
                if np.isnan(left_port_out[k+1][0]):
                    poke_other_side_next.append(np.array([0]))
                else:
                    tmp = np.where(left_port_out[k+1] < trialdata['DemonInitFixation'][k][0])[0].astype(int)
                    poke_other_side_next.append(left_port_out[k+1][tmp])
        else:
           poke_other_side_next.append(np.array([np.nan])) 
    poke_other_side_next = [np.array([0]) if n.shape[0]==0 else n for n in poke_other_side_next]
    
    #Create a label of whether the mouse went to the other side
    #Construct this here such that the switch already counts for the new trial, allowing
    #us to compare effects of behavior to putative categroy encoding
    side_switch = np.zeros([outcome.shape[0]]) * np.nan
    heading_direction = np.zeros([outcome.shape[0]]) * np.nan
    for k in range(1,len(poke_other_side)):
        if (poke_other_side[k][0] > 0) or (poke_other_side_next[k][0] > 0):
                side_switch[k] = 1
                if response_side[k-1] == 0:
                    heading_direction[k] = 1
                elif response_side[k-1] == 1:
                    heading_direction[k] = 0
        elif (poke_other_side[k][0] == 0) or (poke_other_side_next[k][0] == 0):
                side_switch[k] = 0
                if response_side[k-1] == 0:
                    heading_direction[k] = 0
                elif response_side[k-1] == 1:
                    heading_direction[k] = 1
                    
    return side_switch, heading_direction
    

