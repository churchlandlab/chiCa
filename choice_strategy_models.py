# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:47:08 2022

@author: Lukas Oesch
"""


if __name__ == '__main__': #This part is required for using multiprocessing within this script. Why?

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

    session_dir = 'C:/data/LO032/20220215_114758' #Can also be None
    #Select the directory
    if session_dir is None:
        Tk().withdraw() #Don't show the tiny confirmation window
        session_dir = filedialog.askdirectory()
        
    #Load the dataframe
    trialdata = pd.read_hdf(glob.glob(session_dir + '/trial_alignment/*.h5')[0], '/Data')
    #This might need to be adapted if more hdf5 files are added to the alignment directory
    #or if the trial alignment changes its location...
    
#%%-----Prepare the data

response_side = np.array(trialdata['response_side'])
correct_side = np.array(trialdata['correct_side'])
valid_trials = np.isnan(response_side)==0
    
outcome = np.squeeze(np.array([response_side == correct_side], dtype=float))
outcome[np.isnan(response_side)] = np.nan

prior_response = decoding_utils.determine_prior_variable(response_side, valid_trials, 1)
prior_outcome = decoding_utils.determine_prior_variable(outcome, valid_trials, 1)

contingency_multiplier = 1 #Use this one to switch the stimulus sign when the left side is the high rate side
category_boundary = 12

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

intercept = np.ones([trialdata.shape[0]])

valid_trials = np.isnan(prior_right_success)==0
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
