# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:53:54 2022

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
    #%%-----Gather all the user specified values
    
    #use_name = 'prior_choice_by_prior_stim_category_stim_period'
    use_name = None
    session_dir = 'C:/data/LO032/20220215_114758'
    #session_dir =  'C:/Users/Lukas Oesch/Documents/ChurchlandLab/TestDataChipmunk/TestMiniscopeAlignment/LO028/20220209_153012' #Can also be None
    #session_dir = None
    signal_type = 'c' #The type of traces to be used for decoding c = denoised, s = inferred spikes, f = detrended raw fluorescence
    aligned_state = 'PlayStimulus' #State to align to 
    decoder_range = [-5, 21] #The range of frames from the alignment time point that should be included,
    #python style lower inclusive, upper exclusive, thus one frame would be [0,1]
    window_size = 1 #The size of the sliding window
    #CAUTION: Only odd-numbered windows symmetrically distribute the activity data. Even-numbered windows
    #weight the prior frames a little more strongly
    label_name = 'response_side' #The column name of the label in the trialdata dataframe
    label_trials_back = 1 #How many consecutive trials back the label should be looked at
    secondary_label_name = 'correct_side' #The column name of the secondary label in the trialdata dataframe
    secondary_label_trials_back = 1 #How many trials back the secondary lablel should be considered
    k_folds = 10 #Folds for cross-validation
    subsampling_rounds = 100 #Re-drawing of samples from majority class
    model_params = None #Possibility to specify model parameters here
    #%%----Load data 
    
    #Select the directory
    if session_dir is None:
        Tk().withdraw() #Don't show the tiny confirmation window
        session_dir = filedialog.askdirectory()
    
    #Load the alignment files and the interpolated traces
    trial_alignment_files =  glob.glob(session_dir + '/trial_alignment/*.npz')
    for k in trial_alignment_files:
        tmp_data = np.load(k)
        for key,val in tmp_data.items(): #Retrieve all the entries and create variables with the respective name, here, C and S and the average #interval between frames, average_interval, which is 1/framerate.
            exec(key + '=val')
            
    #Load the dataframe
    trialdata = pd.read_hdf(glob.glob(session_dir + '/trial_alignment/*.h5')[0], '/Data')
    #This might need to be adapted if more hdf5 files are added to the alignment directory
    #or if the trial alignment changes its location...
    
    #%%-----Start arranging the data
    
    #--------------TEMPORARY-----------------------------------------
    #Add a generic outcome state and a column for what kind of outcome
    if ((aligned_state == 'DemonReward') or (aligned_state == 'DemonWrongChoice')) or (label_name == 'outcome'):
        outcome_timing = []
        outcome = np.zeros([trialdata.shape[0]]) * np.nan
        for k in range(trialdata.shape[0]):
            if np.isnan(trialdata['DemonReward'][k][0]) == 0:
                outcome_timing.append(np.array([trialdata['DemonReward'][k][0]]))
                outcome[k] = 1
            elif np.isnan(trialdata['DemonWrongChoice'][k][0]) == 0:
                outcome_timing.append(np.array([trialdata['DemonWrongChoice'][k][0]]))
                outcome[k] = -1
            else:
                outcome_timing.append(np.array([np.nan]))
        trialdata.insert(trialdata.shape[1], 'OutcomePresentation', outcome_timing)
        trialdata.insert(trialdata.shape[1], 'outcome', outcome)
        
    #pretty ugly:
        if (aligned_state == 'DemonReward') or (aligned_state == 'DemonWrongChoice'):
            aligned_state = 'OutcomePresentation'
    #--------------------------
    
    #Get the labels
    valid_trials = np.array(np.isnan(trialdata['response_side']) == 0) #Here related to valid trials in present time
    #When one of the labels represents a trial in the past
    if (label_trials_back > 0) or (secondary_label_trials_back > 0): #If one of these looks at a trial in the past one needs to redefine the trials to include
        if label_trials_back > 0:
            temp_labels = decoding_utils.determine_prior_variable(np.array(trialdata[label_name]), valid_trials, label_trials_back)
        else:
            temp_labels = np.array(trialdata[label_name])
        
        if (secondary_label_trials_back > 0) and (secondary_label_name is not None):
            temp_secondary_labels = decoding_utils.determine_prior_variable(np.array(trialdata[secondary_label_name]), valid_trials, secondary_label_trials_back)
        else:
            temp_secondary_labels = np.array(trialdata[secondary_label_name]) #Only used to find valid trials here
        
        valid_trials = (np.isnan(temp_labels) == 0) & (np.isnan(temp_secondary_labels) == 0) #Refering now to all the trials, for which valid label combinations exist
        
        labels = temp_labels[valid_trials]
        if secondary_label_name is not None:
            secondary_labels = temp_secondary_labels[valid_trials]
        else:
            secondary_labels = None
            
    #The case where no past labels are used        
    else:
        labels = np.array(trialdata[label_name][valid_trials])
        if secondary_label_name is not None:
            secondary_labels = np.array(trialdata[secondary_label_name][valid_trials])
        else:
            secondary_labels = None

    
    #Standardize the input signal
    if signal_type == 'c':
        signal = decoding_utils.standardize_signal(C_interpolated)
    elif signal_type == 's':
        signal = decoding_utils.standardize_signal(S_interpolated)
    elif signal_type == 'f':
        signal = decoding_utils.standardize_signal(F_interpolated)
    #Note that now the observations are rows and the columns are features
    
    #Align the frames to the respective state
    state_start_frame, state_time_covered = decoding_utils.find_state_start_frame_imaging(aligned_state,trialdata, average_interval,
                                                                                          trial_start_time_covered) #Recover the first frame that captured that state 
    zero_frame = np.array(state_start_frame[valid_trials] + decoder_range[0], dtype=int) #The firts frame to consider
    
    #Retrieve the data and store each dataset inside an element of a list to loop through later
    half_window = [int(np.floor(window_size/2)), int(np.ceil(window_size/2))] #The amount to add or subtract from the current position to include
    #IMPORTANT: Split the window assymetrically for odd numbers here because python includes values on
    #on the lower edge but excludes the ones on the upper edge. In odd-numbered window sizes
    #the center of the window is the aligned frame, in even-numbered ones the aligned frame follows 
    #the middle.
    data_list = []
    for add_to in np.arange(decoder_range[0], decoder_range[1]):
        tmp = np.zeros([zero_frame.shape[0],signal.shape[1]]) * np.nan
        for k in range(zero_frame.shape[0]):
            temp = signal[zero_frame[k] + add_to - half_window[0] : zero_frame[k] + add_to + half_window[1],:]
            tmp[k,:] = np.mean(temp, axis=0)
            
        data_list.append(tmp)
        
    #%%-----Run the model fitting in multiprocess
    #if __name__ == '__main__': #This part is required for using multiprocessing within this script. Why?
    start_parallel = time.time() #Measure the time
    par_pool = mp.Pool(mp.cpu_count())
    output_models = par_pool.starmap(decoding_utils.balanced_logistic_model_training,
                                 [(data, labels ,k_folds, subsampling_rounds, secondary_labels, model_params
                                   ) for data in data_list])

    par_pool.close()
    stop_parallel = time.time()
    print('-------------------------------------------')
    print(f'Done fitting the models in {round(stop_parallel - start_parallel)} seconds')
        
    #%%---Store the models inside a big data_frame
    
    #Create a data frame that holds all the model data frames
    output_data = pd.DataFrame([output_models,  [label_name] * (decoder_range[1] - decoder_range[0]), [label_trials_back] * (decoder_range[1] - decoder_range[0]),
                                [secondary_label_name] * (decoder_range[1] - decoder_range[0]), [secondary_label_trials_back] * (decoder_range[1] - decoder_range[0]),
                                [aligned_state] * (decoder_range[1] - decoder_range[0]),
                                np.arange(decoder_range[0], decoder_range[1]).tolist(), [window_size] * (decoder_range[1] - decoder_range[0])]).T
    output_data.columns = ['models', 'labels', 'labels_trials_back', 'secondary_labels', 'secondary_labels_trials_back', 'state_aligned_to', 'frame_from_alignment', 'window_size']
    
    if not isdir(session_dir + '/decoding'): 
        mkdir(session_dir + '/decoding')
    
    #If a more human-understandable file name is specified use it, otherwise construct from data
    if use_name is None: 
        if secondary_label_name is None:
            output_name = session_dir + '/decoding/' + label_name + '_' + str(label_trials_back) + '_trials_back_aligned_to_' + aligned_state + '.h5'
        else:
            output_name = session_dir + '/decoding/' + label_name + '_' + str(label_trials_back) + secondary_label_name + '_balanced_by' + str(secondary_label_trials_back) + '_trials_back_aligned_to_' + aligned_state + '.h5'  
    else:
        output_name = session_dir + '/decoding/' + use_name + '.h5'
        
    output_data.to_hdf(output_name, '/Data')


