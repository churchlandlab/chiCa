# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:45:26 2022

@author: Lukas Oesch
"""

#%%--------------Do z-Scoring on the calcium time series
def standardize_signal(signal, scale_only = False):
    '''Apply Z-scoring on the provided input signal. Make sure that cells (features)
    are rows and samples are columns.
    Expects inputs as m features x n timepoints matrix. '''
    
    from scipy.stats import zscore
    import numpy as np #Don't need to import numpy because the signal is already a numpy array object that can be transposed
    
    if scale_only:
       st_signal = signal.T / np.std(signal, axis = 1)
    else:
       st_signal = zscore(signal, axis = 1) #This is where the orientation is specified!
       st_signal = st_signal.transpose() #Now flip the orientation to be able to feed it into the model
    
    return st_signal

#%%----------Center head direction angle----------
def center_angles(angles, jump_at_pi = True):
    '''Function to center a set of angles around their circular mean. Caution:
    Depending on the type of angle definition this might not necessarily lead to
    and arithmetic mean of 0!
    
    Parameters
    ----------
    angles: numpy array, where columns are the types of angles and rows are 
            the samples.
    jump_at_pi: bool, the value at which the phase resets. The default value 
                is True.
            
    Returns
    -------
    centered_angles, numpy array, zero-mean angles.
    
    Examples
    --------
    centered_angles = center_angles(angles, jump_at_pi = True)
    '''

    from scipy.stats import circmean
    import numpy as np
    
    
    #Input check, make 2d array if required 
    if len(angles.shape) == 1:
            angles = angles.reshape(-1,1)
    
    #Define the values at which the angles reset
    if jump_at_pi == True: #Angles range from -pi to +pi
        high_end = np.pi
        low_end = -np.pi
    else:
        high_end = np.pi * 2
        low_end = 0
    
    mean_angle = circmean(angles, high = high_end, low = low_end, axis=0) 
    centered_angles = angles - mean_angle #Subtract the mean angle from all the individual angles
   
    #Add 2pi to every angle that is smaller than the lower bound to keep the circular representation
    centered_angles[centered_angles < low_end] = centered_angles[centered_angles < low_end] + 2*np.pi
    
    return centered_angles

#%%-------------Retrieve the time stamps for the occurence of the task state of interest
def find_state_start_frame_imaging(state_name, trialdata, average_interval, trial_starts, trial_start_time_covered = None, trial_end_state_name = 'FinishTrial'):
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
    average_interval: float, the mean interval in seconds between imaging frames, used
                      to calculate the aligned frame indices from the trial start 
                      frame index and the state occurence timers.
    trial_starts: numpy array, The indices of the trial starts
    trial_start_time_covered: numpy array, vector of time within the trial that 
                              was captured within the imaging trial start frame. Default = None
    
    Retunrs
    -------
    state_start_frame: list, the frame index when the respective state started
                       in every trial.
    state_time_covered: numpy array, the time after state onset that is covered
                        by the imaging frame. This only contains values if trial_start_time_covered
                        was provided.
                        
    Examples
    --------
    state_start_frame, state_time_covered = find_state_start_frame_imaging(state_name, trialdata, average_interval, trial_start_time_covered)
    '''
    import numpy as np
                                                
    state_start_frame = np.zeros([len(trialdata)]) #The frame that covers the start of
    
    if trial_start_time_covered is not None: #This should be the case for the calcium imaging data
        state_time_covered = np.zeros([len(trialdata)]) #The of the side that has been covered by the frame
        for n in range(len(trialdata)): #Subtract one here because the last trial is unfinished
            state_timing = trialdata[state_name][n][0]
            if isinstance(state_timing,np.ndarray): #This happens when states can be visited more than once, like the wait for animal in spatial sparrow
                state_timing = trialdata[state_name][n][-1][0]
            if np.isnan(state_timing) == 0: #The state has been visited
                try:    
                      frame_time = np.arange(trial_start_time_covered[n], (trialdata[trial_end_state_name][n][0] + 30) - trialdata['Sync'][n][0] + average_interval, average_interval) #Add one more frame as safety margin
                      #Generate frame times starting the first frame at the end of its coverage of trial inforamtion
                except:
                      frame_time = np.arange(trial_start_time_covered[n], (trialdata[trial_end_state_name][n][0] + 30) - trialdata['ObsTrialStart'][n][0] + average_interval, average_interval)
                      #If this is the previous implementation of chipmunk
                tmp = frame_time - state_timing #Calculate the time difference
                state_start_frame[n] = int(np.where(tmp > 0)[0][0] + trial_starts[n])
                #np.where returns a tuple where the first element are the indices that fulfill the condition.
                #Inside the array of indices retrieve the first one that is positive, therefore the first
                #frame that caputres some information.
              
                state_time_covered[n] =  tmp[tmp > 0][0] #Retrieve the time that was covered by the frame
            else:
              state_start_frame[n] = np.nan
              state_time_covered[n] = np.nan
              
    else: #This is the case for the behavioral videos
        state_time_covered = None #The of the side that has been covered by the frame
        for n in range(len(trialdata)): #Subtract one here because the last trial is unfinished
            state_timing = trialdata[state_name][n][0]
            if isinstance(state_timing,np.ndarray): #This happens when states can be visited more than once, like the wait for animal in spatial sparrow
                state_timing = trialdata[state_name][n][-1][0]
            if np.isnan(state_timing) == 0: #The state has been visited
                try:    
                      frame_time = np.arange(0, (trialdata[trial_end_state_name][n][0] + 30) - trialdata['Sync'][n][0] + average_interval, average_interval)
                      #Generate frame times starting the first frame at the end of its coverage of trial inforamtion
                except:
                      frame_time = np.arange(0, (trialdata[trial_end_state_name][n][0] + 30) - trialdata['ObsTrialStart'][n][0] + average_interval, average_interval)
                      #If this is the previous implementation of chipmunk
                tmp = frame_time - state_timing #Calculate the time difference
                state_start_frame[n] = int(np.where(tmp > 0)[0][0] + trial_starts[n])
                #np.where returns a tuple where the first element are the indices that fulfill the condition.
                #Inside the array of indices retrieve the first one that is positive, therefore the first
                #frame that caputres some information.
              
            else:
              state_start_frame[n] = np.nan         
          
    return state_start_frame, state_time_covered

#%%---------More general way of aligning imaging data to event or state timestamps
def align_miniscope_to_event(event_timestamps, trial_end_time, frame_interval, trial_start_frames, trial_start_time_covered = None):
    '''Locate the frame during which a certain event or state in the chipmunk task has
    started. The function also returns the index of the trial, during which
    the event(s) occured and the time after state / event onset that was 
    covered by the frame acquisition. This may be helpful when interpreting 
    onset responses and their variability.
    
    Parameters
    ----------
    event_timestamps: list or array, the the time stamp of a certain event relative
                      to the time in the trial. When a list is passed each element
                      of the list represents a trial and the elements of the array 
                      inside each list element are the events captured during the
                      trial. If an array is passed each entry is interpreted as a
                      single event per trial and the shape of the array should be
                      n x 0, where n is the number of trials.
    trial_end_time: list or array, the time stamp at the end of the trial (end of last state)
    frame_interval: float, the mean interval in seconds between imaging frames, used
                      to calculate the aligned frame indices from the trial start 
                      frame index and the state occurence timers.
    trial_start_frames: numpy array, The frame indices for the trial starts
    trial_start_time_covered: numpy array, vector of time within the trial that 
                              was captured within the imaging trial start frame. Default = None
    
    Retunrs
    -------
    event_start_frame: array, the frame index when the respective state started
                       in every trial. Note: The array can contain nan and is 
                       therefore a float. This may require changing to int when 
                       indexing into other data!
    event_time_covered: numpy array, the time after state onset that is covered
                        by the imaging frame. This only contains values if trial_start_time_covered
                        was provided.
                        
    Examples
    --------
    event_start_frame, trial_id, state_time_covered = align_miniscope_to_event(event_timestamps, trial_end_time, frame_interval, trial_start_frames, trial_start_time_covered)
    '''
    import numpy as np
    
    #Input check, this is less important
    if type(event_timestamps) == list: #In this case fuse all the timestamps stored in individual arrays togehter
        for k in range(len(event_timestamps)):
            if k==0:
                tmp = event_timestamps[k]
                trial_id = np.ones([event_timestamps[k].shape[0]]) *k
            else:
                tmp = np.hstack((tmp, event_timestamps[k]))
                trial_id = np.hstack((trial_id, np.ones([event_timestamps[k].shape[0]]) *k)).astype(int)
        event_timestamps = tmp
    else: #If an array is provided assume that there is one entry per trial
         trial_id = np.arange(event_timestamps.shape[0]) 
                           
    event_start_frame = np.zeros([event_timestamps.shape[0]]) * np.nan #Create an array that covers the start of the events
    
    if trial_start_time_covered is not None: #This should be the case for the calcium imaging data
       event_time_covered = np.zeros([event_timestamps.shape[0]]) * np.nan #The amount of frame time for which the event was present
       for n in range(event_timestamps.shape[0]):
            if np.isnan(event_timestamps[n]) == 0: #The the event has occurred or the state has been visited
                frame_time = np.arange(trial_start_time_covered[trial_id[n]], trial_end_time[trial_id[n]] + frame_interval, frame_interval) #Add one more frame as safety margin   
                tmp = frame_time - event_timestamps[n] #Calculate the time difference
                event_start_frame[n] = int(np.where(tmp > 0)[0][0] + trial_start_frames[trial_id[n]])
                #Inside the array of indices retrieve the first one that is positive, therefore the first
                #frame that caputres some information.
                event_time_covered[n] =  tmp[tmp > 0][0] #Retrieve the time that was covered by the frame
              
    else: #This is the case when the coverage of the trial start by frames is unknown
        event_time_covered = None #The of the side that has been covered by the frame
        for n in range(event_timestamps.shape[0]):
            if np.isnan(event_timestamps[n]) == 0: #The the event has occurred or the state has been visited
                frame_time = np.arange(0, trial_end_time[trial_id[n]] + frame_interval, frame_interval) #Add one more frame as safety margin   
                tmp = frame_time - event_timestamps[n] #Calculate the time difference
                event_start_frame[n] = int(np.where(tmp > 0)[0][0] + trial_start_frames[trial_id[n]])
             
    return event_start_frame, trial_id, event_time_covered



#%%------------Track task variable in past trials
def determine_prior_variable(tracked_variable, include_trials, trials_back = 1, mode = 'consecutive'):
    '''Get the label of a task variable or action on a preceeding trial a specified
    number of trials back. The user my specifiy a set of trials at present time 
    for which the task variables at the prior trial should be retrieved. The
    mode allows the user to specifiy how to evaluate the history of the tracked
    variable. 'consecutive' will only consider a sequence of consecutive trails 
    where the variable was valid, 'independent' will ignore all trials between
    the current time point and the amount of trials back and only consider the 
    tracked variable at the specified timepoint and 'memory' will keep track of
    the variable and look at the value it held at the specified amount of valid
    trials back.
   
   Illustration (when all the trials are included): 
   Input:
       [1, 1, nan, 0, 1, 0, nan, 0 ]
   Output (trials_back = 2, mode = 'consecutive'):
       [nan, nan, 1, nan, nan, 0, 1, nan]
   Output (trials_back = 2, mode = 'independent'):
       [nan, nan, 1, 1, nan, 0, 1, 0]
   Output (trials_back = 2, mode = 'memory'):
       [nan, nan, 1, 1, 1, 0, 1, 0]
    
    Parameters
    ----------
    tracked_variable: a numpy array of the variable to be tracked
    include_trials: numpy array of zeros and ones for trials that should be
                    considered, for instance trials with a choice.
    trials_back: numpy array that sets at what trial in the past
                 the prior variable should be assesed.
    force_consecutive: boolean, Enforce that all the trials between trials back
                       the moment of evaluation are valid and thus do not contain
                       nan.
                             
    Returns
    -------
    prior_label: numpy array of the label of the tracked variable at trial = trials_back.
                 
    Examples
    --------
    prior_label = determine_prior_variable(tracked_variable, include_trials, trials_back)
    prior_label = determine_prior_variable(tracked_variable, include_trials)
    '''
    
    import numpy as np
    
    prior_label = np.zeros([tracked_variable.shape[0]]) * np.nan
    remember_last = np.nan #This is the track of the last trial where the variable was valid
    
    for n in range(trials_back, tracked_variable.shape[0]): #First time this can be detected is after consecutive_trials_back
        if mode == 'consecutive':
            if np.isnan(tracked_variable[n - trials_back]) == 0: #Make sure variable to be tracked is not Nan for the trial in question, might be redundant
                if np.sum(np.isnan(tracked_variable[n - trials_back : n ])==0) == trials_back: #Only include if all consecutive trials are valid.
                    prior_label[n] = tracked_variable[n - trials_back]
        elif mode == 'independent':
             if include_trials[n]: #Only assign a value for the desired trials    
                    prior_label[n] = tracked_variable[n - trials_back]
        elif mode == 'memory':
            if np.isnan(tracked_variable[n - trials_back]) == 0: #If at the specified time in the past the variable was valid
                if include_trials[n]: #Only assign a value for the desired trials    
                    prior_label[n] = tracked_variable[n - trials_back]
                #But remember the past anyway
                remember_last = tracked_variable[n - trials_back] #Update the memory to assign last valid observation if there is a stretch of invalid ones following
            else:
                if include_trials[n]: #Only assign a value for the desired trials    
                    prior_label[n] = remember_last             
    
    return prior_label

#%%------Match video frames to imaging
def match_video_to_imaging(miniscope_frames, miniscope_reference_frame,
                           miniscope_frame_interval, video_reference_frame,
                           video_frame_interval):
    '''Function to pick video frames most closely occuring at the time of 
    a series of miniscope frames. This is done based on a know common reference
    frame (such as the trial start frame). The frames are expected to be ordered 
    such that the miniscope frame with the lowest index ocurs first in the array
    and the one with the highest last. The alignment is performed by generating
    an expected frame occurence based on the common reference frame. This means
    that when some indices are higher and some are lower than the reference,
    the imprecision of the alignment grows symmetrically with distance from
    the reference on both sides.
    
    Parameters
    ---------
    miniscope_frames: numpy array, a set of frame indices from the continuous
                      calcium imaging data.
    miniscope_reference_frame: int, the start frame of a trial event or similar to 
                               serve as a reference point for the alignment
    miniscope_frame_interval: float, average interval between frames from the 
                              imaging session.
    video_reference_frame: int, the start frame from a trial event as obtained
                           for the behavioral video data.
    video_frame_interval: float, average frame interval for behavioral video.
                          
    Returns
    -------
    matching_frames: numpy array, vector of behavioral video frames matching 
                     most closely the miniscope frames provided.
                     
    Examples
    --------
    matching_frames = match_video_to_imaging(miniscope_frames, state_start_index,
                                             miniscope_data['frame_interval'],
                                             state_start_video_index,
                                             video_alignment_data['frame_interval'])

    '''
    
    import numpy as np
    
    #Initialize the output
    matching_frames = np.zeros(miniscope_frames.shape[0], dtype=int)
    
    #Generate the expected timestamsp for the miniscope data
    miniscope_timestamps = (miniscope_frames - miniscope_reference_frame) * miniscope_frame_interval
    
    #Find the minimum and maximum of the provided timestamps
    min_time = np.min(miniscope_timestamps) - 1
    max_time = np.max(miniscope_timestamps) + 1
    
    #Check whether the timestamps happen both before and after the common reference.
    #This is important because then the alignment has to either be constructed going
    #in one direction or extending from the reference into positive and negative direction
    if np.sign(miniscope_timestamps[0]) == np.sign(miniscope_timestamps[-1]):
        #In this scenario we can use a single timestamp vector 
        
        frame_times = np.arange(min_time, max_time, video_frame_interval)
    
    else: #For this case we have to symmetrically move away from the reference in both directions
        #np.arange can't do that, so we have to use some tricks...
        
        min_time = -1 * min_time #Convert to positive value to be able to generate a sequence starting at 0 exactly
       
        tmp_min = -1 * np.flip(np.arange(video_frame_interval, min_time + video_frame_interval, video_frame_interval)) 
        tmp_max = np.arange(0,max_time + video_frame_interval,video_frame_interval) #Include the reference frame as time 0
        frame_times = np.concatenate((tmp_min, tmp_max)) #Represents the expected time of the frame acquisition
    
    #Go through the timestamps and find the video frame that most closely matches
    #the occurence of the miniscope frame time
    for k in range(miniscope_timestamps.shape[0]):
        time_diff = np.abs(miniscope_timestamps[k] - frame_times)
        closest = np.where(time_diff == np.min(time_diff))[0][0]
        
        matching_frames[k] = video_reference_frame + int(frame_times[closest] / video_frame_interval)
    
    return matching_frames

#%%-----------Balance the data sets for top level variable and subvariable
def balance_dataset(labels, secondary_labels = None, mode = 'proportional'):
    '''Balance the number of observaitions for each class in the dataset by
    subsampling from the majority classes while retaining all observations of 
    the minority class. This function offers the option to balance for two
    types of classes. When this is desired the proportion of subclass labels
    will be matched for the different classes. There can be an arbitratry number
    of secondary classes as long as they describe non-independent subclasses.
    For example one can balance for previous choices and outcomes in the labels by constructing
    secondary labels as: previous_choice == 0 & previous_outcome == 0 = 0, 
    previous_choice == 0 & previous_outcome == 1 = 1,
    previous_choice == 1 & previous_outcome == 0 = 2,
    previous_choice == 1 & previous_outcome == 1 = 3.
    In this way they span the entire dataset and represent non-independent conditions.
    The mode controls whether the secondary classes should either be sampled with
    fixed proportions between the two main class labels (proportional, default)
    or whether to take equal numbers of samples from all subclasses (equal).
    
    Parameters
    ----------
    labels: numpy array, vector of the class labels of the data. Nans will be
           ignored.
    secondary_labels: numpy array, vector of additional class labels, default = None
    
    Returns
    -------
    pick_to_balance: numpy array, vector of indices for samples to be icluded
                     to obtain a balanced dataset.
    sample_num = int, the number of samples per condition. This means that
                 pick_to_balance.shape[0] == 2*sample_num.
                     
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
    
    
        #Generate contingency table of dimensions classes x secondary_classes
        cont_table = np.zeros([classes.shape[0], subclasses.shape[0]])
        for k in range(subclasses.shape[0]):
            cont_table[0,k] = np.sum(secondary_labels[labels==0]==subclasses[k])
            cont_table[1,k] = np.sum(secondary_labels[labels==1]==subclasses[k])
            
        #Find column minima, these are the minimum amount of samples of a specific secondary class between the classes
        if mode == 'proportional':
            col_min = np.min(cont_table, axis=0).astype(int)
        elif mode == 'equal':
            col_min = np.tile(np.min(cont_table).astype(int), cont_table.shape[1])
            
        sample_num = np.sum(col_min).astype(int) #Sum over the minima to find the number of samples per class
        #If the secondary labels are relatively similraly distributed in the different classes this will
        #result in the number of samples often being identical with the number of observations of the minmum class.
        #Otherwise the number will be smaller
        
        pick_to_balance = []
        for k in range(subclasses.shape[0]):
           for n in range(classes.shape[0]):
               # if cont_table[n,k] == col_min[k]:
               #     pick_to_balance.append(np.where((labels == classes[n]) & (secondary_labels == subclasses[k]))[0])
               # else:
               #     pick_to_balance.append(np.random.permutation(np.where((labels == classes[n]) & (secondary_labels == subclasses[k]))[0])[:col_min[k]])
               pick_to_balance.append(np.random.permutation(np.where((labels == classes[n]) & (secondary_labels == subclasses[k]))[0])[:col_min[k]])
        pick_to_balance = np.hstack(pick_to_balance).T
                
       

    return pick_to_balance, sample_num

#%%-----------Draw
def balance_dataset_cover_all(labels, secondary_labels = None, mode = 'proportional', number_of_draws = 20):
    '''Balance the number of observaitions for each class in the dataset by
    subsampling from the majority classes while retaining all observations of 
    the minority class. This function offers the option to balance for two
    types of classes. When this is desired the proportion of subclass labels
    will be matched for the different classes. There can be an arbitratry number
    of secondary classes as long as they describe non-independent subclasses.
    For example one can balance for previous choices and outcomes in the labels by constructing
    secondary labels as: previous_choice == 0 & previous_outcome == 0 = 0, 
    previous_choice == 0 & previous_outcome == 1 = 1,
    previous_choice == 1 & previous_outcome == 0 = 2,
    previous_choice == 1 & previous_outcome == 1 = 3.
    In this way they span the entire dataset and represent non-independent conditions.
    The mode controls whether the secondary classes should either be sampled with
    fixed proportions between the two main class labels (proportional, default)
    or whether to take equal numbers of samples from all subclasses (equal).
    This function differs from balance_dataset in that it generates a number of
    draws and ensures that all the data are actually sampled at least once. 
    
    
    Parameters
    ----------
    labels: numpy array, vector of the class labels of the data. Nans will be
           ignored.
    secondary_labels: numpy array, vector of additional class labels, default = None
    mode: string, could either be proportional, so that the secondary label proportions
          are kept constant between the primary labels.The other option is equal
          where all secondary labels occur with equal proportion inside the 
          primary labels.
    number_of_draws: int, how many rounds of subsampling should be used
    
    Returns
    -------
    pick_to_balance: list of numpy arrays, vector of indices for samples to be icluded
                     to obtain a balanced dataset, one list element per draw
    sample_num = list of int, the number of samples per label. This means that
                 for binary classification pick_to_balance.shape[0] == 2*sample_num.
                     
    Examples
    --------
    pick_to_balance, sample_num = balance_dataset(labels, secondary_labels, mode = 'proportional', number_of_draws = 20) #Balance by the two variables
    pick_to_balance, sample_num = balance_dataset(labels) #Balance for classes in label only
    '''
    
    import numpy as np
    
    #-----Define a function to iteratively pick samples covering all indices
    def pick_and_refresh_sample(permuted_indices, pick_number, num_removed):
        '''Function to iteratively pick sets of shuffled indices. This function eliminates
        indices that have already been picked and restarts with a fresh permutation after
        all the indices have been selected once.
        '''
        use_up_to = num_removed + pick_number
        
        if use_up_to > permuted_indices.shape[0]: #This is the case when there are no more unpicked samples for this round
            pick_samples = permuted_indices[num_removed:] #Use all the remaining samples 
            use_up_to = use_up_to - permuted_indices.shape[0] #Determine how many more samples are required from a freshly permuted set of yet unselected indices
            remainder_sample = np.random.permutation(permuted_indices[:num_removed]) #Permute the unchosen indices
            pick_samples = np.hstack((pick_samples, remainder_sample[:use_up_to])) #Add the newly chosen indices the already picked ones
            
            #After completing the full set start with a fresh round where all the indices are available again
            permuted_indices = np.random.permutation(permuted_indices)
            num_removed = 0
            
        else: #This is when there are enough samples that have not yet been picked 
            pick_samples = permuted_indices[num_removed : use_up_to]
            num_removed = use_up_to
            
        return pick_samples, num_removed, permuted_indices
    
    #---------------------------------
    classes = np.unique(labels)
    classes = classes[np.isnan(classes)==0] #Determine the classes in the labels and exclude nans
    
    if secondary_labels is None:    
        class_counts = np.array([np.sum(labels==classes[n]) for n in range(classes.shape[0])])
        #Convoluted code: sum up all the labels falling under one class for the two different classes
        pick_number = int(np.min(class_counts)) #Define how many sample per class to retain
        
        assert np.max(class_counts/pick_number) < number_of_draws, f"Please request at least {round(np.ceil(np.max(class_counts)/pick_number))} draws to cover all the labels."         
        
        class_indices = [np.where(labels==classes[n])[0] for n in range(classes.shape[0])] #Get the indices where the labels belong to each class
        for k in range(len(class_indices)): #Permute the indices here to start
            class_indices[k] = np.random.permutation(class_indices[k])
        
        #Start assembling the indices for balancing the classes
        pick_to_balance = []
        num_removed = [0] * classes.shape[0] #Track eliminated samples
        for draw in range(number_of_draws):
            tmp = []
            for cl in range(classes.shape[0]):
                pick_samples, num_removed[cl], class_indices[cl] = pick_and_refresh_sample(class_indices[cl], pick_number, num_removed[cl])
                tmp.append(pick_samples)
            pick_to_balance.append(np.hstack(tmp))
            
        sample_num = pick_number #In this case the number of samples for each primary class
        #is just the same as the number picked from every class. However, when secondary
        #labels are introduced this is not the case anymore because the samples are now drawn from
        #instances of any given primary and secondary label.
                
    elif secondary_labels is not None:
        #If one needs to balance a subclass inside the labels do the same thing again
        subclasses = np.unique(secondary_labels)
        subclasses = subclasses[np.isnan(subclasses)==0] #Determine the classes in the labels and exclude nans    
    
        #Generate contingency table of dimensions classes x secondary_classes
        cont_table = np.zeros([classes.shape[0], subclasses.shape[0]])
        for k in range(subclasses.shape[0]):
            for n in range(classes.shape[0]):
                cont_table[n,k] = np.sum(secondary_labels[labels==classes[n]]==subclasses[k])
                #cont_table[1,k] = np.sum(secondary_labels[labels==1]==subclasses[k])
     
        #Find column minima, these are the minimum amount of samples of a specific secondary class between the classes
        if mode == 'proportional':
            col_min = np.min(cont_table, axis=0).astype(int)
        elif mode == 'equal':
            col_min = np.tile(np.min(cont_table).astype(int), cont_table.shape[1])
        #Here the column minima are going to determine the pick_number       
        
        assert np.max(cont_table / col_min) < number_of_draws, f"Please request at least {round(np.ceil(np.max(cont_table / col_min)))} draws to cover all the labels." 
        
        class_indices = [[np.nan] * subclasses.shape[0] for k in range(classes.shape[0])] #Initialize a list with sublist to hold the indices of the label x secondary label combinations
        #NOTE: This is the correct way to initialize nested lists if one wants to
        #retain each list as an independently changable object!
        
        for prim in range(cont_table.shape[0]):
            for sec in range(cont_table.shape[1]):
                class_indices[prim][sec] = np.where((labels == classes[prim]) & (secondary_labels == subclasses[sec]))[0]
          
        #Apply permutation to the retrieved indices (could also be done above but it might be helpful to keep this separate for troubleshooting)
        for prim in range(cont_table.shape[0]):
            for sec in range(cont_table.shape[1]):
                class_indices[prim][sec] = np.random.permutation(class_indices[prim][sec])
        
        #Start assembling the indices for balancing the classes
        pick_to_balance = []
        num_removed =  [[0] * subclasses.shape[0] for k in range(classes.shape[0])] #Track eliminated samples
        for draw in range(number_of_draws):
            tmp = []
            for prim in range(classes.shape[0]):
                for sec in range(subclasses.shape[0]):
                    pick_samples, num_removed[prim][sec], class_indices[prim][sec] = pick_and_refresh_sample(class_indices[prim][sec], col_min[sec], num_removed[prim][sec])
                    tmp.append(pick_samples)
            pick_to_balance.append(np.hstack(tmp))
                
        sample_num = np.sum(col_min).astype(int) #Sum over the minima to find the number of samples per class
        #If the secondary labels are relatively similraly distributed in the different classes this will
        #result in the number of samples often being identical with the number of observations of the minmum class.
        #Otherwise the number will be smaller

    return pick_to_balance, sample_num





#%%--------Train cross-validated logistic regression model
def train_logistic_regression(data, labels, k_folds, model_params=None):
    '''Train regularized, corss-validated logistic regression models and perform
    a shuffled control. The corss-validation is performed in a stratified way,
    to maintain similar proportions of the input lables. PLease note that secondary
    labels may not be balanced after this procedure. As a control, the labels
    of the training samples are shuffled and a model then trained on these shuffled
    labels. The shuffled model performance is evaluated on the correct testing
    data. This function performs optimization of the regularization strength if 
    the 'inverse_regularization_strength' passed inside the model_params is a list
    or array. The search will be performed on the first fold only and then applied
    to all the subsequent folds.
    
    Parameters
    ----------
    data: numpy array, rows are observations and columns are features.
    labels: numpy array, vector of lables for the corresponding observations.
    k_folds: int, number of folds to perform cross-validation on
    model_params: dict, specifies model parameters. The keys are: penalty 
                  (the type of regularization to be applied),
                  inverse_regularization_strength (as int, or list or array),
                  solver, fit_intercept
                  
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
    np.seterr(invalid='ignore') #Don't report divide by zero warning
    
    #First, get the model parameters
    if model_params is None:
        penalty='l1' 
        inverse_regularization_strength = 1 
        solver='liblinear'
        fit_intercept = True
        normalize = True
        model_params = {'penalty': penalty, 'inverse_regularization_strength': inverse_regularization_strength, 'solver': solver, 'normalize': normalize}
        #Re-create the model_params to store the defaults 
        
    elif model_params is not None:
        penalty = model_params['penalty']
        inverse_regularization_strength = model_params['inverse_regularization_strength']
        solver = model_params['solver']
        fit_intercept = model_params['fit_intercept']
        normalize = model_params['normalize']
        # Mysteriously this loop does not work inside the function
        # for key,val in model_params.items():
        #     exec(key + '=val')
       
    #Set up the dataframe to store the training outputs 
    models = pd.DataFrame(columns=['model_accuracy', 'model_coefficients', 'model_intercept', 'model_n_iter', 
                                   'shuffle_accuracy', 'shuffle_coefficients', 'shuffle_intercept', 'shuffle_n_iter',
                                   'parameters', 'fold_number','number_of_samples', 'test_index',
                                   'model_prediction_logodds', 'shuffle_prediction_logodds','model_predictions','shuffle_predictions'],
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
        
        if normalize:
            #Estimate mean and std on training sample and normalize both samples by these
            sample_mean = np.mean(X_train, axis=0)
            sample_std = np.std(X_train, axis=0)
            sample_std[sample_std==0] = 1 #When nothing changes don't divide, this will then set all the values to 0
            X_train = (X_train - sample_mean) / sample_std
            X_test = (X_test - sample_mean) / sample_std
     
        y_train, y_test = labels[train_index], labels[test_index]
        
        #A short cut here: Estimate the best regularization on the first fold only
        if n == 0:
            if isinstance(model_params['inverse_regularization_strength'], list) | isinstance(model_params['inverse_regularization_strength'], np.ndarray):
                accuracy = []
                for reg in inverse_regularization_strength:
                    log_reg = LogisticRegression(penalty = penalty, C = reg, solver = solver, fit_intercept = fit_intercept).fit(X_train,y_train)
                    accuracy.append(log_reg.score(X_test, y_test))
                accuracy = np.array(accuracy)
                idx = np.where(accuracy == np.max(accuracy))[0][0]
                # if idx.shape[0] > 1: #When more than one reg yield perfect performance...
                #     idx = idx[0]
                inverse_regularization_strength = model_params['inverse_regularization_strength'][idx]       
                
        #Fit the actual models now        
        y_train_shuffled = np.random.permutation(y_train) #Shuffle the labels of the training data for the control
        #The actual model
        log_reg = LogisticRegression(penalty = penalty, C = inverse_regularization_strength, solver = solver, fit_intercept = fit_intercept).fit(X_train,y_train)
        #The shuffled control
        log_reg_shuffled = LogisticRegression(penalty = penalty, C = inverse_regularization_strength, solver = solver, fit_intercept = fit_intercept).fit(X_train,y_train_shuffled)
        
        models['model_accuracy'][n] = log_reg.score(X_test, y_test)
        models['model_coefficients'][n] = log_reg.coef_
        models['model_prediction_logodds'][n] = log_reg.decision_function(X_test)
        models['model_predictions'][n] = log_reg.predict(X_test)
        if fit_intercept:
            models['model_intercept'][n] = log_reg.intercept_[0] #Returns an array in this case
            models['shuffle_intercept'][n] = log_reg_shuffled.intercept_[0]
        else:
            models['model_intercept'][n] = log_reg.intercept_
            models['shuffle_intercept'][n] = log_reg_shuffled.intercept_
            
        models['model_n_iter'][n] = log_reg.n_iter_[0]
        
        models['shuffle_accuracy'][n] = log_reg_shuffled.score(X_test, y_test)
        models['shuffle_coefficients'][n] = log_reg_shuffled.coef_
        models['shuffle_prediction_logodds'][n] = log_reg_shuffled.decision_function(X_test)
        models['shuffle_predictions'][n] = log_reg_shuffled.predict(X_test)
      
        models['shuffle_n_iter'][n] = log_reg_shuffled.n_iter_[0]

        models['parameters'][n] = {'penalty': penalty, 'inverse_regularization_strength': inverse_regularization_strength, 'solver': solver}
        models['fold_number'][n] = n
        models['number_of_samples'][n]= X_train.shape[0]
        models['test_index'][n] = test_index
    
    return models

#%%---Pipeline for training a set of balanced models with multiple rounds of subsampling
def balanced_logistic_model_training(data, labels, k_folds, subsampling_rounds, secondary_labels, model_params, balancing_mode = 'proportional', cover_all_samples = True):
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
    balancing_mode: string, to sample equally from all the different secondary 
                    classes (equal) or by simply balancing the primary labels
                    for all subclasses (proportional).
    cover_all_samples: bool, flag to use 'balance_dataset_cover_all'. If set to
                       true then the balancing function will systematically draw
                       samples until it every sample has been drawn at least once
                       and it will throw an error informing the user how many draws
                       are necessary to sample the entire dataset.
                  
    Returns
    -------
    log_reg_models: pandas dataframe, results of the model fitting.
    
    Examples
    --------
    log_reg_models = balanced_logistic_model_training(data, labels, k_folds, subsampling_rounds, secondary_labels = secondary_labels, model_params = model_params, balancing_mode = 'proportional', cover_all_samples = True)
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
    
    if cover_all_samples:
       pick_to_balance, _ = balance_dataset_cover_all(labels, secondary_labels, balancing_mode, number_of_draws = subsampling_rounds)
    else:
        pick_to_balance = []
        for k in range(subsampling_rounds):
            tmp, _ = balance_dataset(labels, secondary_labels, balancing_mode)
            pick_to_balance.append(tmp)
    
        
    for s in range(subsampling_rounds):
        # pick_to_balance, _ = balance_dataset(labels, secondary_labels, balancing_mode) #If the secondary label is None it will not be considered        
        models = train_logistic_regression(data[pick_to_balance[s],:], labels[pick_to_balance[s]], k_folds, model_params)
        
        models['pick_to_balance'] = [pick_to_balance[s]] * models.shape[0]
        models['subsampling_round'] = np.ones(models.shape[0]) * s #Add a column with the run of subsampling performed
        log_reg_models = pd.concat([log_reg_models, models])
    
    #Stop the time and print
    exe_stop = time.time()
    print(f'Did logistic model fits with {k_folds} folds, subsampling from majority class {subsampling_rounds} times in {round(exe_stop - exe_start)} seconds.')
    print('-------------------------------------------------------------------')
    
    
    return log_reg_models
    
    
#%%---Pipeline for training linear regression models with a number of cross-validations
def train_linear_regression(x_data, y_data, k_folds, fit_intercept=True):
    '''Corss-validated linear regression models and perform
    a shuffled control.
    
    Parameters
    ----------
    x_data: numpy array, rows are observations and columns are features.
    y_data: numpy array, array of responses, rows are observations.
    k_folds: int, number of folds to perform cross-validation on, if set to 1
             no cross-validation is performed
    fit_intercept: boolean, include a static term
                  
    Returns
    -------
    models: pandas dataframe, results of the model fitting.
    
    Examples
    --------
    models = train_linear_regression(x_data, y_data, 10, fit_intercept = True)
        '''

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold   
    from sklearn.linear_model import LinearRegression

    #Set up the dataframe to store the training outputs 
    models = pd.DataFrame(columns=['model_r_squared', 'model_coefficients', 'model_intercept', 'model_singular_value', 'model_rank',
                                    'shuffle_r_squared', 'shuffle_coefficients', 'shuffle_intercept', 'shuffle_singular_value', 'shuffle_rank',
                                     'fold_number','number_of_samples'],
                          index=range(0, k_folds))
    
    if k_folds > 1: #There are folds
        kf = KFold(n_splits = k_folds, shuffle = True) # Shuffle the observations for cross-validation to make sure 
        #that the full range of y values is represented in the model fits.
    
        kf.get_n_splits(x_data, y_data) #Comupte the respective splits, is this actually necessary?
        k_fold_generator = kf.split(x_data, y_data) #This returns a generator object that spits out a different split at each call
        
    elif k_folds <= 1: #There are no folds
        k_folds = 1 #Make sure the loop can run, although it should break already when creating the data frame...
        #Train the regression model
        #print('Fitting the model on all the data, R squared cannot be interpreted as a metric for the goodness of fit!')
        
    for n in range(k_folds):
        if k_folds > 1:
            train_index, test_index = k_fold_generator.__next__() #This is how the generator has to be called if not directly looped over
            X_train, X_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            y_train_shuffled = np.random.permutation(y_train) #Shuffle the labels of the training data for the control
        else:
            X_train = x_data
            X_test = x_data
            y_train = y_data
            y_train_shuffled = np.random.permutation(y_train) #Shuffle the labels of the training data for the control
            y_test = y_data
        #The actual model
        lin_reg = LinearRegression(fit_intercept = fit_intercept).fit(X_train,y_train)
        #The shuffled control
        lin_reg_shuffled = LinearRegression(fit_intercept = fit_intercept).fit(X_train,y_train_shuffled)
        
        models['model_r_squared'][n] = lin_reg.score(X_test, y_test)
        models['model_coefficients'][n] = lin_reg.coef_ #all arrays
        models['model_intercept'][n] = lin_reg.intercept_
        models['model_singular_value'][n] = lin_reg.singular_
        models['model_rank'][n] = lin_reg.rank_ #Is an int already
        
        models['shuffle_r_squared'][n] = lin_reg_shuffled.score(X_test, y_test)
        models['shuffle_coefficients'][n] = lin_reg_shuffled.coef_
        models['shuffle_intercept'][n] = lin_reg_shuffled.intercept_
        models['shuffle_singular_value'][n] = lin_reg_shuffled.singular_
        models['shuffle_rank'][n] = lin_reg_shuffled.rank_

        models['fold_number'][n] = n
        models['number_of_samples'][n]= X_train.shape[0]         
    
    return models

#%%---Pipeline for ridge regression (linear regression with l2 penalty)

def train_ridge_model(x_data, y_data, k_folds, alpha = None, fit_intercept = True):
    '''Corss-validated linear regression model with l2 regularization. Ridge shrinks
        model coefficients that don't contribute to close to but never exactly 0.
        This helps to prevent over-fitting without losing any of the predictors
        because their weight is never exactly 0. When the fold nummber is 1 (or
        below 1) then cross-validation will only be applied to find the best 
        regularization strength but the final model will be fit to all the data.
    
    Parameters
    ----------
    x_data: numpy array, rows are observations and columns are features.
    y_data: numpy array, array of responses, rows are observations.
    k_folds: int, number of folds to perform cross-validation on
    alpha: list of float or None, the regularization strength. If a list is passed
           the function will determine the best of the provided regularization 
           strengths. The list can also hold a single element.
           If None is provided the best alpha will be determined from a default
           set of values, which are: 0.001, 0.01, 0.1,  1, 10, 100  
    fit_intercept: boolean, include a static term
                  
    Returns
    -------
    models: pandas dataframe, results of the model fitting.
    
    Examples
    --------
    models = train_ridge_model(x_data, y_data, 10, alpha = [0.01, 0.1, 1], fit_intercept = True)
        '''

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold   
    from sklearn.linear_model import Ridge
    from scipy.stats import mode

    #Input check for the regularization strength
    if alpha is None:
        alpha = [0.001, 0.01, 0.1,  1, 10, 100]

    #Set up the dataframe to store the training outputs 
    models = pd.DataFrame(columns=['model_r_squared', 'model_coefficients', 'model_intercept',
                                    'shuffle_r_squared', 'shuffle_coefficients', 'shuffle_intercept',
                                     'cv_R_squared','alpha_values', 'best_alpha', 'fold_number','number_of_samples'],
                          index=range(0, k_folds))
    
    if k_folds > 1: #There are folds
       fold_num = k_folds #Use fold number to retain 
       
    elif k_folds <= 1: #Train the final model on all the data but optimize the regularization strength on 10 folds
        k_folds = 1 #Make sure the loop can run, although it should break already when creating the data frame...
        #Train the regression model
        fold_num = 10
        best_regularization = np.zeros(fold_num) #Allocate this to store the best regularization strength per fold
        #print('Fitting the model on all the data but using 10-fold cross validation to determine regularization strength')
        
    
    
    kf = KFold(n_splits = fold_num, shuffle = True) # Shuffle the observations for cross-validation to make sure 
    #that the full range of y values is represented in the model fits.

    kf.get_n_splits(x_data, y_data) #Comupte the respective splits, is this actually necessary?
    k_fold_generator = kf.split(x_data, y_data) #This returns a generator object that spits out a different splitNo documentation available  at each call
    
    #Train the regression model
    for n in range(fold_num):
        train_index, test_index = k_fold_generator.__next__() #This is how the generator has to be called if not directly looped over
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        y_train_shuffled = y_train[np.random.permutation(y_train.shape[0]),:] #Shuffle the order of the trials in the data
        
        ridge_model = [] #Initialize lists for the model objects
        ridge_shuffle = []
        cv_R_squared = [] #Also prepare a list for the cross-validated R-squared values for the models with different regularization strengths        
        for reg_strength in alpha:
            #The actual model
            ridge_model.append(Ridge(alpha = reg_strength, fit_intercept = fit_intercept).fit(X_train,y_train))
            cv_R_squared.append(ridge_model[-1].score(X_test, y_test))                 
            #The shuffled control
            ridge_shuffle.append(Ridge(alpha = reg_strength, fit_intercept = fit_intercept).fit(X_train,y_train_shuffled))
        
        #Select the best model
        best_model = np.where(np.array(cv_R_squared) == np.max(cv_R_squared))[0][0]
        
        if k_folds > 1:
            models['model_r_squared'][n] = ridge_model[best_model].score(X_test, y_test) #Best model maximizes the explained variance
            models['model_coefficients'][n] = ridge_model[best_model].coef_ #all arrays
            models['model_intercept'][n] = ridge_model[best_model].intercept_
            
            models['shuffle_r_squared'][n] = ridge_shuffle[best_model].score(X_test, y_test)
            models['shuffle_coefficients'][n] = ridge_shuffle[best_model].coef_
            models['shuffle_intercept'][n] = ridge_shuffle[best_model].intercept_
    
            models['cv_R_squared'][n] = cv_R_squared
            models['alpha_values'][n] = alpha
            models['best_alpha'][n] = alpha[best_model]
            models['fold_number'][n] = n
            models['number_of_samples'][n]= X_train.shape[0]
        elif k_folds == 1:
            best_regularization[n] = alpha[best_model]
        
    #Now, if fitting will be done on all the data determine the rgularization strength and fit
    if k_folds == 1:
        #best_alpha = mode(best_regularization)[0][0] #Get the value that was best in most cases
        best_alpha = np.mean(best_regularization)
        y_train_shuffled = np.random.permutation(y_data)
        full_ridge_model = Ridge(alpha = best_alpha, fit_intercept = fit_intercept).fit(x_data,y_data)
        full_ridge_shuffle = Ridge(alpha = best_alpha, fit_intercept = fit_intercept).fit(x_data,y_train_shuffled)
    
        models['model_r_squared'][0] = full_ridge_model.score(x_data, y_data)
        models['model_coefficients'][0] = full_ridge_model.coef_ 
        models['model_intercept'][0] = full_ridge_model.intercept_
        
        models['shuffle_r_squared'][0] = full_ridge_shuffle.score(x_data, y_data)
        models['shuffle_coefficients'][0] = full_ridge_shuffle.coef_
        models['shuffle_intercept'][0] = full_ridge_shuffle.intercept_

        models['cv_R_squared'][0] = cv_R_squared #Here this reflects the R squared values during regularization search
        models['alpha_values'][0] = alpha
        models['best_alpha'][0] = best_alpha
        models['fold_number'][0] = 0
        models['number_of_samples'][0]= x_data.shape[0]
            
    
    return models

#%%---Pipeline for Lasso regression (linear regression with l1 regularization)
def train_lasso_model(x_data, y_data, k_folds, alpha = None, fit_intercept = True):
    '''Corss-validated linear regression model with l1 regularization. Lasso shrinks
        many of the model coefficients to 0 and thus effectively performs
        model selection.
    
    Parameters
    ----------
    x_data: numpy array, rows are observations and columns are features.
    y_data: numpy array, array of responses, rows are observations.
    k_folds: int, number of folds to perform cross-validation on
    alpha: list of float or None, the regularization strength. If a list is passed
           the function will determine the best of the provided regularization 
           strengths. The list can also hold a single element.
           If None is provided the best alpha will be determined from a default
           set of values, which are: 0.001, 0.01, 0.1,  1, 10, 100  
    fit_intercept: boolean, include a static term
                  
    Returns
    -------
    models: pandas dataframe, results of the model fitting.
    
    Examples
    --------
    models = train_lasso_model(x_data, y_data, 10, alpha = [0.01, 0.1, 1], fit_intercept = True)
        '''

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold   
    from sklearn.linear_model import Lasso

    #Input check for the regularization strength
    if alpha is None:
        alpha = [0.001, 0.01, 0.1,  1, 10, 100]

    #Set up the dataframe to store the training outputs 
    models = pd.DataFrame(columns=['model_r_squared', 'model_coefficients', 'model_intercept', 'model_dual_gap',
                                    'shuffle_r_squared', 'shuffle_coefficients', 'shuffle_intercept', 'shuffle_dual_gap',
                                     'cv_R_squared','alpha_values', 'best_alpha', 'fold_number','number_of_samples'],
                          index=range(0, k_folds))
     
    if k_folds > 1: #There are folds
       fold_num = k_folds #Use fold number to retain 
       
    elif k_folds <= 1: #Train the final model on all the data but optimize the regularization strength on 10 folds
        k_folds = 1 #Make sure the loop can run, although it should break already when creating the data frame...
        #Train the regression model
        fold_num = 10
        best_regularization = np.zeros(fold_num) #Allocate this to store the best regularization strength per fold
        print('Fitting the model on all the data but using 10-fold cross validation to determine regularization strength')
        
    
        
    kf = KFold(n_splits = fold_num, shuffle = True) # Shuffle the observations for cross-validation to make sure 
    #that the full range of y values is represented in the model fits.

    kf.get_n_splits(x_data, y_data) #Comupte the respective splits, is this actually necessary?
    k_fold_generator = kf.split(x_data, y_data) #This returns a generator object that spits out a different splitNo documentation available  at each call
    
    #Train the regression model
    for n in range(k_folds):
        train_index, test_index = k_fold_generator.__next__() #This is how the generator has to be called if not directly looped over
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        y_train_shuffled = np.random.permutation(y_train) #Shuffle the labels of the training data for the control
        
        lasso_model = [] #Initialize lists for the model objects
        lasso_shuffle = []
        cv_R_squared = [] #Also prepare a list for the cross-validated R-squared values for the models with different regularization strengths        
        for reg_strength in alpha:
            #The actual model
            lasso_model.append(Lasso(alpha = reg_strength, fit_intercept = fit_intercept).fit(X_train,y_train))
            cv_R_squared.append(lasso_model[-1].score(X_test, y_test))                 
            #The shuffled control
            lasso_shuffle.append(Lasso(alpha = reg_strength, fit_intercept = fit_intercept).fit(X_train,y_train_shuffled))
        
        #Select the best model
        best_model = np.where(np.array(cv_R_squared) == np.max(cv_R_squared))[0][0]
        if k_folds > 1:
            models['model_r_squared'][n] = lasso_model[best_model].score(X_test, y_test) #Best model maximizes the explained variance
            models['model_coefficients'][n] = lasso_model[best_model].coef_ #all arrays
            models['model_intercept'][n] = lasso_model[best_model].intercept_
            models['model_dual_gap'][n] = lasso_model[best_model].dual_gap_
            
            models['shuffle_r_squared'][n] = lasso_shuffle[best_model].score(X_test, y_test)
            models['shuffle_coefficients'][n] = lasso_shuffle[best_model].coef_
            models['shuffle_intercept'][n] = lasso_shuffle[best_model].intercept_
            models['shuffle_dual_gap'][n] = lasso_shuffle[best_model].dual_gap_
    
            models['cv_R_squared'][n] = cv_R_squared
            models['alpha_values'][n] = alpha
            models['best_alpha'][n] = alpha[best_model]
            models['fold_number'][n] = n
            models['number_of_samples'][n]= X_train.shape[0]
        
        elif k_folds == 1:
            best_regularization[n] = alpha[best_model]
        
    #Now, if fitting will be done on all the data determine the rgularization strength and fit
    if k_folds == 1:
            best_alpha = np.mean(best_regularization)
            y_train_shuffled = np.random.permutation(y_data)
            full_lasso_model = Lasso(alpha = best_alpha, fit_intercept = fit_intercept).fit(x_data,y_data)
            full_lasso_shuffle = Lasso(alpha = best_alpha, fit_intercept = fit_intercept).fit(x_data,y_train_shuffled)
            
            models['model_r_squared'][n] = full_lasso_model[best_model].score(X_test, y_test) #Best model maximizes the explained variance
            models['model_coefficients'][n] = full_lasso_model[best_model].coef_ #all arrays
            models['model_intercept'][n] = full_lasso_model[best_model].intercept_
            models['model_dual_gap'][n] = full_lasso_model[best_model].dual_gap_
            
            models['shuffle_r_squared'][n] = full_lasso_shuffle[best_model].score(X_test, y_test)
            models['shuffle_coefficients'][n] = full_lasso_shuffle[best_model].coef_
            models['shuffle_intercept'][n] = full_lasso_shuffle[best_model].intercept_
            models['shuffle_dual_gap'][n] = full_lasso_shuffle[best_model].dual_gap_
    
            models['cv_R_squared'][n] = cv_R_squared
            models['alpha_values'][n] = alpha
            models['best_alpha'][n] = best_alpha
            models['fold_number'][n] = 0
            models['number_of_samples'][n]= X_train.shape[0]
    return models

##########################################################################
#%%------------
def determine_approach_side(session_dir, body_part = 'rectum', time_window = [-0.2,0], label_confidence_threshold = 0.99, show_figures = True):
    '''Determine the side the animal is approaching from for every trial.
    
    Parameters
    ----------
    session_dir: string, the path to the respective session directory. You have 
                 to have the dlc analysis downloaded and the chipmunk file present as well.
    body_part: str, the name of the body part you are basing the approach side on
                    default=rectum
    time_window: list or array, the time with respect to trial initiation
                 considered to be useful to extract the approach side info
    label_confidence_threshold: float, the threshold of label likelihood
    show_figure: bool, flag to plot figures    

    Returns
    -------
    approach_side: array, a vector of approach side labels, 0 is approaching from
                    left, 1 indicates approaching from right
                    
    Usage
    ----
    approach_side = determine_approach_side(session_dir, body_part = 'rectum', time_window = [-0.2,0], show_figures = True)
    '''
    
    import numpy as np
    import pandas as pd
    import os
    import glob
    import chiCa
    from chiCa.choice_strategy_models import fit_kmeans
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"
    
    
    #------Loading the data--------------------
    trialdata = pd.read_hdf(glob.glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    
    #Get video alignment
    video_alignment_files = glob.glob(session_dir + '/analysis/*video_alignment.npy')
    if len(video_alignment_files) > 1:
            print('More than one video is currently not supported')
    video_alignment = np.load(video_alignment_files[0], allow_pickle = True).tolist()
    frame_rate = round(1/video_alignment['frame_interval'])
    
    #Retrieve dlc tracking
    dlc_file = glob.glob(session_dir + '/dlc_analysis/*.h5')[-1] #Load the latest extraction!
    dlc_data = pd.read_hdf(dlc_file)
    dlc_metadata = pd.read_pickle(os.path.splitext(dlc_file)[0] + '_meta.pickle')
    
    #---Extract a set of dlc labels and standardize these.
    dlc_keys = dlc_data.keys().tolist()
    specifier = dlc_keys[0][0] #Retrieve the session specifier that is needed as a keyword
    body_part_name = dlc_metadata['data']['DLC-model-config file']['all_joints_names']
    
    temp_body_parts = []
    part_likelihood_estimate = []
    for bp in body_part_name:
        for axis in ['x', 'y']:
            temp_body_parts.append(np.array(dlc_data[(specifier, bp, axis)]))
        part_likelihood_estimate.append(np.array(dlc_data[(specifier, bp, 'likelihood')]))
    temp_body_parts = np.squeeze(temp_body_parts)
 
    #---Set the times up
    aligned_to = 'DemonInitFixation'
    time_frame = np.array(np.array(time_window)*frame_rate, dtype=int)
    
    #Assemble the task variable design matrix
    choice = np.array(trialdata['response_side'])
    prior_choice =  chiCa.determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
    #valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice)==0))
    #valid_trials =  np.where((np.isnan(prior_choice)==0))
    valid_trials = np.arange(trialdata.shape[0]) #Decided to include all the trials but rather look at high confidence labels.
    
    #------align the label position to the task timing
    body_parts = dict()
    part_likelihood = dict()
   
    state_start_frame, state_time_covered = find_state_start_frame_imaging(aligned_to, trialdata, video_alignment['frame_interval'], video_alignment['trial_starts'])                                                                     
    zero_frame = np.array(state_start_frame[valid_trials] + time_frame[0], dtype=int) #The firts frame to consider
   
    for add_to in np.arange(time_frame[1] - time_frame[0]):
        for bp_idx in range(len(body_part_name)):
            if (add_to == 0):
                body_parts[body_part_name[bp_idx]] = temp_body_parts[2*bp_idx : 2*bp_idx+2, zero_frame + add_to, np.newaxis].transpose(2,1,0)
                part_likelihood[body_part_name[bp_idx]] = part_likelihood_estimate[bp_idx][zero_frame + add_to, np.newaxis].T
            else:   
                body_parts[body_part_name[bp_idx]] = np.concatenate((body_parts[body_part_name[bp_idx]], temp_body_parts[2*bp_idx : 2*bp_idx+2, zero_frame + add_to, np.newaxis].transpose(2,1,0)),axis=0)
                part_likelihood[body_part_name[bp_idx]] = np.concatenate((part_likelihood[body_part_name[bp_idx]], part_likelihood_estimate[bp_idx][zero_frame + add_to, np.newaxis].T),axis=0)
            
    #--------Fit a gaussian mixture model using the body part of interest.
    #Here the first input is the training data
    high_confidence =  np.mean(part_likelihood[body_part],axis=0) > label_confidence_threshold
    x_train = body_parts[body_part][:,high_confidence,0].T
    x_test = body_parts[body_part][:,:,0].T
    
    #approach_side, approach_side_likelihood, gm_obj = fit_gaussian_mixture(x_train, x_test)
    approach_side, gm_obj = fit_kmeans(x_train, x_test)
    
    #----Show figure to validate the result
    if show_figures:
        upper_lims = [dlc_metadata['data']['frame_dimensions'][1], dlc_metadata['data']['frame_dimensions'][0]]
        color_names = ['forestgreen', 'darkorchid']
        line_legend = ['left approach', 'right approach']
        fi = plt.figure()
        ax = fi.add_subplot(111)
        for k in range(2):
            ax.plot(body_parts[body_part][:, approach_side==k , 0], body_parts[body_part][:, approach_side==k, 1], linewidth=0.5, color = color_names[k])
        ax.set_xlim([0,upper_lims[0]])
        ax.set_ylim([0, upper_lims[1]])
        ax.set_xlabel('x pixels')
        ax.set_ylabel('y pixels')
        ax.set_title(session_dir + ' - approach side from ' + body_part)
     
    return approach_side

#%%---Cross-decoding over multiple timepoints

def cross_decoding_analysis(decoding_models, data, labels, valid_trials):
    '''Use a fitted decoding model to evaluate the performance and get the 
    confusion matrices for all the other time points
    
    Parameters
    ------
    decoding_models: pandas data frame or list of dicts, the output from the chiCa's
                     balanced_logistic_model_training or the version converted
                     to a dictionary of list. This data structure contains all the
                     decoder information and info on the trial sampling and 
                     balancing.
    data: array, holding the neural data for the decoding with dimensions:
                     trial timepoint x (valid) trials x cells
    labels: array, the full size vector of binary or multi-class labels to be
            predicted by the neural activity
    valid_trials: array, the indices of valid trials, valid_trials.shape[0] 
                  should be equal to data.shape[1]
    
    Returns
    -------
    prediction_accuracy: timepoint x timepoint matrix of prediction accuracy,
                         where columns represnt seed timepoints and rows the 
                         computed performance from their decoders.
    confusion_matrix: list of arrays, one list element per seed timepoint containing
                      a class x class x timepoint tensor. 
                
    ---------------------------------------------------------------------------
    '''
    
    
    #Imports
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix as get_conf_mat
    import warnings
    warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
    warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
    from time import time
    
    start_t = time()
    #input check
    if isinstance(decoding_models, pd.DataFrame):
        dict_list = []
        for k in output_models:
                dict_list.append(k.to_dict('list'))       
        decoding_models = dict_list
        
    prediction_accuracy = []
    shuffled_prediction_accuracy = [] #This will hold average prediction accuracy per fold and sampling run
    confusion_matrix = []
    is_binary = np.shape(decoding_models[0]['model_coefficients'][0].shape)[0] == 1 #Check if it is binary classifier or multinomial
    
    for time_p in range(len(decoding_models)): #This is the reference time point for the cross decoding
        tmp_acc = []
        tmp_shu = []
        tmp_conf = []
        tmp_idx = []
        for k in range(len(decoding_models[time_p]['model_prediction_logodds'])):
            tmp_idx.append(decoding_models[time_p]['pick_to_balance'][k][decoding_models[time_p]['test_index'][k]])
        tmp_idx = np.hstack(tmp_idx)
        trial_idx = np.unique(tmp_idx) #This is with respect to the valid_trials as valid trial indices
        assert trial_idx.shape[0] == valid_trials.shape[0], f'Not all the trials were sampled in session: {session_dir} at timepoint: {time_p}'
        
        for idx in range(len(decoding_models)):
            if idx == time_p: #When the model is the 
                
                tmp_acc.append(np.mean(decoding_models[idx]['model_accuracy'],axis=0))
                tmp_shu.append(np.mean(decoding_models[idx]['shuffle_accuracy'],axis=0))
                #Retrieve model prediction logodds
                if is_binary: #Annoyingly one needs to stack stuff differently for 1d vs 2d...
                    tmp_dec = np.hstack(decoding_models[time_p]['model_prediction_logodds']).T
                    pred = np.sign(tmp_dec) == 1
                else:
                    tmp_dec = np.vstack(decoding_models[time_p]['model_prediction_logodds'])
                    pred = np.argmax(tmp_dec,axis=1) #Love argmax!
                raw_tmp_conf = get_conf_mat(labels[valid_trials][tmp_idx],pred)
                norm_samples = np.array([np.sum(labels[valid_trials][tmp_idx]==k) for k in np.unique(labels[valid_trials][tmp_idx])]).reshape(1,-1)
                tmp_conf.append(raw_tmp_conf / norm_samples)
                
            else: #When the model is not the one that was fitted...
                #Run through all the subsamplings and folds and use the same training and testing splits for the cross-prediction
                tmp_logodds = []
                shu_logodds = []
                for run in range(len(decoding_models[time_p]['model_coefficients'])):
                    training_index = decoding_models[time_p]['pick_to_balance'][run][list(set(np.arange(decoding_models[time_p]['pick_to_balance'][run].shape[0])) - set(decoding_models[time_p]['test_index'][run]))]
                    #Horrible expression, basically getting the indices of the test set within all of the picked (valid) trials and subtracting them from the set of picked trials and then plugging in these
                    #indices into pick_to_balance to retrieve the original indices of trials
                    testing_index = decoding_models[time_p]['pick_to_balance'][run][decoding_models[time_p]['test_index'][run]]
                    #Compute average and standard deviation from training data
                    training_data = data[idx,training_index,:]
                    av = np.mean(training_data,axis=0)
                    st_dev = np.std(training_data,axis=0)
                    # Standardize with the testing data
                    testing_data = (data[idx, testing_index,:] - av) / st_dev
                    testing_data[np.isnan(testing_data)] = 0 #Remove cases where std is 0 leading to nan, or ..
                    testing_data[np.isinf(testing_data)] = 0 #..where both std and av are 0 leading to inf
                    coefs = decoding_models[time_p]['model_coefficients'][run] #Get decoder weights
                    tmp_logodds.append(testing_data @ coefs.T + decoding_models[time_p]['model_intercept'][run]) #Compute logodds for the different labels (if more than two classes)
                    shu_coefs = decoding_models[time_p]['shuffle_coefficients'][run]
                    shu_logodds.append(testing_data @ shu_coefs.T + decoding_models[time_p]['shuffle_intercept'][run])
                #Calculate accuracy and confusion matrix
                if is_binary: #Annoyingly one needs to stack stuff differently for 1d vs 2d...
                    tmp_dec = np.hstack(tmp_logodds).T
                    pred = np.sign(tmp_dec) == 1
                    tmp_dec = np.hstack(shu_logodds).T
                    shu_pred = np.sign(tmp_dec) == 1
                else:
                    tmp_dec = np.vstack(tmp_logodds)
                    pred = np.argmax(tmp_dec,axis=1) #Love argmax!
                    tmp_dec = np.vstack(shu_logodds) #Do the same with shuffles as well
                    shu_pred = np.argmax(tmp_dec,axis=1) 
                    
                tmp_acc.append(np.mean(labels[valid_trials][tmp_idx] == pred))
                tmp_shu.append(np.mean(labels[valid_trials][tmp_idx] == shu_pred))
                raw_tmp_conf = get_conf_mat(labels[valid_trials][tmp_idx],pred)
                norm_samples = np.array([np.sum(labels[valid_trials][tmp_idx]==k) for k in np.unique(labels[valid_trials][tmp_idx])]).reshape(1,-1)
                tmp_conf.append(raw_tmp_conf / norm_samples)
        prediction_accuracy.append(np.squeeze(tmp_acc))
        shuffled_prediction_accuracy.append(np.squeeze(tmp_shu))
        confusion_matrix.append(np.stack(tmp_conf,axis=2))
        
    print(f'Computed cross-decoding performance in {round(time()- start_t)} s')
    print('--------------------------------------------------')
    return np.vstack(prediction_accuracy).T, np.vstack(shuffled_prediction_accuracy).T, confusion_matrix