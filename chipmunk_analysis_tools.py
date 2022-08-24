# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:24:46 2022

@author: Lukas Oesch
"""

#%%

def convert_specified_behavior_sessions(file_names):
    '''Convert a list of specified behavior files from .mat or .obsmat to
    pandas data frames. The data frame is saved in the /Data dataset.
    
    Parameters
    ----------
    file_names: list, a list containing the paths of the files to convert
    
    Returns
    -------
    coverted_files: list, the paths to the converted files. To load in an individual
                    file use pandas.read_hdf(converted_file, '/Data').
    
    Examples
    --------
    converted_files = convert_specified_behavior_sessions(file_names)
    '''
    
    import os 
    import pandas as pd
    from scipy.io import loadmat
    import numpy as np
    import warnings

    converted_files = [] #This is to keep track of the files that were converted or already exist in h5 format

    #-----Set up the loop
    for current_file in file_names:
        if os.path.isfile(os.path.splitext(current_file)[0] + '.h5'):
            print(f'File: {os.path.split(current_file)[1]} was skipped because a corresponding h5 file exists already.')
            print('---------------------------------------------------------------------------------------------------')
            converted_files.append(os.path.splitext(current_file)[0] + '.h5')
        else:
            try:
                
                #---------Do all the loading and conversion here
                sesdata = loadmat(current_file, squeeze_me=True,
                                      struct_as_record=True)['SessionData']
            
                tmp = sesdata['RawEvents'].tolist()
                tmp = tmp['Trial'].tolist()
                uevents = np.unique(np.hstack([t['Events'].tolist().dtype.names for t in tmp])) #Make sure not to duplicate state definitions
                ustates = np.unique(np.hstack([t['States'].tolist().dtype.names for t in tmp]))
                trialevents = []
                trialstates = [] #Extract all trial states and events
                for t in tmp:
                     a = {u: np.array([np.nan]) for u in uevents}
                     s = t['Events'].tolist()
                     for b in s.dtype.names:
                         if isinstance(s[b].tolist(), float) or isinstance(s[b].tolist(), int): 
                             #Make sure to include single values as an array with a dimension
                             #Arrgh, in the unusual case that a value is an int this should also apply!
                             a[b] = np.array([s[b].tolist()])
                         else:
                                a[b] = s[b].tolist()
                     trialevents.append(a)
                     a = {u:None for u in ustates}
                     s = t['States'].tolist()
                     for b in s.dtype.names:
                             a[b] = s[b].tolist()
                     trialstates.append(a)
                trialstates = pd.DataFrame(trialstates)
                trialevents = pd.DataFrame(trialevents)
                trialdata = pd.merge(trialevents,trialstates,left_index=True, right_index=True)
                
                #Add response and stimulus train related information: correct side, rate, event occurence time stamps
                trialdata.insert(trialdata.shape[1], 'response_side', sesdata['ResponseSide'].tolist())
                trialdata.insert(trialdata.shape[1], 'correct_side', sesdata['CorrectSide'].tolist())
                
                #Get stim modality
                tmp_modality_numeric = sesdata['Modality'].tolist()
                temp_modality = []
                for t in tmp_modality_numeric:
                    if t == 1:
                        temp_modality.append('visual')
                    elif t == 2: 
                        temp_modality.append('auditory')
                    elif t == 3:
                        temp_modality.append('audio-visual')
                    else: 
                        temp_modality.append(np.nan)
                        print('Could not determine modality and set value to nan')
                        
                trialdata.insert(trialdata.shape[1], 'stimulus_modality', temp_modality)
                
                #Reconstruct the time stamps for the individual stimuli
                event_times = []
                event_duration = sesdata['StimulusDuration'].tolist()[0]
                for t in range(trialdata.shape[0]):
                    temp_isi = sesdata['InterStimulusIntervalList'].tolist().tolist()[t][tmp_modality_numeric[t]-1]
                    #Index into the corresponding trial and find the isi for the corresponding modality
                    
                    temp_trial_event_times = [temp_isi[0]] 
                    for k in range(1,temp_isi.shape[0]-1): #Start at 1 because the first Isi is already the timestamp after the play stimulus
                        temp_trial_event_times.append(temp_trial_event_times[k-1] + event_duration + temp_isi[k])
                
                    event_times.append(temp_trial_event_times + trialdata['PlayStimulus'][t][0]) #Add the timestamp for play stimulus to the event time
                
                trialdata.insert(trialdata.shape[1], 'stimulus_event_timestamps', event_times)
                
                #Insert the outcome record for faster access to the different trial outcomes
                trialdata.insert(0, 'outcome_record', sesdata['OutcomeRecord'].tolist())
                tmp = sesdata['TrialDelays'].tolist()
                for key in tmp[0].dtype.fields.keys(): #Find all the keys and extract the data associated with them
                    tmp_delay = tmp[key].tolist()
                    trialdata.insert(trialdata.shape[1], key , tmp_delay)
                
                tmp = sesdata['ActualWaitTime'].tolist()
                trialdata.insert(trialdata.shape[1], 'actual_wait_time' , tmp)
                #TEMPORARY: import the demonstrator and observer id
                tmp = sesdata['TrialSettings'].tolist()
                trialdata.insert(trialdata.shape[1], 'demonstrator_ID' , tmp['demonID'].tolist())
                
                #Add a generic state tracking the timing of outcome presentation, this is also a 1d array of two elements
                outcome_timing = []
                for k in range(trialdata.shape[0]):
                    if np.isnan(trialdata['DemonReward'][k][0]) == 0:
                        outcome_timing.append(np.array([trialdata['DemonReward'][k][0], trialdata['DemonReward'][k][0]]))
                    elif np.isnan(trialdata['DemonWrongChoice'][k][0]) == 0:
                        outcome_timing.append(np.array([trialdata['DemonWrongChoice'][k][0],trialdata['DemonWrongChoice'][k][0]]))
                    else:
                        outcome_timing.append(np.array([np.nan]))
                trialdata.insert(trialdata.shape[1], 'outcome_presentation', outcome_timing)
                
                if 'ObsOutcomeRecord' in sesdata.dtype.fields:
                    trialdata.insert(1, 'observer_outcome_record', sesdata['ObsOutcomeRecord'].tolist())
                    tmp = sesdata['ObsActualWaitTime'].tolist()
                    trialdata.insert(trialdata.shape[1], 'observer_actual_wait_time' , tmp)
                    tmp = sesdata['TrialSettings'].tolist()
                    trialdata.insert(trialdata.shape[1], 'dobserver_ID' , tmp['obsID'].tolist())
                    
                #----Now the saving
                trialdata.to_hdf(os.path.splitext(current_file)[0] + '.h5', '/Data') #Save as hdf5
                converted_files.append(os.path.splitext(current_file)[0] + '.h5') #Keep record of the converted files
                
            except:
                warnings.warn(f"CUATON: An error occured and {current_file} could not be converted")
    return converted_files

   ###################################################################################################             
#%%

def pick_files(file_extension='*'):
    ''' Simple file selection function for quick selections.
    
    Parameters
    ----------
    file_extension: str, file extension specifier, for example *.mat
    
    Returns
    -------
    file_names: list, list of file names selected
    
    Examples
    --------
    file_names = pick_files('*.mat')
    '''
    
    from tkinter import Tk #For interactive selection, this part is only used to withdraw() the little selection window once the selection is done.
    import tkinter.filedialog as filedialog
       
    specifier = [(file_extension, file_extension)] #Prepare for the filedialog gui
    
    # Select the session directory
    Tk().withdraw() #Don't show the tiny confirmation window
    file_names = list(filedialog.askopenfilenames(filetypes = specifier))
    
    return file_names

    ###########################################################################
#%%

def align_behavioral_video(camlog_file):
    '''Function to extract video tracking information. Returns the name of the camera from the log file,
    the average interval between acquired frames and the frame indices of the trial starts.
     
    Parameters
    ----------
    camlog_file: The log file from the labcams acquisition. Make sure the data is stored in the specified folder structure
                 that is, the chipmunk folder contains the camlog file and the .mat or .obsmat file from the behavior.
    
    Retunrs
    -------
    trial_start_video_frames: The frames when the trials started
    
    camera_name: The camera identifier as recorded in the log file

    video_frame_interval: The average interval between video frames in seconds
                        
    Examples
    --------
    trial_start_video_frames, camera_name, video_frame_interval = align_behavioral_video(camlog_file)
    '''
    
    import numpy as np
    from labcams import parse_cam_log, unpackbits #Retrieve the correct channel and spot trial starts in frames
    from scipy.io import loadmat #Load behavioral data for ground truth on trial number
    from os import path
    from glob import glob
    
    #First load the associated chipmunk file and get the number of trials to find the correct channel on the camera
    chipmunk_folder = path.split(camlog_file)[0] #Split the path to retrieve the folder only in order to load the mat or obsmat file belonging to the movies.
    file_list = glob(chipmunk_folder + '/*.mat') #There is only one behavioral recording file in the folder
    if not file_list:
        file_list = glob(chipmunk_folder + '/*.obsmat')
        if not file_list:
            raise ValueError('No behavioral event file was found in the folder.')
            
    chipmunk_file = file_list[0]
    sesdata = loadmat(chipmunk_file, squeeze_me=True,
                                      struct_as_record=True)['SessionData']
    trial_num = int(sesdata['nTrials'])
    
    #Extract data from the log file
    logdata, comments = parse_cam_log(camlog_file) #Parse inputs and separate comments from frame data
    
    #Get the camera name
    spaces = [pos for pos, char in enumerate(comments[0]) if char == ' '] #Find all the empty spaces that separate the words
    camera_name = comments[0][spaces[1]+1: spaces[2]] #The name comes between the second and the third space
    
    #Get the video frame interval in s
    video_frame_interval = np.mean(np.diff(logdata['timestamp'])) #Compute average time between frames
    
    #Find trial start frames
    onsets, offsets = unpackbits(logdata['var2']) #The channel log is always unnamed and thus receives the key var2
    for k in onsets.keys():
        if onsets[k].shape[0] == trial_num + 1: # In most cases an unfinished trial will already have been started
            trial_start_video_frames = onsets[k][0 : trial_num]
        elif onsets[k].shape[0] == trial_num: # Sometimes the acquisition stops just after the end of the trial before the beginning of the next one
            trial_start_video_frames = onsets[k]
            
    if not 'trial_start_video_frames' in locals():
        raise ValueError('In none of the camera channels the onset number matched the trial number. Please check the log files and camera setup.')
    
    return trial_start_video_frames, camera_name, video_frame_interval
    
###############################################################################
#%%

def align_miniscope_data(caiman_file): 
    '''Function to align the acquired miniscope data to the behavior and interpolate
    dropped frames in the signal if necessary. Stores the results in '/analysis'
    inside the session directory and outputs the data as a dictionary. When loading
    the data from the file later use:
    miniscope_data = numpy.load(mscope_data_path, allow_pickle = True).tolist()
    
    Parameters
    ----------
    caiman_file: str, the path to the file with the extracted calcium imaging signals
                 using caiman. Make sure to have downloaded the .mscopelog and the 
                 timeStams.csv file in the miniscope folder and the chipmunk behavior
                 file in the chipmunk folder!
                 
    Returns
    -------
    miniscope_data: dict, all the necessay imaging data, timestamps and alignment
                    parameters. Use miniscope_data.keys() to explore the stored 
                    variables.
                    
    Examples
    --------
    miniscope_data = align_miniscope_data(caiman_file)
                 '''
    
    import numpy as np
    from scipy.io import loadmat
    import glob #Search files in directory
    import matplotlib.pyplot as plt
    import warnings #Throw warning for unmatched drops
    import os #To deal with the paths and directories
    from load_cnmfe_outputs import load_data as load_caiman #To get the processed imaging data
    from scipy.interpolate import CubicSpline #For signal interpolation
    from scipy.interpolate import interp1d
    
    #--------Sort out the location of the files-------------------------------
    session_directory = os.path.split(os.path.split(caiman_file)[0])[0] #Move up the folder hierarchy
    mscopelog_file = glob.glob(session_directory + '/miniscope/*.mscopelog')[0] #Only one such file in the directory
    mscope_tStamps_file = session_directory + '/miniscope/timeStamps.csv'
    chipmunk_file = glob.glob(session_directory + '/chipmunk/*.mat')
    if len(chipmunk_file) == 0: #This is the case when it is an obsmat file, for instance
        chipmunk_file = glob.glob(session_directory + '/chipmunk/*.obsmat')
        if  len(chipmunk_file) == 0: #Not copied
            print("It looks like the chipmunk behavior file has not yet been copied to this folder")
            
    
    #---------All the loading----------------------------------------------
    mscope_log = np.loadtxt(mscopelog_file, delimiter = ',', skiprows = 2) #Make sure to skip over the header lines and use comma as delimiter
    mscope_time_stamps = np.loadtxt(mscope_tStamps_file, delimiter = ',', skiprows = 1)
    
    temp = loadmat(chipmunk_file[0], squeeze_me=True, struct_as_record=True) #Load the behavioral data file 
    chipmunk_data = temp['SessionData']
    
    
    #---------Align trial start times for the miniscope---------------------
    # Remember that channel 23 is the Bpod input and channel 2 the miniscope frames,
    # first column is the channel number, second column the event number, third column
    # the teensy time stamp.
    
    #Make sure the teensy file starts with miniscope frames and not trial info
    if mscope_log[0, 0] == 23:
         raise ValueError(f"Possibly the behavior was started before the imaging. Please check {mscopelog_file}.")
    
    #In some of the log files large gaps were seen between the first and the second
    #catured frame. This happens when the logging start automatically already but then
    #the scope DAQ has to be unplugged from the computer to properly start the scope 
    #acquisition. Remove these instances.
    #If first two events are putative frame captures but the second one happens more than a second after the first there is a problem
    if ((mscope_log[0, 0] == 2) & (mscope_log[1, 0] == 2)) & (mscope_log[1,2] - mscope_log[0,2] > 1000): 
        mscope_log = mscope_log[1:mscope_log.shape[0],:] #Remove first timestamp that can't correspond to frame
        warnings.warn(f"The first putative frame in the mscopelog file occurs 1 s before the next one. It will be cut out to align the data. Please check {mscopelog_file}")
    
    trial_starts  = np.where(mscope_log[:,0] == 23) #Find all trial starts
    trial_start_frames = np.array(mscope_log[trial_starts[0]+1,1], dtype='int') #Set the frame just after
    # trial start as the start frame, the trial start takes place in the exposure time of the frame.
    trial_start_time_covered = np.array(mscope_log[trial_starts[0]+1,2] - mscope_log[trial_starts[0],2])
    # Keep the trial time that was covered by the frame acquisition
    # Check whether the number of trials is as expected
    trial_number_matches = trial_start_frames.shape[0] == (int(chipmunk_data['nTrials']) + 1)
    # Adding one to the number of trials is necessary becuase nTrials measures completed trials
    if not trial_number_matches: #For sanity check
        raise ValueError('Number of trials recorded on the log file does not match the one from the behavior file.')
    
    
    #--------Check for dropped frames-----------------------------
    teensy_frames_collected = np.where(mscope_log[:,0] == 2)[0]
    #These are the frames that the miniscope DAQ acknowledges as recorded.
    teensy_frames = np.transpose(mscope_log[teensy_frames_collected,2])
    teensy_frames = teensy_frames - teensy_frames[0] #Zero the start
    miniscope_frames = mscope_time_stamps[:,1] #These are the frames have been received by the computer and written to the HD.
    miniscope_frames = miniscope_frames - miniscope_frames[0]
    
    num_dropped = teensy_frames.shape[0] - miniscope_frames.shape[0]   
    
    
    #----Match the teensy frames with the actually recorded ones and detect position of dropped--------------------
    average_interval = np.mean(np.diff(teensy_frames, axis = 0), axis = 0)
    #Determine th expected interval between the frames from the teensy output.
    
    acquired_frame_num = miniscope_frames.shape[0]
    clock_time_difference = miniscope_frames - teensy_frames[0:acquired_frame_num]
    #Map the frame times 1:1 here first to already reveal unexpected jumps in the time difference
    clock_sync_slope = (miniscope_frames[-1] - teensy_frames[-1])/acquired_frame_num
    #This variable is a rough estimate of the slope of the desync between the CPU clock and the teensy clock.
    intercept_term = 0 #This is expected to be 0 in the initial segment of the line
    
    # Generate line and subtract from the data to obtain "rectified" version of the data
    line_estimate = (np.arange(0,acquired_frame_num) - 1) * clock_sync_slope + intercept_term
    residuals = clock_time_difference - line_estimate
    
    # Compute the first derivative of the residuals
    diff_residuals = np.diff(residuals,axis=0)
    
    #Find possible frame drop events
    #candidate_drops = [0] + np.array(np.where(diff_residuals > 0.5*average_interval))[0,:].tolist() + [residuals.shape[0]] #Mysteriously one gets a tuple of indices and zeros
    candidate_drops = [0] + np.array(np.where(diff_residuals > (average_interval - average_interval*0.1)))[0,:].tolist() + [residuals.shape[0]] #Mysteriously one gets a tuple of indices and zeros
    #Turn into list and add the 0 in front and the teensy signal in the end to make looping easier.
    
    frame_drop_event = []
    last_frame_drop = 0 #Keep a pointer to the last frame drop that occured to iteratively lengthen the window to average over
    #Go through the candidates and assign as an event if the following frames are
    #all shifted more than one expected frame duration (with 20% safety margin).
    #Whenever a candidate event does not qualify as a frame drop merge the residuals
    #after that event into the current evaluation process.
    for k in range(1,len(candidate_drops)-1):
        if  (np.mean(residuals[candidate_drops[k]+1:candidate_drops[k+1]]+1) - np.mean(residuals[last_frame_drop + 1:candidate_drops[k]]+1)) > (average_interval - (average_interval*0.2)):
        #if (np.mean(residuals[k+1:k + window]) - np.mean(residuals[k - window:k])) > (average_interval - (average_interval*0.1)):
            #Adding two here seeks to avoid some of the overshoot that can be seen when the computer is trying to catch up.
            frame_drop_event.append(candidate_drops[k])
            last_frame_drop = candidate_drops[k]
    
    #Estimate the number of frames dropped per 
    segments = [0] + frame_drop_event + [residuals.shape[0]]
    
    dropped_per_event = [None] * len(frame_drop_event) #Estimated dropped frames per event
    rounding_error = [None] * len(frame_drop_event) #Estimates confidence in precise number of dropped
    jump_size = [None] * len(frame_drop_event) #The estimated shift in time differencebetween the teensz
    # and the miniscope log when a frame is dropped
    
    for k in range(len(frame_drop_event)):
        jump_size[k] = np.mean(residuals[segments[k+1]+1:segments[k+2]+1]) - np.mean(residuals[segments[k]+1:segments[k+1]+1])
        
        #Divide the observed jump in the signal by the expected interval
        approx_loss_per_event = jump_size[k] / average_interval
        dropped_per_event[k] = round(approx_loss_per_event)
        if dropped_per_event[k] > 0:
            rounding_error[k] = (dropped_per_event[k] - approx_loss_per_event)/dropped_per_event[k]
        else:
            rounding_error[k] = dropped_per_event[k] - approx_loss_per_event
        
    #Verify the number of dropped frames
    if np.sum(dropped_per_event) == num_dropped:
        print(f"Matched {num_dropped} dropped frames in the miniscope recording")
        print("-----------------------------------------------------")
        
        if 0 in dropped_per_event: #Print an additional message if one of the drop events is empty
            warnings.warn("One or more empty drops have been detected. Please check all the outputs carefully")
    
    else: 
        raise ValueError("Dropped frames could not be localized inside the recording. Please check manualy.")
        
        
    #-----Plot summary of the matching-------------------------
    #plt.ioff() #This will make the plot pop up at the end of the execution
    fi = plt.figure()
    ax = fi.add_subplot(111)   
    ax.plot(residuals, label = "Rectified frame time difference")
    ax.scatter(frame_drop_event, residuals[frame_drop_event], s=100, facecolors='none', edgecolors='r', label = "Frame drop event")
    ax.legend()
    ax.set_title("Detected dropping of single or multiple frames")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Clock time difference between teensy and CPU (ms)")
    
    #------Load the caiman data and interpolate the frames that were dropped    
    A, C_short, S_short, F_short, image_dims, frame_rate, neuron_num, recording_length, movie_file, spatial_spMat = load_caiman(caiman_file)
    
    if num_dropped > 0: #The case with dropped frames
        #Generate a vector with the time stamps matching the acquired frames (leaky)
        time_vect = np.arange(acquired_frame_num + num_dropped) #The actual recording length.
        leaky_time = np.array(time_vect) #initialize a copy where time stamps will be removed
        for k in range(len(frame_drop_event)): #Iteratively remove the missing timestamps 
            if dropped_per_event[k] > 0:
                leaky_time = np.delete(leaky_time, np.arange(frame_drop_event[k],frame_drop_event[k] + dropped_per_event[k]))
                #Make sure to iterate to increasing indices, so that the correct position
                #is found
            
            #---Calculate the cubic spline on the denoised calcium signal and apply a 
            #---linear interpolation to the deconvolved spiking activity    
    
        cubic_spline_interpolation = CubicSpline(leaky_time, C_short, axis=1)
        C = cubic_spline_interpolation(time_vect)
    
        linear_interpolation = interp1d(leaky_time, S_short, axis=1)
        S = linear_interpolation(time_vect)
        
        #Handle cases where the detrending was not applied
        if F_short is not None:
           linear_interpolation = interp1d(leaky_time, F_short, axis=1)
           F = linear_interpolation(time_vect) 
        else:
           F = F_short
        
    else:
        C = C_short
        S = S_short
        F = F_short

    print(f'Successfully interpolated signals for {num_dropped} dropped frames')
    
    #-------Assemble a dictionary for the results
    miniscope_data = {'trial_starts': trial_start_frames, #
                          'num_dropped': num_dropped,
                          'frame_drop_event': frame_drop_event,
                          'dropped_per_event': dropped_per_event,
                          'jump_size_per_event': jump_size,
                          'rounding_error': rounding_error, 
                          'frame_interval': average_interval/1000, #Set to seconds
                          'acquired_frame_num': acquired_frame_num,
                          'trial_start_time_covered': trial_start_time_covered/1000, #Set to seconds
                          'A': A,
                          'C': C,
                          'S': S,
                          'F': F,
                          'neuron_num': neuron_num,
                          'frame_rate': frame_rate,
                          'image_dims': image_dims,
                          'recording_length': recording_length}
    
    #-------Save the obtained results to an easily python-readable file--------
    #Generate a new directory if needed
    if os.path.isdir(session_directory + "/analysis") == False:
        os.mkdir(session_directory + "/analysis")
        
    #Get some info on session date and animal id    
    temp = os.path.split(session_directory)
    session_date = temp[1]
    animalID = os.path.split(temp[0])[1]    
    
    #Set the output file name and save relevant variables
    output_file = session_directory + "/analysis/" + animalID + "_" + session_date + "_" + "miniscope_data.npy"    
    np.save(output_file, miniscope_data)
    
    return miniscope_data
        
#############################################################################
        
        
        
        
        