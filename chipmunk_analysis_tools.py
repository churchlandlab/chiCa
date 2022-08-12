# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:24:46 2022

@author: Lukas Oesch
"""

# def convert_behavior_sessions(directory_name, directory_structure = 'local'):
#     '''Convert the recorded behavioral .mat files to pandas data frames'''
    
    
#     from glob import glob
#     import os 
#     import pandas as pd
#     from scipy.io import loadmat
    
#     #%%----------Get the subjects inside the directory
#     subject_dirs = os.listdir(directory_name) #List all the directories and files here and check for directory later
    
#     #%%------Start looping through the subjects and list the .obsmat files
#     if directory_structure == 'local': #Where all the .mat files are losely floating inside the subject folder 
#         for subject in subject_dirs:
#             if os.path.isdir(directory_name + subject): #Verify that we look inside a folder
#                obsmat_list = glob(directory_name + subject + '/*.obsmat') #Retrieve all files with .obsmat extension
    
#     #%%------Loading and converting
 
#         try:
#             sesdata = loadmat(chipmunk_file[0], squeeze_me=True,
#                                       struct_as_record=True)['SessionData']
            
#             tmp = sesdata['RawEvents'].tolist()
#             tmp = tmp['Trial'].tolist()
#             uevents = np.unique(np.hstack([t['Events'].tolist().dtype.names for t in tmp])) #Make sure not to duplicate state definitions
#             ustates = np.unique(np.hstack([t['States'].tolist().dtype.names for t in tmp]))
#             trialevents = []
#             trialstates = [] #Extract all trial states and events
#             for t in tmp:
#                  a = {u:None for u in uevents}
#                  s = t['Events'].tolist()
#                  for b in s.dtype.names:
#                         a[b] = s[b].tolist()
#                  trialevents.append(a)
#                  a = {u:None for u in ustates}
#                  s = t['States'].tolist()
#                  for b in s.dtype.names:
#                          a[b] = s[b].tolist()
#                  trialstates.append(a)
#             trialstates = pd.DataFrame(trialstates)
#             trialevents = pd.DataFrame(trialevents)
#             trialdata = pd.merge(trialevents,trialstates,left_index=True, right_index=True)
            
#             #Add response and stimulus train related information: correct side, rate, event occurence time stamps
#             trialdata.insert(trialdata.shape[1], 'response_side', sesdata['ResponseSide'].tolist())
#             trialdata.insert(trialdata.shape[1], 'correct_side', sesdata['CorrectSide'].tolist())
            
#             #Get stim modality$
#             tmp_modality_numeric = sesdata['Modality'].tolist()
#             temp_modality = []
#             for t in tmp_modality_numeric:
#                 if t == 1:
#                     temp_modality.append('visual')
#                 elif t == 2: 
#                     temp_modality.append('auditory')
#                 elif t == 3:
#                     temp_modality.append('audio-visual')
#                 else: 
#                     temp_modality.append(np.nan)
#                     print('Could not determine modality and set value to nan')
                    
#             trialdata.insert(trialdata.shape[1], 'stimulus_modality', temp_modality)
            
#             #Reconstruct the time stamps for the individual stimuli
#             event_times = []
#             event_duration = sesdata['StimulusDuration'].tolist()[0]
#             for t in range(trialdata.shape[0]):
#                 temp_isi = sesdata['InterStimulusIntervalList'].tolist().tolist()[t][tmp_modality_numeric[t]-1]
#                 #Index into the corresponding trial and find the isi for the corresponding modality
                
#                 temp_trial_event_times = [temp_isi[0]] 
#                 for k in range(1,temp_isi.shape[0]-1): #Start at 1 because the first Isi is already the timestamp after the play stimulus
#                     temp_trial_event_times.append(temp_trial_event_times[k-1] + event_duration + temp_isi[k])
            
#                 event_times.append(temp_trial_event_times + trialdata['PlayStimulus'][t][0]) #Add the timestamp for play stimulus to the event time
            
#             trialdata.insert(trialdata.shape[1], 'stimulus_event_timestamps', event_times)
            
#             #Add the miniscope alignment parameters

#             trialdata.insert(0, 'trial_start_frame_index', trial_start_frames[0:len(tmp)]) #Exclude the last trial that has not been completed.
#             trialdata.insert(1, 'trial_start_time_covered', trial_start_time_covered[0:len(tmp)])
            
#             #Add alignment for the video tracking
#             for n in range(len(trial_start_video_frame)):
#                 trialdata.insert(2, camera_name[n]+ '_trial_start_index', trial_start_video_frame[n][0:len(tmp)])
#    return
#    #############################################################################
#%%

def convert_specified_behavior_sessions(file_names):
    '''converted_files = convert_specified_behavior_sessions(file_names)
    Convert a specified set of behavioral sessions to pandas data frames'''
    
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
                
                #Add a generic state tracking the timing of outcome presentation
                outcome_timing = []
                for k in range(trialdata.shape[0]):
                    if np.isnan(trialdata['DemonReward'][k][0]) == 0:
                        outcome_timing.append(np.array([trialdata['DemonReward'][k][0]]))
                    elif np.isnan(trialdata['DemonWrongChoice'][k][0]) == 0:
                        outcome_timing.append(np.array([trialdata['DemonWrongChoice'][k][0]]))
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
    '''file_names = pick_files(file_extension='*.mat')
    Simple file selection function for quick selections'''
    
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
            print('No behavioral event file was detected in the folder.')
            return
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
        print('In none of the camera channels the onset number matched the trial number. Please check the log files and camera setup.')
        return
    
    return trial_start_video_frames, camera_name, video_frame_interval
    
    
    
    
    
    
    
    