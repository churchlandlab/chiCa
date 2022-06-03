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
                         if isinstance(s[b].tolist(), float):
                             #Make sure to include single values as an array with a dimension
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
                
                if 'ObsOutcomeRecord' in sesdata.dtype.fields:
                    trialdata.insert(1, 'observer_outcome_record', sesdata['ObsOutcomeRecord'].tolist())
                    tmp = sesdata['ObsActualWaitTime'].tolist()
                    trialdata.insert(trialdata.shape[1], 'observer_actual_wait_time' , tmp)
                
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