# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:24:46 2022

@author: Lukas Oesch
"""

#%%

def convert_specified_behavior_sessions(file_names, overwrite = False):
    '''Convert a list of specified behavior files from .mat or .obsmat to
    pandas data frames. The data frame is saved in the /Data dataset.
    
    Parameters
    ----------
    file_names: list, a list containing the paths of the files to convert
    overwrite: boolean, specifiy if you want to ovrwrite an exisiting version of the current file
    
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
            if not overwrite:
                print(f'File: {os.path.split(current_file)[1]} was skipped because a corresponding h5 file exists already.')
                print('---------------------------------------------------------------------------------------------------')
                converted_files.append(os.path.splitext(current_file)[0] + '.h5')
                continue #Move on to the next iteration if the file exists already and if it should not be overwritten
        #else:
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
            
            # Insert a column for DemonWrongChoice in trialda if necessary
            if 'DemonWrongChoice' not in trialdata.columns:
                trialdata.insert(trialdata.shape[1], 'DemonWrongChoice', [np.array([np.nan, np.nan])] * trialdata.shape[0])
            
            
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
                if tmp_modality_numeric[t] < 3: #Unisensory
                    temp_isi = sesdata['InterStimulusIntervalList'].tolist().tolist()[t][tmp_modality_numeric[t]-1]
                    #Index into the corresponding trial and find the isi for the corresponding modality
                else:
                    temp_isi = sesdata['InterStimulusIntervalList'].tolist().tolist()[t][0]
                    #For now assume synchronous and only look at visual stims
                    warnings.warn('Found multisensory trials, assumed synchronous condition')
                
                temp_trial_event_times = [temp_isi[0]] 
                for k in range(1,temp_isi.shape[0]-1): #Start at 1 because the first Isi is already the timestamp after the play stimulus
                    temp_trial_event_times.append(temp_trial_event_times[k-1] + event_duration + temp_isi[k])
            
                event_times.append(temp_trial_event_times + trialdata['PlayStimulus'][t][0]) #Add the timestamp for play stimulus to the event time
            
            trialdata.insert(trialdata.shape[1], 'stimulus_event_timestamps', event_times)
            
            #Insert the outcome record for faster access to the different trial outcomes
            trialdata.insert(0, 'outcome_record', sesdata['OutcomeRecord'].tolist())
            
            try: 
                tmp = sesdata['TrialDelays'].tolist()
                for key in tmp[0].dtype.fields.keys(): #Find all the keys and extract the data associated with them
                    tmp_delay = tmp[key].tolist()
                    trialdata.insert(trialdata.shape[1], key , tmp_delay)
            except:
                print('For this version of chipmunk the task delays struct was not implemented yet.\nDid not generate the respective columns in the data frame.')
                
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
                    outcome_timing.append(np.array([np.nan, np.nan]))
            trialdata.insert(trialdata.shape[1], 'outcome_presentation', outcome_timing)
            
            # Retrieve the flag for revised choices
            trialdata.insert(trialdata.shape[1], 'revise_choice_flag', np.ones(trialdata.shape[0], dtype = bool) * sesdata['ReviseChoiceFlag'].tolist())
            
            #Get the Bpod timestamps for the start of each new trial
            trialdata.insert(trialdata.shape[1], 'trial_start_time' , sesdata['TrialStartTimestamp'].tolist())
            
            #----Get the timestamp of when the mouse gets out of the response poke
            #Here, a minimum poke duration of 100 ms is a requirement. Coming out
            #of the response port after less than 100 ms is not considered a retraction
            #an will be ignored and the next Poke out event will be counted.
            #Note: The timestamps are always calculated with respect to the 
            #start of the trial during which the response happened, even if the 
            #mouse only retracted on the next trial (usually when rewarded!)
            #Also note that there is always a < 100 ms period where Bpod does
            #not run a state machine and thus doesn't log events. It is possible that 
            #that retraction events might be missed because of this interruption.
            #These missed events will be nan.
            response_port_out = []
            for k in range(trialdata.shape[0]):
                event_name = None
                if trialdata['response_side'][k] == 0:
                    event_name = 'Port1Out'
                elif trialdata['response_side'][k] == 1:
                    event_name = 'Port3Out'
                
                poke_ev = None
                if event_name is not None:
                    #First check current trial
                    add_past_trial_time = 0 #Time to be added if the retrieval only happens in the following trial
                    tmp = trialdata[event_name][k][trialdata[event_name][k] > trialdata['outcome_presentation'][k][0]]
                    
                    #Now sometimes the mice retract very fast, faster than the normal reaction time
                    #skip these events in this case and consider the next one
                    candidates = tmp.shape[0]
                    looper = 0
                    while (poke_ev is None) & (looper < candidates):
                        tmptmp = tmp[looper]
                        if tmptmp - trialdata['outcome_presentation'][k][0] > 0.1:
                            poke_ev = tmptmp
                        looper = looper + 1
                    
                    if poke_ev is None: #Did not find a poke out event that fullfills the criteria
                        if k < (trialdata.shape[0] - 1): # Stop from checking at the very end...
                            tmp = trialdata[event_name][k+1][trialdata[event_name][k+1] < trialdata['Port2In'][k+1][0]]
                            #Check all the candidate events at the beginning of the next trial but before center fixation.
                            add_past_trial_time = trialdata['trial_start_time'][k+1] - trialdata['trial_start_time'][k]
                            if tmp.shape[0] > 0:
                                poke_ev = np.min(tmp) #These would actually come sorted already...
                if poke_ev is not None:          
                    response_port_out.append(np.array([poke_ev + add_past_trial_time, poke_ev + add_past_trial_time]))
                else:
                    response_port_out.append(np.array([np.nan, np.nan]))

            trialdata.insert(trialdata.shape[1], 'response_port_out', response_port_out)
            
            outcome_end = []
            for k in range(trialdata.shape[0]):
                if np.isnan(trialdata['DemonReward'][k][0]) == 0:
                    outcome_end.append(np.array([trialdata['response_port_out'][k][0], trialdata['response_port_out'][k][0]]))
                elif np.isnan(trialdata['DemonWrongChoice'][k][0]) == 0:
                    outcome_end.append(np.array([trialdata['FinishTrial'][k][0],trialdata['FinishTrial'][k][0]]))
                else:
                    outcome_end.append(np.array([np.nan, np.nan]))
            trialdata.insert(trialdata.shape[1], 'outcome_end', outcome_end)
            
            if 'ObsOutcomeRecord' in sesdata.dtype.fields:
                trialdata.insert(1, 'observer_outcome_record', sesdata['ObsOutcomeRecord'].tolist())
                tmp = sesdata['ObsActualWaitTime'].tolist()
                trialdata.insert(trialdata.shape[1], 'observer_actual_wait_time' , tmp)
                tmp = sesdata['TrialSettings'].tolist()
                trialdata.insert(trialdata.shape[1], 'dobserver_ID' , tmp['obsID'].tolist())
       
            
            #Finally, verify the the number of trials
            
            
            #----Now the saving
            trialdata.to_hdf(os.path.splitext(current_file)[0] + '.h5', '/Data') #Save as hdf5
            converted_files.append(os.path.splitext(current_file)[0] + '.h5') #Keep record of the converted files
            
        except:
            warnings.warn(f"CUATON: An error occured and {current_file} could not be converted")
    return converted_files

   ###################################################################################################             
    
#%%
def load_trialdata(file_name):
    '''Read the behavioral data from .mat or .obsmat and return a
    pandas data frame with the respective data for each trial
    
    Parameters
    ----------
    file_name: string, path to a chipmunk .mat or .obsmat file

    
    Returns
    -------
    trialdata: pandas data frame, data about the timing of states and events, choices and outcomes etc.
    
    Examples
    --------
    trial_data = load_trialdata(file_name)
    '''
    
    import os 
    import pandas as pd
    from scipy.io import loadmat
    import numpy as np
    import warnings

    #---Start the loading
    try:
        
        #---------Do all the loading and conversion here
        sesdata = loadmat(file_name, squeeze_me=True,
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
        
        # Insert a column for DemonWrongChoice in trialda if necessary
        if 'DemonWrongChoice' not in trialdata.columns:
            trialdata.insert(trialdata.shape[1], 'DemonWrongChoice', [np.array([np.nan, np.nan])] * trialdata.shape[0])
        
        
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
            if tmp_modality_numeric[t] < 3: #Unisensory
                temp_isi = sesdata['InterStimulusIntervalList'].tolist().tolist()[t][tmp_modality_numeric[t]-1]
                #Index into the corresponding trial and find the isi for the corresponding modality
            else:
                temp_isi = sesdata['InterStimulusIntervalList'].tolist().tolist()[t][0]
                #For now assume synchronous and only look at visual stims
                warnings.warn('Found multisensory trials, assumed synchronous condition')
            
            temp_trial_event_times = [temp_isi[0]] 
            for k in range(1,temp_isi.shape[0]-1): #Start at 1 because the first Isi is already the timestamp after the play stimulus
                temp_trial_event_times.append(temp_trial_event_times[k-1] + event_duration + temp_isi[k])
        
            event_times.append(temp_trial_event_times + trialdata['PlayStimulus'][t][0]) #Add the timestamp for play stimulus to the event time
        
        trialdata.insert(trialdata.shape[1], 'stimulus_event_timestamps', event_times)
        
        #Insert the outcome record for faster access to the different trial outcomes
        trialdata.insert(0, 'outcome_record', sesdata['OutcomeRecord'].tolist())
        
        try: 
            tmp = sesdata['TrialDelays'].tolist()
            for key in tmp[0].dtype.fields.keys(): #Find all the keys and extract the data associated with them
                tmp_delay = tmp[key].tolist()
                trialdata.insert(trialdata.shape[1], key , tmp_delay)
        except:
            print('For this version of chipmunk the task delays struct was not implemented yet.\nDid not generate the respective columns in the data frame.')
            
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
                outcome_timing.append(np.array([np.nan, np.nan]))
        trialdata.insert(trialdata.shape[1], 'outcome_presentation', outcome_timing)
        
        # Retrieve the flag for revised choices
        trialdata.insert(trialdata.shape[1], 'revise_choice_flag', np.ones(trialdata.shape[0], dtype = bool) * sesdata['ReviseChoiceFlag'].tolist())
        
        #Get the Bpod timestamps for the start of each new trial
        trialdata.insert(trialdata.shape[1], 'trial_start_time' , sesdata['TrialStartTimestamp'].tolist())
        
        #----Get the timestamp of when the mouse gets out of the response poke
        #Here, a minimum poke duration of 100 ms is a requirement. Coming out
        #of the response port after less than 100 ms is not considered a retraction
        #an will be ignored and the next Poke out event will be counted.
        #Note: The timestamps are always calculated with respect to the 
        #start of the trial during which the response happened, even if the 
        #mouse only retracted on the next trial (usually when rewarded!)
        #Also note that there is always a < 100 ms period where Bpod does
        #not run a state machine and thus doesn't log events. It is possible that 
        #that retraction events might be missed because of this interruption.
        #These missed events will be nan.
        response_port_out = []
        for k in range(trialdata.shape[0]):
            event_name = None
            if trialdata['response_side'][k] == 0:
                event_name = 'Port1Out'
            elif trialdata['response_side'][k] == 1:
                event_name = 'Port3Out'
            
            poke_ev = None
            if event_name is not None:
                #First check current trial
                add_past_trial_time = 0 #Time to be added if the retrieval only happens in the following trial
                tmp = trialdata[event_name][k][trialdata[event_name][k] > trialdata['outcome_presentation'][k][0]]
                
                #Now sometimes the mice retract very fast, faster than the normal reaction time
                #skip these events in this case and consider the next one
                candidates = tmp.shape[0]
                looper = 0
                while (poke_ev is None) & (looper < candidates):
                    tmptmp = tmp[looper]
                    if tmptmp - trialdata['outcome_presentation'][k][0] > 0.1:
                        poke_ev = tmptmp
                    looper = looper + 1
                
                if poke_ev is None: #Did not find a poke out event that fullfills the criteria
                    if k < (trialdata.shape[0] - 1): # Stop from checking at the very end...
                        tmp = trialdata[event_name][k+1][trialdata[event_name][k+1] < trialdata['Port2In'][k+1][0]]
                        #Check all the candidate events at the beginning of the next trial but before center fixation.
                        add_past_trial_time = trialdata['trial_start_time'][k+1] - trialdata['trial_start_time'][k]
                        if tmp.shape[0] > 0:
                            poke_ev = np.min(tmp) #These would actually come sorted already...
            if poke_ev is not None:          
                response_port_out.append(np.array([poke_ev + add_past_trial_time, poke_ev + add_past_trial_time]))
            else:
                response_port_out.append(np.array([np.nan, np.nan]))

        trialdata.insert(trialdata.shape[1], 'response_port_out', response_port_out)
        
        outcome_end = []
        for k in range(trialdata.shape[0]):
            if np.isnan(trialdata['DemonReward'][k][0]) == 0:
                outcome_end.append(np.array([trialdata['response_port_out'][k][0], trialdata['response_port_out'][k][0]]))
            elif np.isnan(trialdata['DemonWrongChoice'][k][0]) == 0:
                outcome_end.append(np.array([trialdata['FinishTrial'][k][0],trialdata['FinishTrial'][k][0]]))
            else:
                outcome_end.append(np.array([np.nan, np.nan]))
        trialdata.insert(trialdata.shape[1], 'outcome_end', outcome_end)
        
        if 'ObsOutcomeRecord' in sesdata.dtype.fields:
            trialdata.insert(1, 'observer_outcome_record', sesdata['ObsOutcomeRecord'].tolist())
            tmp = sesdata['ObsActualWaitTime'].tolist()
            trialdata.insert(trialdata.shape[1], 'observer_actual_wait_time' , tmp)
            tmp = sesdata['TrialSettings'].tolist()
            trialdata.insert(trialdata.shape[1], 'dobserver_ID' , tmp['obsID'].tolist())
        
    except:
        warnings.warn(f"CUATON: An error occured and {file_name} could not be loaded")
        
    return trialdata

   ################################################################################################### 
#%%
def load_SpatialSparrow(file_name):
    '''Read the SpatialSparrow data from .mat or and return a
    pandas data frame with the respective data for each trial so that it is similar to the chipmunk data
    
    Parameters
    ----------
    file_name: string, path to a chipmunk .mat or .obsmat file

    
    Returns
    -------
    trialdata: pandas data frame, data about the timing of states and events, choices and outcomes etc.
    
    Examples
    --------
    trial_data = load_SpatialSparrow(file_name)
    '''
    
    import os 
    import pandas as pd
    from scipy.io import loadmat
    import numpy as np
    import warnings

    #---Start the loading
    trialdata = None
    try:
        
        sesdata = loadmat(file_name, squeeze_me=True,
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

        #Give the touch shaker columns more useful names
        ts_codes = ['TouchShaker1_1', 'TouchShaker1_2', 'TouchShaker1_3', 'TouchShaker1_4', 'TouchShaker1_5', 'TouchShaker1_6',
                    'TouchShaker1_7', 'TouchShaker1_8', 'TouchShaker1_9', 'TouchShaker1_14', 'TouchShaker1_15'] #The specific touch shaker codes
        ts_names = ['left_spout_touch', 'right_spout_touch', 'left_spout_release', 'right_spout_release',
                    'left_handle_touch', 'right_handle_touch', 'both_handle_touch', 'left_handle_release', 'right_handle_release',
                    'acknowledge', 'error']
        
        for col_name in trialdata.keys():
            if col_name in ts_codes:
                trialdata.rename(columns={col_name: ts_names[ts_codes.index(col_name)]}, inplace=True)

        #Also rename the trial start and end for convenience
        trialdata.rename(columns={'TrialStart': 'Sync', 'TrialEnd': 'FinishTrial'}, inplace=True)        

        #Add response and stimulus train related information: correct side, rate, event occurence time stamps
        trialdata.insert(trialdata.shape[1], 'response_side', np.array(sesdata['ResponseSide'].tolist())-1)
        trialdata.insert(trialdata.shape[1], 'correct_side', np.array(sesdata['CorrectSide'].tolist())-1)
        
        #Get stim modality
        tmp_modality_numeric = sesdata['StimType'].tolist() #Here this is called stim type
        temp_modality = []
        for t in tmp_modality_numeric:
            if t == 1:
                temp_modality.append('visual')
            elif t == 2: 
                temp_modality.append('auditory')
            else: 
                temp_modality.append(np.nan)
                print('Could not determine modality and set value to nan')
                
        trialdata.insert(trialdata.shape[1], 'stimulus_modality', temp_modality)
        
        #Here row 0 is the right side vs 1 is the left
        stim_strength = sesdata['StimSideValues'].tolist()[0,:].astype(int) - sesdata['StimSideValues'].tolist()[1,:].astype(int)
        trialdata.insert(trialdata.shape[1], 'stimulus_strength', stim_strength)

        #Get the timestamps for the stims on the left and right side, these are not exactly the same number as intended though!!
        event_times = []
        for t in range(trialdata.shape[0]):
            temp_evs = sesdata['stimEvents'].tolist()[t].tolist()
            event_times.append([temp_evs[1], temp_evs[0]]) #append these here like this because the sides were flipped. Index 0 in the original data means right 
        trialdata.insert(trialdata.shape[1], 'stimulus_event_timestamps', event_times)
        
                    
        # #Insert the outcome record for faster access to the different trial outcomes
        # trialdata.insert(0, 'outcome_record', sesdata['OutcomeRecord'].tolist())
        
        # try: 
        #     tmp = sesdata['TrialDelays'].tolist()
        #     for key in tmp[0].dtype.fields.keys(): #Find all the keys and extract the data associated with them
        #         tmp_delay = tmp[key].tolist()
        #         trialdata.insert(trialdata.shape[1], key , tmp_delay)
        # except:
        #     print('For this version of chipmunk the task delays struct was not implemented yet.\nDid not generate the respective columns in the data frame.')
            
        # tmp = sesdata['ActualWaitTime'].tolist()
        # trialdata.insert(trialdata.shape[1], 'actual_wait_time' , tmp)
        # #TEMPORARY: import the demonstrator and observer id
        # tmp = sesdata['TrialSettings'].tolist()
        # trialdata.insert(trialdata.shape[1], 'demonstrator_ID' , tmp['demonID'].tolist())
        
        #Add a generic state tracking the timing of outcome presentation, this is also a 1d array of two elements
        outcome_timing = []
        last_spout_out = []
        for k in range(trialdata.shape[0]):
            if np.isnan(trialdata['Reward'][k][0]) == 0:
                outcome_timing.append(np.array([trialdata['Reward'][k][0], trialdata['Reward'][k][0]]))
                last_spout_out.append(np.array([trialdata['FinishTrial'][k][0], trialdata['FinishTrial'][k][0]])) # This is now called finish trial because it got renamed above to be compliant with chipmunk
            elif np.isnan(trialdata['HardPunish'][k][0]) == 0:
                outcome_timing.append(np.array([trialdata['HardPunish'][k][0],trialdata['HardPunish'][k][0]]))
                last_spout_out.append(np.array([trialdata['HardPunish'][k][0], trialdata['HardPunish'][k][0]]))
            else:
                outcome_timing.append(np.array([np.nan, np.nan]))
                last_spout_out.append(np.array([np.nan, np.nan]))
        trialdata.insert(trialdata.shape[1], 'outcome_presentation', outcome_timing)
        trialdata.insert(trialdata.shape[1], 'last_spout_out', last_spout_out)
        # # Retrieve the flag for revised choices
        # trialdata.insert(trialdata.shape[1], 'revise_choice_flag', np.ones(trialdata.shape[0], dtype = bool) * sesdata['ReviseChoiceFlag'].tolist())
        
        #Get the Bpod timestamps for the start of each new trial
        trialdata.insert(trialdata.shape[1], 'trial_start_time' , sesdata['TrialStartTimestamp'].tolist())
        trialdata.insert(trialdata.shape[1], 'assisted_trial', sesdata['SingleSpout'].tolist())
        
        
        # #----Get the timestamp of when the mouse gets out of the response poke
        # #Here, a minimum poke duration of 100 ms is a requirement. Coming out
        # #of the response port after less than 100 ms is not considered a retraction
        # #an will be ignored and the next Poke out event will be counted.
        # #Note: The timestamps are always calculated with respect to the 
        # #start of the trial during which the response happened, even if the 
        # #mouse only retracted on the next trial (usually when rewarded!)
        # #Also note that there is always a < 100 ms period where Bpod does
        # #not run a state machine and thus doesn't log events. It is possible that 
        # #that retraction events might be missed because of this interruption.
        # #These missed events will be nan.
        # response_port_out = []
        # for k in range(trialdata.shape[0]):
        #     event_name = None
        #     if trialdata['response_side'][k] == 0:
        #         event_name = 'Port1Out'
        #     elif trialdata['response_side'][k] == 1:
        #         event_name = 'Port3Out'
            
        #     poke_ev = None
        #     if event_name is not None:
        #         #First check current trial
        #         add_past_trial_time = 0 #Time to be added if the retrieval only happens in the following trial
        #         tmp = trialdata[event_name][k][trialdata[event_name][k] > trialdata['outcome_presentation'][k][0]]
                
        #         #Now sometimes the mice retract very fast, faster than the normal reaction time
        #         #skip these events in this case and consider the next one
        #         candidates = tmp.shape[0]
        #         looper = 0
        #         while (poke_ev is None) & (looper < candidates):
        #             tmptmp = tmp[looper]
        #             if tmptmp - trialdata['outcome_presentation'][k][0] > 0.1:
        #                 poke_ev = tmptmp
        #             looper = looper + 1
                
        #         if poke_ev is None: #Did not find a poke out event that fullfills the criteria
        #             if k < (trialdata.shape[0] - 1): # Stop from checking at the very end...
        #                 tmp = trialdata[event_name][k+1][trialdata[event_name][k+1] < trialdata['Port2In'][k+1][0]]
        #                 #Check all the candidate events at the beginning of the next trial but before center fixation.
        #                 add_past_trial_time = trialdata['trial_start_time'][k+1] - trialdata['trial_start_time'][k]
        #                 if tmp.shape[0] > 0:
        #                     poke_ev = np.min(tmp) #These would actually come sorted already...
        #     if poke_ev is not None:          
        #         response_port_out.append(np.array([poke_ev + add_past_trial_time, poke_ev + add_past_trial_time]))
        #     else:
        #         response_port_out.append(np.array([np.nan, np.nan]))

        # trialdata.insert(trialdata.shape[1], 'response_port_out', response_port_out)
        
        
        # if 'ObsOutcomeRecord' in sesdata.dtype.fields:
        #     trialdata.insert(1, 'observer_outcome_record', sesdata['ObsOutcomeRecord'].tolist())
        #     tmp = sesdata['ObsActualWaitTime'].tolist()
        #     trialdata.insert(trialdata.shape[1], 'observer_actual_wait_time' , tmp)
        #     tmp = sesdata['TrialSettings'].tolist()
        #     trialdata.insert(trialdata.shape[1], 'dobserver_ID' , tmp['obsID'].tolist())
   
        
        # #Finally, verify the the number of trials
        
        
        # #----Now the saving
        # trialdata.to_hdf(os.path.splitext(current_file)[0] + '.h5', '/Data') #Save as hdf5
        # converted_files.append(os.path.splitext(current_file)[0] + '.h5') #Keep record of the converted files
        
    except:
        warnings.warn(f"CUATON: An error occured and {file_name} could not be converted")
    return trialdata

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

def pick_files_multi_session(data_type, file_extension, file_keyword = None):
    '''Tool to select specified files of a data type over a user selected set of sessions.
    This relies on the hierarchical Churchland lab data folder structure with:
    animal_name -> session_datetime -> data_type
    The function uses wxpython, please install wx before running.
       
    
    Parameters
    ----------
    data_type: str, the directory with the specific data type, for example chipumnk, caiman, etc.
    file_extension: str, file extension specifier, for example *.mat
    file_keyword: str, a pattern that should be detected inside the file name to
                  distinguish the desired files from other files with the same extension.
    
    Returns
    -------
    file_names: list, list of file names selected
    
    Examples
    --------
    file_names = pick_files_multi_session(data_type, file_extension, file_keyword)
    '''
    
    import os
    import wx #To build the selction app
    import wx.lib.agw.multidirdialog as MDD #For the selection of multiple directories
    import glob #To spot the files of interest

    #-----Start the selection app and let the user choose the session directory-------
    app = wx.App(0) #Start the app
    
    dlg = MDD.MultiDirDialog(None, title="Select sessions", defaultPath=os.getcwd(),
                             agwStyle=MDD.DD_MULTIPLE|MDD.DD_DIR_MUST_EXIST) #Dialog settings
    if dlg.ShowModal() != wx.ID_OK: #Show the dialog
        print("You Cancelled The Dialog!")
        dlg.Destroy()
        
    paths = dlg.GetPaths() #Retrieve the paths, they will show as windows paths in the IDE but they are actually correct!
    del  app #Clean up
    del dlg
    
    #-----
    file_names = []
    for p in paths: #Convert and search the directories
        search_path = os.path.join(p, data_type, file_extension) #Construct the search path to identify the desired files in the data_type directory
        tmp = glob.glob(search_path)
        
        if len(tmp) > 0: #Skip when no files found
            match_description = []
            if file_keyword is not None: #Check for a certain keyword inside the file name
                for candidate_file in tmp:
                    candidate_name = os.path.split(candidate_file)[1]
                    result = candidate_name.find(file_keyword)
                    if result > -1:
                        match_description.append(candidate_file)        
            else: #No key word is given
                if len(tmp) > 1:
                    raise ValueError('Multiple files of same datatype and with same extension were found. Please provide a search keyword.')
                else:
                    match_description.append(tmp[0])
            
            #Check if more than one file matches the description
            if len(match_description) > 1: #Multiple files were found
                    raise ValueError(f'More than one file matching the keyword were found in the following search path:\n{search_path}')    
            elif len(match_description) == 0:
                    pass #Just ignore if no file has been found
            else:
                    file_names.append(match_description[0])
    
    return file_names

##############################################################################
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
    video_alignment_data: dict with keys: camera_name (the labbel of the camera
                          perspective acquired), trial_starts (the frame, during
                          which the trial start), frame_interval (average interval
                          between frames, calculated from the entire timeseries)
                        
    Examples
    --------
    video_alignment_data = align_behavioral_video(camlog_file)
    '''
    
    import numpy as np
    from labcams import parse_cam_log, unpackbits #Retrieve the correct channel and spot trial starts in frames
    from scipy.io import loadmat #Load behavioral data for ground truth on trial number
    from scipy.stats import mode
    from os import path, makedirs
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
    tmp_interval = np.diff(logdata['timestamp'])
    if np.sum(tmp_interval > 2*np.mean(tmp_interval)) > 0: #Check whether the intervals make sense
        raise ValueError('There is at least one very unusual frame interval.\nPlease check the log file.')
    else:
        video_frame_interval = np.mean(np.diff(logdata['timestamp'])) #Compute average time between frames
    #TODO: Try except block to check using unpack bits and trial starts comments
    #Find trial start frames
    try:
        trial_start_video_frames = [] #Uuugly!!!
        onsets, offsets = unpackbits(logdata['var2']) #The channel log is always unnamed and thus receives the key var2
        for k in onsets.keys():
            if onsets[k].shape[0] == trial_num + 1: # In most cases an unfinished trial will already have been started
                trial_start_video_frames = onsets[k][0 : trial_num] + 1 #Onsets seems to pick the frame identity just before the real frame, as validated by checking the videos.
            elif onsets[k].shape[0] == trial_num: # Sometimes the acquisition stops just after the end of the trial before the beginning of the next one
                trial_start_video_frames = onsets[k] + 1
                
        if len(trial_start_video_frames) == 0:
            raise ValueError('In none of the camera channels the onset number matched the trial number. Please check the log files and camera setup.')
    except:
        print('The onset numbers did not match the expected trial number, trying to reconstruct missing onsets from comments...')
        
        bits = np.unique(logdata['var2'])
        bit_rank =  np.flip(np.argsort([np.sum(logdata['var2']==k) for k in bits]))
        low_bit = bits[bit_rank[0]] #Take most aboundant state for individual frame identifier
        high_bit = bits[bit_rank[1]] #Second most abundant should represent the ttl high during the sync state
        
        ttl_high = []
        trial_start_video_frames = []
        comment_idx = []
        trial_id = []
        for k in comments:
            if 'trial_start' in k:
               idx = int(k.split(',')[0][2:])
               if idx == -1:
                   break
               uu = np.unique(logdata['var2'][idx-2:idx+14])
               ttl_high.append(high_bit in uu)
               comment_idx.append(idx)
               if high_bit in uu:
                   #This is really tailored to some very irregular data and might not be a very general solution.
                   #Try to improve the stability of the synchronization...
                   looper = -2
                   success = False
                   while looper <= 14 and not success:
                       if (logdata['var2'][idx + looper] == high_bit) and (logdata['var2'][idx + looper - 1] != high_bit): #Any change to the high bit 
                           trial_start_video_frames.append(idx + looper)
                           trial_id.append(int(k.split(':')[1]))
                           success = True
                       looper = looper + 1
                   if not success:
                       trial_start_video_frames.append(np.nan)
                       trial_id.append(int(k.split(':')[1]))
               else:
                     trial_start_video_frames.append(np.nan)
        ttl_high = np.array(ttl_high)
        trial_start_video_frames = np.array(trial_start_video_frames)
        comment_idx = np.array(comment_idx)
        #Infer the actual trial start from the most frequent delay
        expected_delay = mode(trial_start_video_frames - comment_idx)[0][0]
        trial_start_video_frames[np.where(ttl_high==0)[0]] = comment_idx[np.where(ttl_high==0)[0]] + expected_delay
        trial_start_video_frames = np.array(trial_start_video_frames)
        
        print(f'Inferred the trial start for the following trials: {np.where(ttl_high==0)[0]}')
        if not ((np.sum(np.isnan(trial_start_video_frames))==0) & ((trial_start_video_frames.shape[0]==trial_num) or (trial_start_video_frames.shape[0]==trial_num-1))): #Complicated expression
            raise ValueError('Could not identify or reconstruct the trial onsets from the camlog file. Please check the log files and camera setup.')
    #Check if the proper directory exists to store the data
    directory = path.join(path.split(camlog_file)[0],'..', 'analysis')
    
    if not path.exists(directory):
        makedirs(directory)
        print(f"Directory {directory} created")
    else:
        print(f"Directory {directory} already exists")
    
    video_alignment_data = dict({'camera_name': camera_name, 'trial_starts': trial_start_video_frames.astype(int), 'frame_interval': video_frame_interval})
    
    output_name = path.splitext(path.split(camlog_file)[1])[0]
    np.save(path.join(path.split(camlog_file)[0],'..', 'analysis', output_name + '_video_alignment.npy'), video_alignment_data)
    print(f'Video alignment file created for {camlog_file}!')  

    return video_alignment_data
    
###############################################################################
#%%
def quaternion_to_euler(qw, qx, qy, qz, coax_position = 'l'):
    '''Convert the quaternions from the miniscope BNO to euler angles.
    Here the first axis of rotation is in the direction of the coax - ewl PCBs, 
    the second one along the MCU pcb axis, and the third one along the miniscope
    body's hight. Rotation around this last axis are always the yaw. For the two 
    first axis whether they represent the pitch or roll will be determined by
    how the miniscope attaches to the baseplate. Here, the default case considered
    is when the coax cable is located on the left side of the animal's head. In this 
    case rotations around the first axis represent the pitch and rotations around the
    second axis the roll. This function maps all possible placements to this reference
    configuration resulting in the following:
        
    Looking from the left side of the animal, counter-clockwise rotations yield
    positive increases pitch angles,
    Looking from the back of the animal, counter-clockwise rotations yield
    positive increases in roll angles
    Looking from the top of the animal, counter-clockwise rotations yield
    positive increases in yaw angles (from the bottom counter-clockwise rotations will be negative!)
    
    Parameters
    ----------
    qw, qx, qy, qz: Scalar value sconstituting the quaternion at time t
                    as obtained from the head orientation tracking file.
    coax_position: string, letter indicating where the coax is located with
                   respect to the animal's head ('l' = left, 'r' = right, 'f' = front, 'b' = back),
                   defalut is 'l'

    Retunrs
    -------
    P, R, Y: Pitch, Roll and Yaw in radians
    
    Examples
    --------
    P, R, Y = quaternion_to_euler(qw, qx, qy, qz, coax_position = 'b')
    ---------------------------------------------------------------------
    '''
    import math
    m00 = 1.0 - 2.0*qy*qy - 2.0*qz*qz
    m01 = 2.0*qx*qy + 2.0*qz*qw
    m02 = 2.0*qx*qz - 2.0*qy*qw
    m10 = 2.0*qx*qy - 2.0*qz*qw
    m11 = 1 - 2.0*qx*qx - 2.0*qz*qz
    m12 = 2.0*qy*qz + 2.0*qx*qw
    m20 = 2.0*qx*qz + 2.0*qy*qw
    m21 = 2.0*qy*qz - 2.0*qx*qw
    m22 = 1.0 - 2.0*qx*qx - 2.0*qy*qy
    
    temp_P = math.atan2(m12, m22)
    c2 = math.sqrt(m00*m00 + m01*m01)
    temp_Y = math.atan2(-m02, c2)
    s1 = math.sin(temp_P)
    c1 = math.cos(temp_P)
    temp_R = math.atan2(s1*m20 - c1*m10, c1*m11-s1*m21)
    
    if coax_position == 'l':
        Y = temp_Y
        R = temp_R
        P = temp_P
    elif coax_position == 'r': #If the coax is on the other side, flip the sign
        Y = -temp_Y
        R = -temp_R
        P = temp_P
    elif coax_position == 'b': #If the coax points backwards
        P = -temp_Y
        R = temp_P
        Y = temp_R
    elif coax_position == 'f': #Coax sits in front
       print('Not yet implemented')
        
    return P, R, Y        

###############################################################################
#%%

def align_miniscope_data(caiman_file, coax_position = None): 
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
                 
    coax_position: str, the positionin of the miniscope coaxial cable on the mouse
                   head, 'l' = left, 'r' = right, 'b' = back, 'f' = front. This 
                   is used to retrieve the head orientation data and make sure
                   the rotational axes are as expected.
                 
    Returns
    -------
    miniscope_data: dict, all the necessay imaging data, timestamps and alignment
                    parameters. Use miniscope_data.keys() to explore the stored 
                    variables.
                    
    Examples
    --------
    miniscope_data = align_miniscope_data(caiman_file, coax_position = 'l')
                 '''
    
    import numpy as np
    from scipy.io import loadmat
    import pandas as pd #Reads faster than numpy!
    import glob #Search files in directory
    import matplotlib.pyplot as plt
    import warnings #Throw warning for unmatched drops
    import os #To deal with the paths and directories
    from scipy.interpolate import CubicSpline #For signal interpolation
    from scipy.interpolate import interp1d
    import sys
    sys.path.append('/home/lukas/code/chiCa/chiCa/')
    from load_cnmfe_outputs import load_data as load_caiman #To get the processed imaging data
    
    #--------Sort out the location of the files-------------------------------
    session_directory = os.path.split(os.path.split(caiman_file)[0])[0] #Move up the folder hierarchy
    mscopelog_file = glob.glob(session_directory + '/miniscope/*.mscopelog')[0] #Only one such file in the directory
    mscope_tStamps_file = session_directory + '/miniscope/timeStamps.csv'
    mscope_head_ori_file = session_directory + '/miniscope/headOrientation.csv'

    chipmunk_file = glob.glob(session_directory + '/chipmunk/*.mat')
    if len(chipmunk_file) == 0: #This is the case when it is an obsmat file, for instance
        chipmunk_file = glob.glob(session_directory + '/chipmunk/*.obsmat')
    if  len(chipmunk_file) == 0: #Maybe it is actually spotlight...
        chipmunk_file = glob.glob(session_directory + '/spotlight/*.mat')
        print('No chipmunk folder found, looking for spotlight instead...')
    if  len(chipmunk_file) == 0: #Could still be SpatialSparrow...
        chipmunk_file = glob.glob(session_directory + '/SpatialSparrow/*.mat')
        print("Nope, let's check for SpatialSparrow...")    
    if  len(chipmunk_file) == 0: #Not copied
        print("It looks like the behavior file has not yet been copied to this folder")
            
    
    #---------All the loading----------------------------------------------
    mscope_log = np.loadtxt(mscopelog_file, delimiter = ',', skiprows = 2) #Make sure to skip over the header lines and use comma as delimiter
    mscope_time_stamps = pd.read_csv(mscope_tStamps_file).to_numpy()
    head_ori = pd.read_csv(mscope_head_ori_file).to_numpy() #Load head orientation and drop
    
    
    #Sanity check on the timestamps file
    if coax_position is not None:
        include_head_orientation = True
    else:
        include_head_orientation = False
        print('Specifiy the position of the coaxial cable of the miniscope to retrieve head orientation data')
        
    if  np.where((head_ori[:,1:] == np.array([0,0,0,0])).all(-1))[0].shape[0] > 0:
        #If all of the head orientation entries are zero at a given moment there is a 
        #problem and the file cannot be used
        include_head_orientation = False
        print('The head orientation sensor seems to have crashed during the recording')
        print('The head orientation data was not included.')
    
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
    trial_number_matches = trial_start_frames.shape[0] == chipmunk_data['TrialStartTimestamp'].tolist().shape[0] + 1
    #Compare number of recorded start times in the miniscope log with recorded trial start from bpod
   
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
    
    #--------Initialize and retrieve the euler angles if requested
    if include_head_orientation:
        P_short = np.zeros([C_short.shape[1]]) * np.nan #pitch
        R_short = np.zeros([C_short.shape[1]]) * np.nan #roll
        Y_short = np.zeros([C_short.shape[1]]) * np.nan #yaw
        
        for k in range(P_short.shape[0]):
            P_short[k], R_short[k], Y_short[k] = quaternion_to_euler(head_ori[k,1],head_ori[k,2],head_ori[k,3],head_ori[k,4], coax_position = coax_position)
    
    #-----Interpolation of the data to correct for dropped frames
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
            
        if include_head_orientation:
            if P_short.shape[0] == leaky_time.shape[0]: 
                cubic_spline_interpolation = CubicSpline(leaky_time, P_short, axis=0)
                pitch = cubic_spline_interpolation(time_vect)
                
                cubic_spline_interpolation = CubicSpline(leaky_time, R_short, axis=0)
                roll = cubic_spline_interpolation(time_vect)
                
                cubic_spline_interpolation = CubicSpline(leaky_time, Y_short, axis=0)
                yaw = cubic_spline_interpolation(time_vect)
                
                #Ensure that the interpolation doesn't violate the possible range of angles
                pitch[pitch > np.pi] = np.pi
                pitch[pitch < -np.pi] = -np.pi
                
                roll[roll > np.pi] = np.pi
                roll[roll < -np.pi] = -np.pi
                
                yaw[yaw > np.pi] = np.pi
                yaw[yaw < -np.pi] = -np.pi    
            else:
                 pitch = None
                 roll = None
                 yaw = None
                 print('The head orientation data drop events do not correspond to the miniscope frame drops')
                 print('No head orientation data was saved')
           
        
    else:
        C = C_short
        S = S_short
        F = F_short
        pitch = P_short
        roll = R_short
        yaw = Y_short

    print(f'Interpolated signals for {num_dropped} dropped frames')
    
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
                          'pitch': pitch,
                          'roll': roll,
                          'yaw': yaw,
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
#%%

def get_experienced_stimulus_events(trialdata, stim_modalities = ['visual', 'auditory', 'audio-visual']):
    '''Retrieve a list of timestamps for the stimulus events experienced by the animal.
    Note: When the extended stim time > 0 there might be stimulus events after
    the 1s of the stim train and the number of experienced stim events might not match
    the intended stim rate.
    
    Paramters
    ---------
    trialdata: pandas data frame, the behavioral data for this session
    stim_modalities: list of strings, the modalities of the stimuli to consider,
                     default is are all.
                     
    Returns
    -------
    t_stamps: list of numpy arrays, experienced sensory stimulus events. The length
              of the list is the number of trials, any array containing a nan
              at the first position represents an early withdrawal before stimulus
              onset.
              
    ---------------------------------------------------------------------------
    '''
    
    import numpy as np
    import pandas as pd
    
    t_stamps = trialdata['stimulus_event_timestamps'].tolist()
    
    #Check if for some or all the trials an exended stimulus time was given
    for k in range(len(t_stamps)):
        ext_t = trialdata['ExtraStimulusDuration'][k] #How much time on top of the original 1 s stim train was given?
        #Make sure you only include stimuli that were actually played. In the case of 
        #an early withdrawal for example some stimulus events might be missing...
        if np.isnan(trialdata['DemonEarlyWithdrawal'][k][0]):  #Trials where there is NO early withdrawal
            #Now also check if extended stimulus time was given, in which case the
            #stim train was simply repeated after 1 s
            if ext_t > 0:
                onset = trialdata['PlayStimulus'][k][0]
                include = (t_stamps[k] - onset < ext_t) & (t_stamps[k] - onset + 1 < trialdata['DemonWaitForResponse'][k][1] - onset)
                #Include timestamps that are within the bounds of the exentended stim time and that were seen
                if include.shape[0] > 0:
                    t_stamps[k] = np.hstack((t_stamps[k], t_stamps[k][include] + 1)) #Add the 1 s of the actual stim train
                
        elif np.isnan(trialdata['DemonEarlyWithdrawal'][k][0]) == 0:  #Trials with early withdrawal
               if np.isnan(trialdata['PlayStimulus'][k][0]) == 0: #Stim will not be played if the mouse withdraws before
                    onset = trialdata['PlayStimulus'][k][0]
                    include = t_stamps[k] - onset < trialdata['DemonEarlyWithdrawal'][k][1] - onset
                    if include.shape[0] > 0:
                        t_stamps[k] = t_stamps[k][include]
                    else:
                        t_stamps[k] = np.array([np.nan])
                    #Now make sure to capture stims during the extended stim period.
                    #Note: If t_stamps[k] has already be truncated because the early withdrawal 
                    #happened before the one second mark no other ones will be included below.
                    if ext_t > 0:
                        include = t_stamps[k] - onset + 1 < trialdata['DemonEarlyWithdrawal'][k][1] - onset
                        #Include timestamps that are within the bounds of the exentended stim time and that were seen
                        if include.shape[0] > 0:    
                            t_stamps[k] = np.hstack((t_stamps[k], t_stamps[k][include] + 1)) #Add the 1 s of the actual stim train
    
    

    #Now go through the trials and set stim times to nan where the modalities don't match
    for k in range(len(t_stamps)):
        if trialdata['stimulus_modality'][k] not in stim_modalities:
            t_stamps[k] = np.array([np.nan])

    return t_stamps
    
#------------------------------------------------------------------------------
#%%-----Compute trial history and valid trials, etc.
def get_chipmunk_behavior(session_dir):
    '''Retrieve animal behavior information including previous choices and
    outcomes, stim strengths, easy stims, etc.
    '''
    
    import numpy as np
    import pandas as pd
    from glob import glob
    import os
    from chiCa import load_trialdata, determine_prior_variable
    
    #Load the trialdata
    if isinstance(session_dir, str):
        if len(glob(session_dir + '/chipmunk/*.h5')) == 1:
            trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
        elif len(glob(session_dir + '/chipmunk/*.mat')) == 1:
            trialdata = load_trialdata(glob(session_dir + '/chipmunk/*.mat')[0])
        else:
            raise RuntimeError(f"Can't find a behavioral file in {session_dir}")
    # elif isinstance(data_source, pd.DataFrame):
    #     trialdata = data_source
     
    #Get the data   
    out_dict = dict()
    out_dict['choice']  = np.array(trialdata['response_side'])
    out_dict['category'] = np.array(trialdata['correct_side'])
    out_dict['prior_choice'] =  determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
    out_dict['prior_category'] =  determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')
    
    out_dict['outcome'] = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained later
    out_dict['outcome'][np.array(np.isnan(trialdata['response_side']))] = np.nan
    out_dict['prior_outcome'] =  determine_prior_variable(out_dict['outcome'], np.ones(len(trialdata)), 1, 'consecutive')
       
    modality = np.zeros([trialdata.shape[0]])
    modality[trialdata['stimulus_modality'] == 'auditory'] = 1
    modality[trialdata['stimulus_modality'] == 'audio-visual'] = 2
    out_dict['modality'] = modality
    out_dict['prior_modality'] = determine_prior_variable(modality, np.ones(len(trialdata)), 1, 'consecutive')
    
    #Find stimulus strengths
    tmp_stim_strengths = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
    for k in range(trialdata.shape[0]):
        tmp_stim_strengths[k] = trialdata['stimulus_event_timestamps'][k].shape[0]
    out_dict['stim_strengths'] = tmp_stim_strengths
    
    #Get category boundary to normalize the stim strengths
    unique_freq = np.unique(tmp_stim_strengths)
    category_boundary = (np.min(unique_freq) + np.max(unique_freq))/2
    stim_strengths = (tmp_stim_strengths- category_boundary) / (np.max(unique_freq) - category_boundary)
    if trialdata['stimulus_modality'][0] == 'auditory':
        stim_strengths = stim_strengths * -1
    out_dict['relative_stim_strengths'] = stim_strengths
    out_dict['easy_stim'] = np.abs(stim_strengths) == 1 #Either extreme
    
    out_dict['all_valid'] = np.isnan(out_dict['choice'])==0
    out_dict['valid_past'] = (np.isnan(out_dict['choice'])==0) & (np.isnan(out_dict['prior_choice'])==0)
    
    out_dict['choice_two_back'] =  determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 2, 'consecutive')
    out_dict['outcome_two_back'] =  determine_prior_variable(out_dict['outcome'], np.ones(len(trialdata)), 2, 'consecutive')
    out_dict['category_two_back'] =  determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 2, 'consecutive')
    
    out_dict['valid_two_back'] = ((np.isnan(out_dict['choice'])==0) & (np.isnan(out_dict['prior_choice'])==0)) & (np.isnan(out_dict['choice_two_back'])==0)
    
    out_dict['session'] = [os.path.split(session_dir)[1]] * trialdata.shape[0]
    return pd.DataFrame(out_dict)
    
#-----------------------------------------
#%%
def diagnose_performance(subject):
    '''Get some performance metrics on the level of the subject and on individual
    sessions to decide whether to keep this subject and include specific sessions.
    
    ---------------------------------------------------------------------------'''
    
    from chiCa import convert_specified_behavior_sessions
    import pandas as pd
    from labdatatools import get_labdata_preferences
    import os
    import subprocess
    from glob import glob
    from time import time
    import numpy as np
    
    datatype = 'chipmunk'
    
    start = time()
    #Get a list of all the chipmunk sessions for the subject
    #res = subprocess.check_output(f'labdata sessions {subject} -f {datatype}').decode().split("\n") #List all chipmunk sessions for this subject
    res = subprocess.check_output(['labdata', 'sessions', subject, '-f', datatype]).decode().split("\n")
    ses_list = []
    for x in res:
        if len(x) > 0:
            if x[0]==" " and x[1:2]!="\t":
                ses_list.append(x[1:])
    
    #Go through the sessions, copy and convert if session is not dowloaded yet
    #and extract some metrics
    metrics = {'session': [],
               'corrupted': [],
               'modality': [],
               'stim_strengths': [],
               'revise_choice': [],
               'wait_time': [],
               'post_stim_delay': [],
               'completed_trials': [],
               'early_withdrawal_rate': [],
               'performance_easy': [],
               'extended_stim': []}
    
    
    #Locate the data in folder
    base_dir = get_labdata_preferences()['paths'][0]
    for session in ses_list:
        metrics['session'].append(session)
        if len(glob(os.path.join(base_dir, subject, session, datatype, '*.mat'))) == 0: #Check if chipmunk file is there
            subprocess.run(['labdata', 'get', subject, '-s', session, '-d', datatype, '-i', "*.mat"])
        metrics['corrupted'].append(False) #Assume things are find with this mat file
        if len(glob(os.path.join(base_dir, subject, session, datatype, '*.h5'))) == 0: #Check if converted h5 file is present
            try:
                _ = convert_specified_behavior_sessions([glob(os.path.join(base_dir, subject, session, datatype, '*.mat'))[0]])
            except:
                pass #Use pass here because if a matfile exists but there is some issue with the conversion it only throws a warning
        if len(glob(os.path.join(base_dir, subject, session, datatype, '*.h5'))) == 0: #Check again whether a file exists now
                metrics['corrupted'][-1] = True
                metrics['modality'].append(None)
                metrics['stim_strengths'].append([None])
                metrics['revise_choice'].append(None)
                metrics['wait_time'].append(None)
                metrics['post_stim_delay'].append(None)
                metrics['completed_trials'].append(None)
                metrics['early_withdrawal_rate'].append(None)
                metrics['performance_easy'].append(None)
                metrics['extended_stim'].append(None)
                continue #Skip to the next iteration if the file is broken
        
        #Extract some other metrics
        trialdata = pd.read_hdf(glob(os.path.join(base_dir, subject, session, datatype, '*.h5'))[0])
        metrics['revise_choice'].append(trialdata['revise_choice_flag'][0]) #Were animals allowed to change their mind?
        
        mods = np.unique(trialdata['stimulus_modality'].tolist())
        if mods.shape[0] == 1:
            metrics['modality'].append(mods[0])
        elif mods.shape[0] > 1:
            metrics['modality'].append('mixed')
        
        stim_rate = [x.shape[0] for x in trialdata['stimulus_event_timestamps']]
        metrics['stim_strengths'].append(np.unique(stim_rate))
        try:
            metrics['wait_time'].append(np.mean(trialdata['waitTime'])) #This is the time animals are required to wait
        except:
            metrics['wait_time'].append(np.mean(trialdata['actual_wait_time']))
        try:
            metrics['post_stim_delay'].append(np.mean(trialdata['postStimDelay'])>0) #This is true when a random delay was added
        except:
            metrics['post_stim_delay'].append(False) #If it is not specified rely on the wait time criterion alone.
        metrics['completed_trials'].append(np.sum(np.isnan(trialdata['response_side'])==0)) #Number of completed trials
        metrics['early_withdrawal_rate'].append(np.sum(trialdata['outcome_record']==-1)/trialdata.shape[0]) #Early withdrawal rate
        easy_str = [np.min(np.unique(stim_rate)), np.max(np.unique(stim_rate))]
        consider = ((stim_rate == easy_str[0]) | (stim_rate == easy_str[1])) & (np.isnan(trialdata['response_side'])==0)
        metrics['performance_easy'].append(np.mean(trialdata['outcome_record'][consider]))
        try:
            metrics['extended_stim'].append(np.mean(trialdata['ExtraStimulusDuration']))
        except:
            metrics['extended_stim'].append(0) #Be conservative here and assume no extended stim if it can't be found
    session_metrics = pd.DataFrame(metrics)
    print(f'Computed session metrics for {subject} in {time() - start} seconds.')
    print('-------------------------------------------------------------------')
    return session_metrics

#-------------------------------------------------------------------------------