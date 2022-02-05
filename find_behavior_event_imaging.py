# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:08:16 2021

@author: Lukas Oesch
"""

from scipy.io import loadmat
import pandas as pd
import numpy as np
import random #To pick a random frame 
from tkinter import Tk #For interactive selection, this part is only used to withdraw() the little selection window once the selection is done.
import tkinter.filedialog as filedialog
import glob #To pick files of a specified type

#%%---------Select session folder and get the required files--------------
# Select the session directory
Tk().withdraw() #Don't show the tiny confirmation window
directory_name = filedialog.askdirectory()

#Load first the alignment and the interpolated traces
trial_alignment_files =  glob.glob(directory_name + '/trial_alignment/*.npz')
for k in trial_alignment_files:
    tmp_data = np.load(k)
    for key,val in tmp_data.items(): #Retrieve all the entries and create variables with the respective name, here, C and S and the average #interval between frames, average_interval, which is 1/framerate.
        exec(key + '=val')

chipmunk_file =  glob.glob(directory_name + '/chipmunk/*.mat')
if len(chipmunk_file) == 0: #This is the case when it is an obsmat file, for instance
    chipmunk_file = glob.glob(directory_name + '/chipmunk/*.obsmat')
    if  len(chipmunk_file) == 0: #Not copied
        print("It looks like the chipmunk behavior file has not yet been copied to this folder")

sesdata = loadmat(chipmunk_file[0], squeeze_me=True,
                          struct_as_record=True)['SessionData']

#%%----Shamelessly stolen from spatial sparrow dj utils
tmp = sesdata['RawEvents'].tolist()
tmp = tmp['Trial'].tolist()
uevents = np.unique(np.hstack([t['Events'].tolist().dtype.names for t in tmp])) #Make sure not to duplicate state definitions
ustates = np.unique(np.hstack([t['States'].tolist().dtype.names for t in tmp]))
trialevents = []
trialstates = [] #Extract all trial states and events
for t in tmp:
     a = {u:None for u in uevents}
     s = t['Events'].tolist()
     for b in s.dtype.names:
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

trialdata.insert(trialdata.shape[1], 'response_side', sesdata['ResponseSide'].tolist())
#Add the response side of the animal to the end of the data frame

#Add the miniscope alignment parameters
trialdata.insert(0, 'trial_start_frame_index', trial_start_frame_index[0:len(tmp)]) #Exclude the last trial that has not been completed.
trialdata.insert(1, 'trial_start_time_covered', trial_start_time_covered[0:len(tmp)])

#Add alignment for the video tracking
for n in range(len(trial_start_video_frame)):
    trialdata.insert(2, camera_name[n]+ '_trial_start_index', trial_start_video_frame[n][0:len(tmp)])

#%%------Check how much the recorded trial duration and the trial duration 
#        reconstructed from the imaging or video frame indicies deviate. If the
#        are more than one frame apart inform the user about the mismatch.

bpod_trial_dur = np.diff(np.array(sesdata['TrialStartTimestamp'].tolist())) #The duration of the trial as recorded by Bpod
approx_imaging_frame_time = np.zeros(len(trialdata)-1)
approx_video_frame_time = np.zeros(len(trialdata)-1)
for k in range(len(trialdata)-2):
    approx_imaging_frame_time[k] = (trialdata['trial_start_frame_index'][k+1] - trialdata['trial_start_frame_index'][k]) * (average_interval/1000)
    approx_video_frame_time[k] = (trialdata['trial_start_video_frame_index'][k+1] - trialdata['trial_start_video_frame_index'][k]) * average_video_frame_interval
    
imaging_time_difference = approx_imaging_frame_time - bpod_trial_dur
unexpected_imaging_time_diff = imaging_time_difference[imaging_time_difference > average_interval/1000] #Average interval is in ms
if unexpected_imaging_time_diff.shape[0]:
    print("There is a mismatch between the recorded trial duration \nand the expected time of trials from the imaging frames.")

video_time_difference = approx_video_frame_time - bpod_trial_dur
unexpected_video_time_diff = video_time_difference[video_time_difference > average_video_frame_interval] #Average interval is in ms
if unexpected_video_time_diff.shape[0]:
    print("Trial duration and reconstructed video frame time mismatch.\nChekc video alignment")

#%%----Extract the time stamps aligned to a certain task state

def find_state_start_frame_imaging(state_name, trialdata, average_interval, trial_start_time_covered):
    '''Locate the frame during which a certain state in the chipmunk task has
    started. Requires state_name (string with the name of the state of interest)
    trialdata (a pandas dataframe with the trial start frames
    and the state timers) and average interval (the average frame interval as 
    as recorded by teensy).'''
    
    state_start_frame = [None] * len(trialdata) #The frame that covers the start of
    state_time_covered = np.zeros([len(trialdata)]) #The of the side that has been covered by the frame
    
    for n in range(len(trialdata)): #Subtract one here because the last trial is unfinished
          if np.isnan(trialdata[state_name][n][0]) == 0: #The state has been visited
              frame_time = np.arange(trial_start_time_covered[n]/1000, trialdata['FinishTrial'][n][0] - trialdata['Sync'][n][0], average_interval/1000)
              #Generate frame times starting the first frame at the end of its coverage of trial inforamtion
              
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
          

#%%--------Extract time stamps aligned to a defined task state for video tracking

def find_state_start_frame_video(state_name, trialdata, average_video_frame_interval, camera_name):
    '''Locate the frame during which a certain state in the chipmunk task has
    started. Requires state_name (string with the name of the state of interest)
    trialdata (a pandas dataframe with the trial start frames
    and the state timers), average_video_frame_interval (the average frame interval as 
    as recorded by the FLIR camera of interest, in s and the name of the respective camera).
    Usage example:
    state_start_video_frame = find_state_start_frame_video(state_name, trialdata, average_video_frame_interval, camera_name)
    '''
    
    
    state_start_video_frame = [None] * len(trialdata) #The frame that covers the start of
    
    for n in range(len(trialdata)): #Subtract one here because the last trial is unfinished
          if np.isnan(trialdata[state_name][n][0]) == 0: #The state has been visited
              frame_time = np.arange(average_video_frame_interval, trialdata['FinishTrial'][n][0] - trialdata['Sync'][n][0], average_video_frame_interval)
              #Generate frame times starting the first frame at the end of its coverage of trial inforamtion

              tmp = frame_time - trialdata[state_name][n][0] #Calculate the time difference
              state_start_video_frame[n] = int(np.where(tmp > 0)[0][0] + trialdata[camera_name + "_trial_start_index"][n])
              #np.where returns a tuple where the first element are the indices that fulfill the condition.
              #Inside the array of indices retrieve the first one that is positive, therefore the first
              #frame that caputres some information.
         
          else:
              state_start_video_frame[n] = np.nan

    return state_start_video_frame
          


#%%---Experimental align signals to reward delivery

#First get rewarded trials
rewarded = [i for i, val in enumerate(sesdata['Rewarded'].tolist()) if val]

#Find the frame where the reard was delivered
reward_delivery_frame = [None] * len(rewarded)
random_frame_matched = np.empty((1000, len(rewarded))) #Randomly select a frame from within the same trial

for n in range(len(rewarded)):
    frame_time = np.arange(0,trialdata["trial_start_frame_index"][rewarded[n] + 1] - trialdata["trial_start_frame_index"][rewarded[n]]) * average_interval/1000
    #This line assumes that the trial just starts with the new frame, when in reality it has started somewhat earlier
    
    tmp = frame_time - trialdata["DemonReward"][rewarded[n]][0] #Calculate the time differenec
    reward_delivery_frame[n] = int(np.where(tmp > 0)[0][0] + trialdata["trial_start_frame_index"][rewarded[n]] + 1)
    #np.where returns a tuple where the first element are the indices that fulfill the condition.
    #Inside the array of indices retrieve the first one that is positive, therefore the first
    #frame where some reward delivery information can be found and add to the trial start frame.
    #Add 1 because the first frame time denotes the frame 0 that has not yet acquired trial
    #information only.
    # for q in range(1000):
    #     tmp = random.randrange(0, frame_time.shape[0])
    #     random_frame_matched[q,n] = int(tmp + trialdata["trial_start_frame_index"][rewarded[n]] + 1)
    # #Draw 1000 random frames from the same trial, might also be the rewarded one!

#Define the amount of data prior and after reward delivery to look at
window = 40 #Look at the second before and after reward

C_rew = np.zeros((window+1, len(reward_delivery_frame), C.shape[0] ))
S_rew = np.zeros((window+1, len(reward_delivery_frame), C.shape[0] ))


# C_rew_shuffled = np.zeros((window+1, len(rewarded), C.shape[0] ))
# S_rew_shuffled = np.zeros((window+1, len(rewarded), C.shape[0] ))

# tempC = np.empty((window+1, 1000)) #Represents all 1000 traces drawn with random center frame from one rewarded trial
# tempS = np.empty((window+1, 1000))

for k in range(C.shape[0]-1): #k for all the cells
    for n in range(len(reward_delivery_frame)-1): #n for all the rewards delivered
        C_rew[:,n,k] = C[k, int(reward_delivery_frame[n] - window/2) : int(reward_delivery_frame[n]  + window/2 + 1)]
        S_rew[:,n,k] = S[k, int(reward_delivery_frame[n] - window/2) : int(reward_delivery_frame[n]  + window/2 + 1)]
    
        # for q in range(1000): #q for all the shuffles
        #     tempC[:,q] =  C[k, int(random_frame_matched[q,n] - window/2) : int(random_frame_matched[q,n]  + window/2 + 1)]
        #     tempS[:,q] =  S[k, int(random_frame_matched[q,n] - window/2) : int(random_frame_matched[q,n]  + window/2 + 1)]
        # C_rew_shuffled[:,n,k] = np.mean(tempC, axis=1)
        # S_rew_shuffled[:,n,k] = np.mean(tempS, axis=1)
        
    print(f"Ran over cell number {k}")
        
#%%Check alignment
vect = np.arange(-1, 1.05, 0.05)

grand_average = np.mean(np.mean(C_rew, axis=1), axis=1)
standard_dev = np.std(np.mean(C_rew, axis=1), axis=1)
standard_error = standard_dev / C_rew.shape[2]

grand_average_shuffled = np.mean(np.mean(C_rew_shuffled, axis=1), axis=1)
standard_dev_shuffled = np.std(np.mean(C_rew_shuffled, axis=1), axis=1)
standard_error_shuffled = standard_dev_shuffled / C_rew.shape[2]

dAv = plt.figure()
plt.plot(vect, grand_average_shuffled, color=(0.3, 0.01, 0.5))
plt.fill_between(vect, grand_average_shuffled-standard_error_shuffled, grand_average_shuffled+standard_error_shuffled, color=(0.3, 0.01, 0.5), alpha=0.2)

plt.plot(vect, grand_average, color=(0.03, 0.2, 0.5))
plt.fill_between(vect, grand_average-standard_error, grand_average+standard_error, color=(0.03, 0.2, 0.5), alpha=0.2)

plt.title("Grand average fluorescence on reward delivery")
plt.xlabel("Time from reward delivery (s)")
plt.ylabel("Fluorescence (A.U.)")
plt.legend(['Random trial-matched frame', 'Reward delivery'])


cell_average = np.mean(C_rew, axis=1)
cAv = plt.figure()
for k in range(C.shape[0]-1):
    plt.plot(vect,cell_average[:,k], color=(0.6, 0.6, 0.6),linewidth=1)
plt.plot(vect,grand_average, color='k')
plt.title("Individual cell and grand average fluorescence on reward delivery")
plt.xlabel("Time from reward delivery (s)")
plt.ylabel("Fluorescence (A.U.)")


trial_average = np.mean(C_rew, axis=2)
cAv = plt.figure()
for k in range(len(reward_delivery_frame)-1):
    plt.plot(vect,trial_average[:,k], color=(0.6, 0.6, 0.6), linewidth = 1)
plt.plot(vect,grand_average, color='k')
plt.title("Individual cell and grand average fluorescence on reward delivery")
plt.xlabel("Time from reward delivery (s)")
plt.ylabel("Fluorescence (A.U.)")


#Look at S
grand_average_S = np.mean(np.mean(S_rew, axis=1), axis=1)
standard_dev_S = np.std(np.mean(S_rew, axis=1), axis=1)
standard_error_S = standard_dev / S_rew.shape[2]

grand_average_shuffled_S = np.mean(np.mean(S_rew_shuffled, axis=1), axis=1)
standard_dev_shuffled_S = np.std(np.mean(S_rew_shuffled, axis=1), axis=1)
standard_error_shuffled_S = standard_dev_shuffled / S_rew.shape[2]

dAv = plt.figure()
plt.plot(vect, grand_average_shuffled_S, color=(0.3, 0.01, 0.5))
plt.fill_between(vect, grand_average_shuffled_S-standard_error_shuffled_S, grand_average_shuffled_S+standard_error_shuffled_S, color=(0.3, 0.01, 0.5), alpha=0.2)

plt.plot(vect, grand_average_S, color=(0.03, 0.2, 0.5))
plt.fill_between(vect, grand_average_S-standard_error_S, grand_average_S +standard_error_S, color=(0.03, 0.2, 0.5), alpha=0.2)

plt.title("Grand average deconvlolved fluorescence on reward delivery")
plt.xlabel("Time from reward delivery (s)")
plt.ylabel("Deconvolved luorescence (A.U.)")
plt.legend(['Random trial-matched frame', 'Reward delivery'])


# #Visualization    
# import time

# fi = plt.figure()
# for k in range(C.shape[0]-1):
#     for n in range(len(reward_delivery_frame)-1):
#         plt.plot(C_rew[:,n,k], color=(0.6, 0.6, 0.6), linewidth=1)
#     plt.plot(np.mean(C_rew[:,:,k], axis=1), color='k')
    