# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:08:16 2021

@author: Lukas Oesch
"""

from scipy.io import loadmat
import pandas as pd
import numpy as np
import random #To pick a random frame 

#%%

chipmunk_file_name = "C:/Users/Lukas Oesch/Documents/ChurchlandLab/TestDataChipmunk/TestMiniscopeAlignment/LO012/20210824_112833/chipmunk/LO012_chipmunk_DemonstratorVisTaskPaced_20210824_112833.mat"

sesdata = loadmat(chipmunk_file_name, squeeze_me=True,
                          struct_as_record=True)['SessionData']

#%%----Shamelessly stolen from spatial sparrow dj utils
tmp = sesdata['RawEvents'].tolist()
tmp = tmp['Trial'].tolist()
uevents = np.unique(np.hstack([t['Events'].tolist().dtype.names for t in tmp]))
ustates = np.unique(np.hstack([t['States'].tolist().dtype.names for t in tmp]))
trialevents = []
trialstates = []
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

trial_start_frame_index = trial_start_frames[0:len(tmp)] #Because an incomplete trial is started
trialdata.insert(0, 'trial_start_frame_index', trial_start_frame_index)

#%%---------Some checking

bpod_trial_dur = np.diff(np.array(sesdata['TrialStartTimestamp'].tolist()))
approx_frame_time = np.zeros(len(trialdata)-1)
for k in range(len(trialdata)-2):
    approx_frame_time[k] = (trialdata['trial_start_frame_index'][k+1] - trialdata['trial_start_frame_index'][k]) * (average_interval/1000)
    
    
time_difference = approx_frame_time - bpod_trial_dur
unexpected_time_diff = time_difference[time_difference > average_interval/1000] #Average interval is in ms
if unexpected_time_diff.shape[0]:
    print("There is a mismatch between the recorded trial duration \nand the expected time of trials from the imaging frames.")

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
    for q in range(1000):
        tmp = random.randrange(0, frame_time.shape[0])
        random_frame_matched[q,n] = int(tmp + trialdata["trial_start_frame_index"][rewarded[n]] + 1)
    #Draw 1000 random frames from the same trial, might also be the rewarded one!

#Define the amount of data prior and after reward delivery to look at
window = 40 #Look at the second before and after reward

C_rew = np.zeros((window+1, len(rewarded), C.shape[0] ))
S_rew = np.zeros((window+1, len(rewarded), C.shape[0] ))


C_rew_shuffled = np.zeros((window+1, len(rewarded), C.shape[0] ))
S_rew_shuffled = np.zeros((window+1, len(rewarded), C.shape[0] ))

tempC = np.empty((window+1, 1000)) #Represents all 1000 traces drawn with random center frame from one rewarded trial
tempS = np.empty((window+1, 1000))

for k in range(C.shape[0]-1):
    for n in range(len(reward_delivery_frame)-1):
        C_rew[:,n,k] = C[k, int(reward_delivery_frame[n] - window/2) : int(reward_delivery_frame[n]  + window/2 + 1)]
        S_rew[:,n,k] = S[k, int(reward_delivery_frame[n] - window/2) : int(reward_delivery_frame[n]  + window/2 + 1)]
    
        for q in range(1000):
            tempC[:,q] =  C[k, int(random_frame_matched[q,n] - window/2) : int(random_frame_matched[q,n]  + window/2 + 1)]
            tempS[:,q] =  S[k, int(random_frame_matched[q,n] - window/2) : int(random_frame_matched[q,n]  + window/2 + 1)]
        C_rew_shuffled[:,n,k] = np.mean(tempC, axis=1)
        S_rew_shuffled[:,n,k] = np.mean(tempS, axis=1)
        
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
    