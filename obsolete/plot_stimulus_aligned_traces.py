# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:13:50 2022

@author: Lukas Oesch
"""



#%%
def stim_time_stamps(trialdata, average_interval, upper_frequency_limit, lower_frequency_limit, only_first):
    '''Locate the frame during which a stimulus is delivered. This function
    allows the user to specify a stim train frequency limit, to ensure that 
    the calcium imaging can track single events reliably.'''
    
    stim_start_frame = []
    
    for n in range(len(trialdata)): 
        if (np.isnan(trialdata['stimulus_event_timestamps'][n][0]) == 0) & (np.isnan(trialdata['DemonEarlyWithdrawal'][n][0]) == 1): #The stimulus has been played
            if (trialdata['stimulus_event_timestamps'][n].shape[0] <= upper_frequency_limit) & (trialdata['stimulus_event_timestamps'][n].shape[0] >= lower_frequency_limit):
                if only_first:
                    pick_stims = 1
                else:
                    pick_stims = trialdata['stimulus_event_timestamps'][n].shape[0]
                for k in range(pick_stims):
                    try:   
                        stim_time = np.arange(trialdata['trial_start_time_covered'][n]/1000, trialdata['FinishTrial'][n][0] - trialdata['Sync'][n][0], average_interval/1000)
                        #Generate frame times starting the first frame at the end of its coverage of trial inforamtion
                    except:
                        stim_time = np.arange(trialdata['trial_start_time_covered'][n]/1000, trialdata['FinishTrial'][n][0] - trialdata['ObsTrialStart'][n][0], average_interval/1000)
                        #This is to fit in the previous implementation of chipmunk
                    tmp = stim_time - trialdata['stimulus_event_timestamps'][n][k] #Calculate the time difference
                    stim_start_frame.append(int(np.where(tmp > 0)[0][0] + trialdata["trial_start_frame_index"][n]))
                #np.where returns a tuple where the first element are the indices that fulfill the condition.
                #Inside the array of indices retrieve the first one that is positive, therefore the first
                #frame that caputres some information.

    return stim_start_frame 
#%%-------Extract the first stimulus


upper_frequency_limit = 24 #No more than 10 Hz stimuli, because in the 4 and 5 hz condition they happen very aligned to the poking
lower_frequency_limit = 10
only_first = True #Just use the first stimulus inside the stim train
stim_start_frame = stim_time_stamps(trialdata, average_interval, upper_frequency_limit, lower_frequency_limit, only_first)

window = 2 #Look at the 0.5 seconds before and after the stimulus
interval = np.floor(average_interval)/1000 #Short hand for the average interval between frames, assumes scope acquisition is slower than expected 

aligned_signal = np.zeros([int(window/interval+1), len(stim_start_frame), signal.shape[0]])
for k in range(signal.shape[0]):
    for n in range(len(stim_start_frame)):
        if np.isnan(stim_start_frame[n]) == 0:
            aligned_signal[:, n, k] = signal[k, int(stim_start_frame[n] - window/interval /2) : int(stim_start_frame[n]  + window/interval /2 + 1)]
        else: 
            aligned_signal[:,n, k] = np.nan
  

#%%--- average for each cell and sort by responsiveness

mean_stim_response = np.nanmean(aligned_signal, axis = 1)
activity_diff = np.mean(mean_stim_response[int(np.ceil((window/2)/interval)):int(window/interval+2), :], axis=0) - np.mean(mean_stim_response[0:int(np.ceil((window/2)/interval)),:],axis=0)
sort_by_stim_modulation = np.argsort(activity_diff)

fi = plt.figure(figsize=(7,10))
ax = fi.add_axes((0.15,0.15,0.5,0.7))
im_handle = ax.imshow(np.transpose(mean_stim_response[:,sort_by_stim_modulation]), aspect='auto', cmap='inferno')
ax.axvline((window/2)/interval, color='w', linestyle='--')
ax.set_xlabel('Time from first stimulus event (s)', fontsize = 14)
ax.set_xticks([0, (window/4)/interval, (window/2)/interval, (window*3/4)/interval, window/interval])
ax.tick_params(axis='both', labelsize=12) #Works also for both axes at the same time
#ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels([-window/2, -window/4, 0, window/4, window/2])
ax.set_ylabel('Neuron number', fontsize = 14)
co_handle = fi.colorbar(mappable = im_handle) #The object to map the colorbar to is the image!
co_handle.set_label(label='z-scored inferred spike rate (A.U.)', fontsize = 14)
im_handle.figure.axes[1].tick_params(axis='y', labelsize = 12) #Very weitd notation, access sub-instance of the imshow handle...
fi.suptitle('Average response to first stimulus event', fontsize= 16, fontweight='bold')


x_vect = np.arange(-window/2, window/2 + interval, interval)
fi = plt.figure()
ax = fi.add_subplot(111)
ax.plot(x_vect, aligned_signal[:,:,sort_by_stim_modulation[377]], color=(0.5,0.5,0.5), linewidth = 0.5)
ax.plot(x_vect, mean_stim_response[:,sort_by_stim_modulation[377]], color='#980000')
ax.axvline(0, color ='k', linestyle='--')
ax.set_xlabel('Time from first stimulus event (s)')
ax.set_ylabel('z-scored inferred spike rate (A.U.)')




#aligned_signal, x_vect =  get_state_start_signal(signal, stim_start_frame, average_interval, window)

# prior_choice =  determine_prior_variable(np.array(trialdata['response_side']))

# outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
# outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
# prior_outcome =  determine_prior_variable(outcome)

# color_specs = ['#FC7659','#b20707','#ABBAFF','#062999']

#%%

fig_stim= plt.figure(figsize=(5, 11))
fig_stim.suptitle(f'Cell number {current_neuron} aligned to individual stimuli')
stim_ax = fig_stim.add_subplot(111)

# line_label = 'Mean +- std'

# tmp_mean = np.nanmean(aligned_signal, axis=1)
# tmp_sem = np.nanstd(aligned_signal, axis=1)
# #tmp_sem = np.nanstd(aligned_signal, axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal)==0))

# stim_ax.fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color='#25aa3c', alpha=0.2) 
# stim_ax.plot(x_vect, tmp_mean, color='#25aa3c', label=line_label)


#Find reasonable spacing
average_peak = np.max(np.nanmean(aligned_signal,axis=1))
spacing = average_peak*0.25 + average_peak

spacing= 0

for k in range(aligned_signal.shape[1]):
    stim_ax.plot(x_vect, aligned_signal[:,k] + (k-1)*spacing, color='#25aa3c', linewidth=0.3)
    
stim_ax.axvline(x=0, color='k',linestyle='--', linewidth=0.3)
stim_ax.set_xlabel('Time from stimulus occurence (s)')
stim_ax.set_ylabel('Fluorescence intensity (A.U.)')

