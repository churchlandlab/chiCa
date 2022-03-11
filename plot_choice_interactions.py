# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:07:34 2022

Visualize interactions between prior choice and prior outcome and upcoming 
choice and stimulus category. This script requires some functions of "visualize
task_aligned_traces" to be imported and uses the global current_neuron.
Make sure to have signal and window defined and fill in the desired state on top
of the script.

@author: Lukas Oesch
"""





align_to_state = 'PlayStimulus'

state_start_frame = state_time_stamps(align_to_state, trialdata, average_interval)
aligned_signal, x_vect =  get_state_start_signal(signal, state_start_frame, average_interval, window)

prior_choice =  determine_prior_variable(np.array(trialdata['response_side']))

outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
prior_outcome =  determine_prior_variable(outcome)

color_specs = ['#FC7659','#b20707','#ABBAFF','#062999']

fig_prev= plt.figure(figsize=(12, 8))
fig_prev.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
plax = [None] * 2
#%%
plax[0] = fig_prev.add_subplot(211)
line_labels = ['Wrong on right on last trial', 'Correct on right on last trial', 'Wrong on left on last trial', 'Correct on left on last trial']
#Right chosen right incorrect
grouping_variable = np.squeeze(np.array([(prior_choice==1) & (prior_outcome==0)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[0].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[0], alpha=0.2) #Right in red
plax[0].plot(x_vect, tmp_mean, color=color_specs[0], label=line_labels[0]) #Right in red

#Right chosen right correct
grouping_variable = np.squeeze(np.array([(prior_choice==1) & (prior_outcome==1)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[0].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[1], alpha=0.2) #Right in red
plax[0].plot(x_vect, tmp_mean, color=color_specs[1], label=line_labels[1]) #Right in red

#Left chosen left incorrect
grouping_variable = np.squeeze(np.array([(prior_choice==0) & (prior_outcome==0)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[0].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[2], alpha=0.2) #Right in red
plax[0].plot(x_vect, tmp_mean, color=color_specs[2], label=line_labels[2]) #Right in red

#Left chosen left correct
grouping_variable = np.squeeze(np.array([(prior_choice==0) & (prior_outcome==1)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[0].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[3], alpha=0.2) #Right in red
plax[0].plot(x_vect, tmp_mean, color=color_specs[3], label=line_labels[3]) #Right in red

plax[0].axvline(x=0, color='k',linestyle='--', linewidth=0.3)
plax[0].legend(loc='upper left')
plax[0].set_xlabel('Time from stim onset (s)')
plax[0].set_ylabel('Fluorescence intensity (A.U.)')

#%%
plax[1] = fig_prev.add_subplot(212)
line_labels = ['Should go left, goes right', 'Should go right, goes right', 'Should go right, goes left', 'Should go left, goes left']

#Right to be chosen, left correct
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[1].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[0], alpha=0.2) #Right in red
plax[1].plot(x_vect, tmp_mean, color=color_specs[0], label=line_labels[0]) #Right in red

#Right to be chosen, right correct
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[1].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[1], alpha=0.2) #Right in red
plax[1].plot(x_vect, tmp_mean, color=color_specs[1], label=line_labels[1]) #Right in red

#Left to be chosen, right correct
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[1].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[2], alpha=0.2) 
plax[1].plot(x_vect, tmp_mean, color=color_specs[2], label=line_labels[2]) 

#Left to be chosen, left correct
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
tmp_mean = np.nanmean(aligned_signal[:,grouping_variable], axis=1)
tmp_sem = np.nanstd(aligned_signal[:,grouping_variable], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,grouping_variable])==0))

plax[1].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[3], alpha=0.2) 
plax[1].plot(x_vect, tmp_mean, color=color_specs[3], label=line_labels[3]) 

plax[1].axvline(x=0, color='k',linestyle='--', linewidth=0.3)
plax[1].legend(loc='upper left')
plax[1].set_xlabel('Time from stim onset (s)')
plax[1].set_ylabel('Fluorescence intensity (A.U.)')
#%%
tmp = np.zeros([2,2]) 
for k in range(len(plax)):
     tmp[k,:] = plax[k].get_ylim()
    
y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
for k in range(len(plax)):
    plax[k].set_ylim(y_lims)
del y_lims