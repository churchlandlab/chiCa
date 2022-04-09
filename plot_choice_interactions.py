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
prior_category =  determine_prior_variable(np.array(trialdata['correct_side']))

outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
prior_outcome =  determine_prior_variable(outcome)

color_specs = ['#FC7659','#b20707','#ABBAFF','#062999' ]

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




#%%---------Make plots to split the current choice into past choice traces

fig_split_current = plt.figure(figsize=(10, 8))
fig_split_current.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
spAx = [None] * 4

#Right will correctly be chosen
spAx[0] = fig_split_current.add_subplot(221)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
left_lines = spAx[0].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
right_lines = spAx[0].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
spAx[0].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
#spAx[0].set_xlabel('Time from stim onset (s)')
spAx[0].set_ylabel('Fluorescence intensity (A.U.)')
spAx[0].set_title('Upcoming correct right choice')

#Left will correctly be chosen
spAx[1] = fig_split_current.add_subplot(224)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
left_lines = spAx[1].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
right_lines = spAx[1].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
spAx[1].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
spAx[1].set_xlabel('Time from stim onset (s)')
spAx[1].set_ylabel('Fluorescence intensity (A.U.)')
spAx[1].set_title('Upcoming correct left choice')

#Right will incorrectly be chosen
spAx[2] = fig_split_current.add_subplot(223)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
left_lines = spAx[2].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
right_lines = spAx[2].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
spAx[2].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
spAx[2].set_xlabel('Time from stim onset (s)')
spAx[2].set_ylabel('Fluorescence intensity (A.U.)')
spAx[2].set_title('Upcoming incorrect right choice')

#Left will incorrectly be chosen
spAx[3] = fig_split_current.add_subplot(222)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
left_lines = spAx[3].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
right_lines = spAx[3].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
spAx[3].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
#spAx[3].set_xlabel('Time from stim onset (s)')
spAx[3].set_ylabel('Fluorescence intensity (A.U.)')
spAx[3].set_title('Upcoming incorrect left choice')

#Adjust y-axis limtis
tmp = np.zeros([4,2]) 
for k in range(len(spAx)):
     tmp[k,:] = spAx[k].get_ylim()
    
y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
for k in range(len(spAx)):
    spAx[k].set_ylim(y_lims)
del y_lims

#%%

fig_split_out_current = plt.figure(figsize=(10, 8))
fig_split_out_current.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
opAx = [None] * 4

color_specs = ['#587f0a','#d48611']

#Right will correctly be chosen
opAx[0] = fig_split_out_current.add_subplot(221)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
reward_lines = opAx[0].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
punish_lines = opAx[0].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
opAx[0].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
#opAx[0].set_xlabel('Time from stim onset (s)')
opAx[0].set_ylabel('Fluorescence intensity (A.U.)')
opAx[0].set_title('Upcoming correct right choice')

#Left will incorrectly be chosen
opAx[1] = fig_split_out_current.add_subplot(222)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
reward_lines = opAx[1].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
punish_lines = opAx[1].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
opAx[1].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
#opAx[0].set_xlabel('Time from stim onset (s)')
opAx[1].set_ylabel('Fluorescence intensity (A.U.)')
opAx[1].set_title('Upcoming incorrect left choice')

#Right will incorrectly be chosen
opAx[2] = fig_split_out_current.add_subplot(223)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
reward_lines = opAx[2].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
punish_lines = opAx[2].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
opAx[2].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
opAx[2].set_xlabel('Time from stim onset (s)')
opAx[2].set_ylabel('Fluorescence intensity (A.U.)')
opAx[2].set_title('Upcoming incorrect right choice')

#Left will correctly be chosen
opAx[3] = fig_split_out_current.add_subplot(224)
grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
reward_lines = opAx[3].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
punish_lines = opAx[3].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
opAx[3].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
opAx[3].set_xlabel('Time from stim onset (s)')
opAx[3].set_ylabel('Fluorescence intensity (A.U.)')
opAx[3].set_title('Upcoming correct left choice')

#Adjust y-axis limtis
tmp = np.zeros([4,2]) 
for k in range(len(opAx)):
     tmp[k,:] = opAx[k].get_ylim()
    
y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
for k in range(len(spAx)):
    opAx[k].set_ylim(y_lims)
del y_lims

#%%--------Further subdivide the traces into prior choice by category 

color_specs = ['#b20707','#ABBAFF','#FC7659','#062999']

possibilities = np.array([[1,1],[1,0],[0,1],[0,0]])
#Possibile combinations to balance
subplot_titles = ['Upcoming correct right choice', 'Upcoming incorrect left choice',
               'Upcoming incorrect right choice', 'Upcoming correct left choice']

line_labels = ['Prior right category and right choice', 'Prior right category and left choice',
               'Prior left category and right choice', 'Prior left category and left choice']
x_label_string = 'Seconds from stimulus onset'

fig_pat_cat_choice = plt.figure(figsize=(15, 9))
fig_pat_cat_choice.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
pat_cat = [None] * 4


for k in range(possibilities.shape[0]):
    pat_cat[k] = fig_pat_cat_choice.add_subplot(2,2,k+1)
    grouping_variable = np.squeeze(np.array([(trialdata['response_side']==possibilities[k,0]) & (trialdata['correct_side']==possibilities[k,1])])) #Squeeze the second dimension
    
    splits = []
    for n in range(possibilities.shape[0]):
        splits.append(np.where(((prior_category == possibilities[n,0]) & (prior_choice== possibilities[n,1])) & (grouping_variable==1))[0])
        
        if np.any(splits[n]):
            tmp_mean = np.nanmean(aligned_signal[:,splits[n]], axis=1)
            tmp_sem = np.nanstd(aligned_signal[:,splits[n]], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,splits[n]])==0))
 
            pat_cat[k].fill_between(x_vect, tmp_mean-tmp_sem, tmp_mean+tmp_sem , color=color_specs[n], alpha=0.2) 
            pat_cat[k].plot(x_vect, tmp_mean, color=color_specs[n], label= line_labels[n] + f', {splits[n].shape[0]} trials')  #Right in red
            pat_cat[k].axvline(0, color='k', linestyle='--')
            
            
    pat_cat[k].set_xlabel(x_label_string)
    pat_cat[k].set_ylabel('Fluorescence intensity (A.U.)')
    pat_cat[k].set_title(subplot_titles[k])
    pat_cat[k].legend(loc='best')

#Adjust y-axis limtis
tmp = np.zeros([4,2]) 
for k in range(len(pat_cat)):
     tmp[k,:] = pat_cat[k].get_ylim()

y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
for k in range(len(pat_cat)):
    pat_cat[k].set_ylim(y_lims)
del y_lims





