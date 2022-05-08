icht# -*- coding: utf-8 -*-
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
#%%
color_specs = ['#FC7659','#b20707','#ABBAFF','#062999' ]

fig_prev= plt.figure(figsize=(12, 8))
fig_prev.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
plax = [None] * 2

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
plax[1].set_xlabel('Time from center poke (s)')
plax[1].set_ylabel('Fluorescence intensity (A.U.)')
#%%
tmp = np.zeros([2,2]) 
for k in range(len(plax)):
     tmp[k,:] = plax[k].get_ylim()
    
y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
for k in range(len(plax)):
    plax[k].set_ylim(y_lims)
del y_lims




# #%%---------Make plots to split the current choice into past choice traces

# fig_split_current = plt.figure(figsize=(10, 8))
# fig_split_current.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
# spAx = [None] * 4

# #Right will correctly be chosen
# spAx[0] = fig_split_current.add_subplot(221)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
# left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
# right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
# left_lines = spAx[0].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
# right_lines = spAx[0].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
# spAx[0].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
# #spAx[0].set_xlabel('Time from stim onset (s)')
# spAx[0].set_ylabel('Fluorescence intensity (A.U.)')
# spAx[0].set_title('Upcoming correct right choice')

# #Left will correctly be chosen
# spAx[1] = fig_split_current.add_subplot(224)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
# left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
# right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
# left_lines = spAx[1].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
# right_lines = spAx[1].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
# spAx[1].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
# spAx[1].set_xlabel('Time from stim onset (s)')
# spAx[1].set_ylabel('Fluorescence intensity (A.U.)')
# spAx[1].set_title('Upcoming correct left choice')

# #Right will incorrectly be chosen
# spAx[2] = fig_split_current.add_subplot(223)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
# left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
# right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
# left_lines = spAx[2].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
# right_lines = spAx[2].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
# spAx[2].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
# spAx[2].set_xlabel('Time from stim onset (s)')
# spAx[2].set_ylabel('Fluorescence intensity (A.U.)')
# spAx[2].set_title('Upcoming incorrect right choice')

# #Left will incorrectly be chosen
# spAx[3] = fig_split_current.add_subplot(222)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
# left_split = np.where((prior_choice==0) & (grouping_variable==1))[0]
# right_split = np.where((prior_choice==1) & (grouping_variable==1))[0]
# left_lines = spAx[3].plot(x_vect, aligned_signal[:,left_split], color=color_specs[3],linewidth=0.5)
# right_lines = spAx[3].plot(x_vect, aligned_signal[:,right_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior left choice, {len(left_lines)} trials', f'Prior right choice, {len(right_lines)} trials']
# spAx[3].legend([left_lines[0], right_lines[0]], line_labels, loc='upper left')
# #spAx[3].set_xlabel('Time from stim onset (s)')
# spAx[3].set_ylabel('Fluorescence intensity (A.U.)')
# spAx[3].set_title('Upcoming incorrect left choice')

# #Adjust y-axis limtis
# tmp = np.zeros([4,2]) 
# for k in range(len(spAx)):
#      tmp[k,:] = spAx[k].get_ylim()
    
# y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
# for k in range(len(spAx)):
#     spAx[k].set_ylim(y_lims)
# del y_lims

# #%%

# fig_split_out_current = plt.figure(figsize=(10, 8))
# fig_split_out_current.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
# opAx = [None] * 4

# color_specs = ['#587f0a','#d48611']

# #Right will correctly be chosen
# opAx[0] = fig_split_out_current.add_subplot(221)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
# reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
# punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
# reward_lines = opAx[0].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
# punish_lines = opAx[0].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
# opAx[0].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
# #opAx[0].set_xlabel('Time from stim onset (s)')
# opAx[0].set_ylabel('Fluorescence intensity (A.U.)')
# opAx[0].set_title('Upcoming correct right choice')

# #Left will incorrectly be chosen
# opAx[1] = fig_split_out_current.add_subplot(222)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==1)])) #Squeeze the second dimension
# reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
# punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
# reward_lines = opAx[1].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
# punish_lines = opAx[1].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
# opAx[1].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
# #opAx[0].set_xlabel('Time from stim onset (s)')
# opAx[1].set_ylabel('Fluorescence intensity (A.U.)')
# opAx[1].set_title('Upcoming incorrect left choice')

# #Right will incorrectly be chosen
# opAx[2] = fig_split_out_current.add_subplot(223)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==1) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
# reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
# punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
# reward_lines = opAx[2].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
# punish_lines = opAx[2].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
# opAx[2].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
# opAx[2].set_xlabel('Time from stim onset (s)')
# opAx[2].set_ylabel('Fluorescence intensity (A.U.)')
# opAx[2].set_title('Upcoming incorrect right choice')

# #Left will correctly be chosen
# opAx[3] = fig_split_out_current.add_subplot(224)
# grouping_variable = np.squeeze(np.array([(trialdata['response_side']==0) & (trialdata['correct_side']==0)])) #Squeeze the second dimension
# reward_split = np.where((prior_outcome==1) & (grouping_variable==1))[0]
# punish_split = np.where((prior_outcome==0) & (grouping_variable==1))[0]
# reward_lines = opAx[3].plot(x_vect, aligned_signal[:,reward_split], color=color_specs[0],linewidth=0.5)
# punish_lines = opAx[3].plot(x_vect, aligned_signal[:,punish_split], color=color_specs[1],linewidth=0.5)
# line_labels = [f'Prior reward, {len(reward_lines)} trials', f'Prior punishment, {len(punish_lines)} trials']
# opAx[3].legend([reward_lines[0], punish_lines[0]], line_labels, loc='upper left')
# opAx[3].set_xlabel('Time from stim onset (s)')
# opAx[3].set_ylabel('Fluorescence intensity (A.U.)')
# opAx[3].set_title('Upcoming correct left choice')

# #Adjust y-axis limtis
# tmp = np.zeros([4,2]) 
# for k in range(len(opAx)):
#      tmp[k,:] = opAx[k].get_ylim()
    
# y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
# for k in range(len(spAx)):
#     opAx[k].set_ylim(y_lims)
# del y_lims

#%%--------Further subdivide the traces into prior choice by category 

color_specs = ['#b20707','#ABBAFF','#FC7659','#062999']

possibilities = np.array([[1,1],[1,0],[0,1],[0,0]])
#Possibile combinations to balance
subplot_titles = ['Upcoming correct right choice', 'Upcoming incorrect right choice',
               'Upcoming incorrect left choice', 'Upcoming correct left choice']

line_labels = ['Prior right category and right choice', 'Prior right category and left choice',
               'Prior left category and right choice', 'Prior left category and left choice']
x_label_string = 'Seconds from center poke'

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
            #tmp_sem = np.nanstd(aligned_signal[:,splits[n]], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,splits[n]])==0))
            tmp_std = np.nanstd(aligned_signal[:,splits[n]], axis=1)
            
            pat_cat[k].fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=color_specs[n], alpha=0.2) 
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
plt.tight_layout()

#%%----Now instead plot all the current trial conditions  for each prior choice-category combination
# This should show no differences if the it is the category that is encoded

color_specs = ['#b20707','#FC7659','#ABBAFF','#062999']
#Upcoming correct right in dark red, upcoming incorrect right in light red, 
#upcoming correct left in dark blue, upcoming incorrect left in light blue

possibilities = np.array([[1,1],[1,0],[0,1],[0,0]])
#Possible combinations to balance
line_labels = ['Upcoming correct right choice', 'Upcoming incorrect right choice',
               'Upcoming incorrect left choice', 'Upcoming correct left choice']

subplot_titles = ['Prior right category and right choice', 'Prior right category and left choice',
               'Prior left category and right choice', 'Prior left category and left choice']
x_label_string = 'Seconds from center poke'

fig_by_prior = plt.figure(figsize=(15, 9))
fig_by_prior.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')
by_prior = [None] * 4


for k in range(possibilities.shape[0]):
    by_prior[k] = fig_by_prior.add_subplot(2,2,k+1)
    grouping_variable = np.squeeze(np.array([(prior_category==possibilities[k,0]) & (prior_choice==possibilities[k,1])])) #Squeeze the second dimension
    
    splits = []
    for n in range(possibilities.shape[0]):
        splits.append(np.where(((np.array(trialdata['response_side']) == possibilities[n,0]) & (np.array(trialdata['correct_side'])== possibilities[n,1])) & (grouping_variable==1))[0])
        
        if np.any(splits[n]):
            tmp_mean = np.nanmean(aligned_signal[:,splits[n]], axis=1)
            tmp_std = np.nanstd(aligned_signal[:,splits[n]], axis=1)
 
            by_prior[k].fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=color_specs[n], alpha=0.2) 
            by_prior[k].plot(x_vect, tmp_mean, color=color_specs[n], label= line_labels[n] + f', {splits[n].shape[0]} trials')  #Right in red
            by_prior[k].axvline(0, color='k', linestyle='--')
            
            
    by_prior[k].set_xlabel(x_label_string)
    by_prior[k].set_ylabel('Fluorescence intensity (A.U.)')
    by_prior[k].set_title(subplot_titles[k])
    by_prior[k].legend(loc='best')

#Adjust y-axis limtis
tmp = np.zeros([4,2]) 
for k in range(len(pat_cat)):
     tmp[k,:] = by_prior[k].get_ylim()

y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
for k in range(len(by_prior)):
    by_prior[k].set_ylim(y_lims)
del y_lims
plt.tight_layout()

#%%--------Plot traces as split by coming from the right port or the left one
#separated by rewarded on side and switching

#Run the part of side switching in choice strategy first

fig_heading = plt.figure(figsize=(12, 8))
fig_heading.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')

heading_combo = np.array([[0,0,0],[1,0,0],[0,1,1],[1,1,1],
                          [0,0,1],[1,0,1],[0,1,0],[1,1,0]])


#First column is correct side, response side, did switch sides, first line represents the rightward heading, second the leftward heading

line_labels = ['Rewarded on left, no switch', 'Punished on left, no switch', 'Punished on right, switch', 'Rewarded on right, switch',
                   'Rewarded on left, switch', 'Punished on left, switch', 'Punished on right, no switch', 'Rewarded on right, no switch']

line_colors = plt.cm.seismic(np.linspace(0, 1, heading_combo.shape[0]))
h_ax = fig_heading.add_subplot(111)
heading_splits = []

for k in range(heading_combo.shape[0]):
    heading_splits.append(np.where(((prior_category==heading_combo[k,0]) & (prior_choice==heading_combo[k,1])) & (went_to_other==heading_combo[k,2]))[0])

    if np.any(heading_splits[k]):
            tmp_mean = np.nanmean(aligned_signal[:,heading_splits[k]], axis=1)
            tmp_std = np.nanstd(aligned_signal[:,heading_splits[k]], axis=1)
 
            h_ax.fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=line_colors[k,:], alpha=0.2) 
            h_ax.plot(x_vect, tmp_mean, color=line_colors[k,:], label=line_labels[k] + f', {heading_splits[k].shape[0]} trials')  #Right in red
    
h_ax.axvline(0, color='k', linestyle='--')
h_ax.set_xlabel('Time from center poke (s)')
h_ax.set_ylabel('Fluorescence intensity (A.U.)')
h_ax.set_title('Heading direction, right = blue, left = red')
h_ax.legend(loc='best')
plt.tight_layout()


#%%-----

#Run the part of side switching in choice strategy first

fig_heading = plt.figure(figsize=(12, 8))
fig_heading.suptitle(f'Cell number {current_neuron} aligned to {align_to_state}')

heading_combo = np.array([[0,0,0],[1,0,0],[0,1,1],[1,1,1],
                          [0,0,1],[1,0,1],[0,1,0],[1,1,0]])


heading_combo = np.array([[1,1,1],[1,0,0],[0,1,1],[0,0,0],
                          [1,1,0],[1,0,1],[0,1,0],[0,0,1]])



#First column is correct side, response side, did switch sides, first line represents the rightward heading, second the leftward heading

line_labels = ['Rewarded on right, switch', 'Punished on left, no switch', 'Punished on right, switch', 'Rewarded on left, no switch',
                   'Rewarded on right, no switch', 'Punished on left, switch', 'Punished on right, no switch', 'Rewarded on left switch']

line_colors = ['#062999','#b20707']
heading_splits = []
h_ax = []

for k in range(int(heading_combo.shape[0]/2)):
    h_ax.append(fig_heading.add_subplot(2,2,k+1))
    
    #Split by rightward heading
    heading_splits.append(np.where(((prior_category==heading_combo[k,0]) & (prior_choice==heading_combo[k,1])) & (went_to_other==heading_combo[k,2]))[0])
    if np.any(heading_splits[-1]):
            tmp_mean = np.nanmean(aligned_signal[:,heading_splits[-1]], axis=1)
            tmp_std = np.nanstd(aligned_signal[:,heading_splits[-1]], axis=1)
 
            h_ax[k].fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=line_colors[0], alpha=0.2) 
            h_ax[k].plot(x_vect, tmp_mean, color=line_colors[0], label=line_labels[k] + f', {heading_splits[-1].shape[0]} trials')  #Right in red
    
    heading_splits.append(np.where(((prior_category==heading_combo[k+4,0]) & (prior_choice==heading_combo[k+4,1])) & (went_to_other==heading_combo[k+4,2]))[0])
    if np.any(heading_splits[-1]):
            tmp_mean = np.nanmean(aligned_signal[:,heading_splits[-1]], axis=1)
            tmp_std = np.nanstd(aligned_signal[:,heading_splits[-1]], axis=1)
 
            h_ax[k].fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=line_colors[1], alpha=0.2) 
            h_ax[k].plot(x_vect, tmp_mean, color=line_colors[1], label=line_labels[k+4] + f', {heading_splits[-1].shape[0]} trials')  #Right in red
    
    
    h_ax[k].axvline(0, color='k', linestyle='--')
    h_ax[k].set_xlabel('Time from center poke (s)')
    h_ax[k].set_ylabel('Fluorescence intensity (A.U.)')
    h_ax[k].set_title('Heading direction, right = blue, left = red')
    h_ax[k].legend(loc='best')
    
#Adjust y-axis limtis
tmp = np.zeros([4,2]) 
for k in range(len(h_ax)):
     tmp[k,:] = h_ax[k].get_ylim()

y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
for k in range(len(by_prior)):
    h_ax[k].set_ylim(y_lims)
del y_lims
plt.tight_layout()

