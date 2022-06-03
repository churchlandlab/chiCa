# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:11:58 2022

@author: Lukas Oesch
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%

#Plug in the last outcome presentation
last_outcome = -2.3791 #Hard-coded for now







#######################################################################
#%% Decoding prior category

aligned_to = 'OutcomePresentation'
draw_last_outcome = False

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_']

if aligned_to == 'OutcomePresentation':
    balanced_decoders.append('C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_outcome_0_trials_back_aligned_to_')
    x_ax_label = 'time from outcome (s)'
elif aligned_to == 'DemonInitFixation':
    draw_last_outcome = True
    x_ax_label = 'time from center poke (s)'
elif aligned_to == 'PlayStimulus':
    x_ax_label = 'time from stimulus onset (s)'
elif aligned_to == 'DemonWaitForResponse':
    x_ax_label = 'time from movement onset (s)'
    
color_scheme = ['#5b5b5b', '#ecbe90', '#660099', '#04b18f', '#445bc7', '#8a590b']



line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle','by current outcome, data', 'by current outcome, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k] + aligned_to + '.h5','/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
if draw_last_outcome:
    d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
    
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel(x_ax_label)
d_ax.legend(loc='lower left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior category, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%%---Decoding prior response

aligned_to = 'DemonWaitForResponse'
draw_last_outcome = False

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_']

if aligned_to == 'OutcomePresentation':
    balanced_decoders.append('C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_outcome_0_trials_back_aligned_to_')
    x_ax_label = 'time from outcome (s)'
elif aligned_to == 'DemonInitFixation':
    draw_last_outcome = True
    x_ax_label = 'time from center poke (s)'
elif aligned_to == 'PlayStimulus':
    x_ax_label = 'time from stimulus onset (s)'
elif aligned_to == 'DemonWaitForResponse':
    x_ax_label = 'time from movement onset (s)'
    
color_scheme = ['#5b5b5b', '#ee638f', '#660099', '#04b18f', '#445bc7', '#8a590b']
line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle','by current outcome, data', 'by current outcome, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k] + aligned_to + '.h5','/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
if draw_last_outcome:
    d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
    
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel(x_ax_label)
d_ax.legend(loc='lower left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior choice, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)

#%%-------Prior outcome

aligned_to = 'PlayStimulus'
draw_last_outcome = False

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_']

if aligned_to == 'OutcomePresentation':
    balanced_decoders.append('C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_outcome_0_trials_back_aligned_to_')
    x_ax_label = 'time from outcome (s)'
elif aligned_to == 'DemonInitFixation':
    draw_last_outcome = True
    x_ax_label = 'time from center poke (s)'
elif aligned_to == 'PlayStimulus':
    x_ax_label = 'time from stimulus onset (s)'
elif aligned_to == 'DemonWaitForResponse':
    x_ax_label = 'time from movement onset (s)'
    
color_scheme = ['#5b5b5b', '#ee638f', '#ecbe90', '#04b18f', '#445bc7', '#8a590b']
line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior category, data', 'by prior category, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle','by current outcome, data', 'by current outcome, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k] + aligned_to + '.h5','/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
if draw_last_outcome:
    d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
    
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel(x_ax_label)
d_ax.legend(loc='lower right')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior outcome, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%%---current category

aligned_to = 'PlayStimulus'
draw_last_outcome = False

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_response_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_outcome_1_trials_back_aligned_to_',       
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_response_side_0_trials_back_aligned_to_']

if aligned_to == 'OutcomePresentation':
    balanced_decoders.append('C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_outcome_0_trials_back_aligned_to_')
    x_ax_label = 'time from outcome (s)'
elif aligned_to == 'DemonInitFixation':
    draw_last_outcome = True
    x_ax_label = 'time from center poke (s)'
elif aligned_to == 'PlayStimulus':
    x_ax_label = 'time from stimulus onset (s)'
elif aligned_to == 'DemonWaitForResponse':
    x_ax_label = 'time from movement onset (s)'
    
color_scheme = ['#5b5b5b', '#ecbe90','#ee638f','#660099', '#445bc7', '#8a590b']
line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle','by prior category, data', 'by prior category, shuffle','by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current outcome, data', 'by current outcome, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k] + aligned_to + '.h5','/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
if draw_last_outcome:
    d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
    
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel(x_ax_label)
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding current category, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)

#%%----current response

aligned_to = 'PlayStimulus'
draw_last_outcome = False

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_response_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_',
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_outcome_1_trials_back_aligned_to_',       
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_']

if aligned_to == 'OutcomePresentation':
    balanced_decoders.append('C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_outcome_0_trials_back_aligned_to_')
    x_ax_label = 'time from outcome (s)'
elif aligned_to == 'DemonInitFixation':
    draw_last_outcome = True
    x_ax_label = 'time from center poke (s)'
elif aligned_to == 'PlayStimulus':
    x_ax_label = 'time from stimulus onset (s)'
elif aligned_to == 'DemonWaitForResponse':
    x_ax_label = 'time from movement onset (s)'
    
color_scheme = ['#5b5b5b', '#ecbe90','#ee638f','#660099', '#04b18f', '#8a590b']
line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle','by prior category, data', 'by prior category, shuffle','by prior outcome, data', 'by prior outcome, shuffle',
               'by current choice, data', 'by current choice, shuffle','by current outcome, data', 'by current outcome, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k] + aligned_to + '.h5','/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
if draw_last_outcome:
    d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
    
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel(x_ax_label)
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding current choice, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%%





###############################################################

#%%----Prior stimulus category

#Unofortunate: the number of samples for the decoders were not retained. Input them manually below
#trial_counts = [235,76,76,234,173] #The the number of observations the decoder
#for category was trained on, taking balancing into account and also the 10-fold 
#cross-validation


# balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_aligned_to_DemonInitFixation.h5',
#                      'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_DemonInitFixation.h5',
#                      'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_DemonInitFixation.h5',
#                      'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_ balanced_by_correct_side_0_trials_back_aligned_to_DemonInitFixation.h5',
#                      'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_DemonInitFixation.h5']

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_response_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_outcome_1_trials_back_aligned_to_PlayStimulus.h5',
                     ]

color_scheme = ['#5b5b5b', '#ecbe90', '#660099', '#04b18f', '#445bc7']

line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts[k]} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts[k]} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from poking')
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior category, balancing by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%%----for choice

trial_counts = [254,76,76,230,184] #The the number of observations the decoder
#for category was trained on, taking balancing into account and also the 10-fold 
#cross-validation


balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_DemonInitFixation.h5']
color_scheme = ['#5b5b5b', '#ee638f', '#660099', '#04b18f', '#445bc7']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts[k]} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts[k]} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from poking')
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior choice, balancing by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)



#%%----For outcome

trial_counts = [97,76,76,61,76] #The the number of observations the decoder
#for category was trained on, taking balancing into account and also the 10-fold 
#cross-validation


balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_DemonInitFixation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_DemonInitFixation.h5']
color_scheme = ['#5b5b5b', '#ee638f', '#ecbe90', '#04b18f', '#445bc7']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior choice, data', 'by prior choice, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts[k]} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts[k]} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from poking')
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior outcome, balancing by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)

#%%-----Current category at play stim

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_response_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_outcome_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_0_trials_back_balanced_by_response_side_0_trials_back_aligned_to_PlayStimulus.h5']

color_scheme = ['#5b5b5b', '#ecbe90', '#ee638f','#660099', '#445bc7']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current choice, data', 'by current choice, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
#d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from stimulus onset')
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding current category, balancing by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)



#%%--------Current choice 

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_response_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_outcome_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_0_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_PlayStimulus.h5']

color_scheme = ['#5b5b5b', '#ecbe90', '#ee638f','#660099','#04b18f']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
#d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from stimulus onset')
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding upcoming choice, balancing by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)

#%%----Prior choice at play stim


balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_PlayStimulus.h5',]

color_scheme = ['#5b5b5b','#ee638f','#660099','#04b18f','#445bc7']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle']

fi = plt.figure(figsize=(12,8))
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
#d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from stimulus onset')
d_ax.legend(loc='lower right')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior choice, balancing by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)

#%% Prior category at play stim 

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_PlayStimulus.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_PlayStimulus.h5']

color_scheme = ['#5b5b5b', '#ecbe90', '#660099', '#04b18f', '#445bc7']

line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
#d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from stimulus onset')
d_ax.legend(loc='lower right')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior category, balancing by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%% Prior outcome at play stim


#trial_counts = [97,76,76,61,76] #The the number of observations the decoder
#for category was trained on, taking balancing into account and also the 10-fold 
#cross-validation


balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_response_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_response_side_0_trials_back_aligned_to_OutcomePresentation.h5']
color_scheme = ['#5b5b5b', '#ee638f', '#ecbe90', '#04b18f', '#445bc7']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior choice, data', 'by prior choice, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle']

fi = plt.figure(figsize=(12,8))
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
        
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from poking')
d_ax.legend(loc='upper left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior outcome, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%% Prior category at outcome presentation


trial_counts = [246,75,75,234,172,96] #The the number of observations the decoder
balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_outcome_0_trials_back_aligned_to_OutcomePresentation.h5',]

color_scheme = ['#5b5b5b', '#ecbe90', '#660099', '#04b18f', '#445bc7', '#8a590b']

line_labels = ['full set, data', 'full set shuffle', 'by prior choice, data', 'by prior choice, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle','by current outcome, data', 'by current outcome, shuffle']

fi = plt.figure()
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    #trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts[k]} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts[k]} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
#d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from outcome presentation')
d_ax.legend(loc='lower left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior category, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%% Prior choice at outcome presentation

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_outcome_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/response_side_1_trials_back_balanced_by_outcome_0_trials_back_aligned_to_OutcomePresentation.h5']

color_scheme = ['#5b5b5b', '#ee638f','#660099','#04b18f','#445bc7', '#8a590b']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior outcome, data', 'by prior outcome, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle','by current outcome, data', 'by current outcome, shuffle']

trial_counts = [253,75,75,230,184,100] 

fi = plt.figure(figsize=(12,8))
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    
    
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts[k]} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts[k]} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
#d_ax.axvline(last_outcome, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from outcome presentation (s)')
d_ax.legend(loc='lower left')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior choice, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)


#%% Prior outcome at outcome presentation

trial_counts = [97,76,76,61,76,40] #The the number of observations the decoder
#for category was trained on, taking balancing into account and also the 10-fold 
#cross-validation


balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_response_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_1_trials_back_balanced_by_outcome_0_trials_back_aligned_to_OutcomePresentation.h5']
color_scheme = ['#5b5b5b', '#ee638f', '#ecbe90', '#04b18f', '#445bc7','#8a590b']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior choice, data', 'by prior choice, shuffle',
               'by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle','by current outcome, data', 'by current outcome, shuffle']

fi = plt.figure(figsize=(12,8))
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    #trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts[k]} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts[k]} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from outcome presentation (s)')
d_ax.legend(loc='upper right')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding prior outcome, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)

#%%-Current outcome at outcome presentation

balanced_decoders = ['C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_correct_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_response_side_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_outcome_1_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_correct_side_0_trials_back_aligned_to_OutcomePresentation.h5',
                     'C:/data/LO032/20220215_114758/decoding/outcome_0_trials_back_balanced_by_response_side_0_trials_back_aligned_to_OutcomePresentation.h5']
color_scheme = ['#5b5b5b', '#ee638f', '#ecbe90', '#660099', '#04b18f', '#445bc7','#8a590b']

line_labels = ['full set, data', 'full set shuffle', 'by prior category, data', 'by prior category, shuffle', 'by prior choice, data', 'by prior choice, shuffle',
               'by prior outcome, data', 'by current outcome, shuffle','by current category, data', 'by current category, shuffle','by current choice, data', 'by current choice, shuffle']

fi = plt.figure(figsize=(12,8))
d_ax = fi.add_axes([0.1,0.1,0.8,0.8])

for k in range(len(balanced_decoders)):
    decoders = pd.read_hdf(balanced_decoders[k],'/Data')
    time_vect = np.array(decoders['frame_from_alignment']/20)
    perf = []
    perf_shuf = []
    for n in decoders['models']:
        perf.append(np.mean(n['model_accuracy']))
        perf_shuf.append(np.mean(n['shuffle_accuracy']))
    trial_counts = n['number_of_samples'].tolist()[0]
    d_ax.plot(time_vect, perf, color=color_scheme[k], label=line_labels[2*k] + f', {trial_counts} trials')
    d_ax.plot(time_vect, perf_shuf, color=color_scheme[k], alpha = 0.2, label=line_labels[2*k + 1] + f', {trial_counts} trials')
    
d_ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
d_ax.set_ylabel('Decoding accuracy')
d_ax.set_xlabel('Time from outcome presentation (s)')
d_ax.legend(loc='lower right')
d_ax.set_ylim((0.4,1))
d_ax.set_title('Decoding outcome, balanced by different task variables')
d_ax.grid(axis='y')
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)