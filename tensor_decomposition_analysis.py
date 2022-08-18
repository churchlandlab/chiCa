# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:29:46 2022

@author: Lukas Oesch
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorly as tl
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac
from decoding_utils import find_state_start_frame_imaging
from decoding_utils import determine_prior_variable
from sklearn.decomposition import PCA
from choice_strategy_models import post_outcome_side_switch

#%%-------Temporary data preprocessing
aligned_state = 'PlayStimulus'
state_start_frame, state_time_covered = find_state_start_frame_imaging(aligned_state,trialdata, average_interval, trial_start_time_covered)
    
prior_choice =  determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1)
prior_category =  determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1)

outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
prior_outcome =  determine_prior_variable(outcome ,np.ones(len(trialdata)), 1)
switched, heading_direction = post_outcome_side_switch(trialdata)

interval = np.floor(average_interval)/1000 #Short hand for the average interval between frames, assumes scope acquisition is slower than expected
aligned_signal = np.zeros([int(window/interval +1), len(state_start_frame), signal.shape[0]])
for k in range(signal.shape[0]):
    for n in range(len(state_start_frame)):
        if np.isnan(state_start_frame[n]) == 0:
            aligned_signal[:, n, k] = signal[k, int(state_start_frame[n] - window/interval /2) : int(state_start_frame[n]  + window/interval /2 + 1)]
        else: 
            aligned_signal[:,n, k] = np.nan
  
valid_trials = np.array((np.isnan(prior_choice)==0) & (np.isnan(trialdata['response_side'])==0))
tensor = aligned_signal[:,valid_trials,:]


#%%---------do tensor decomposition
rank = 10
factors_tl = parafac(tensor, rank) 
tensor_components = factors_tl.factors


#%%--------Generate a matrix of task variables of interest na dinspect its correlations

label_mat = np.transpose(np.array((prior_category[valid_trials], prior_choice[valid_trials], prior_outcome[valid_trials], heading_direction[valid_trials], trialdata['correct_side'][valid_trials], trialdata['response_side'][valid_trials], outcome[valid_trials])))

task_vars = ['prior category', 'prior choice', 'prior outcome', 'heading direction', 'category', 'choice', 'outcome']
var_cor = np.corrcoef(np.transpose(label_mat))

fi = plt.figure(figsize=(7,5))
ax = fi.add_axes((0.2,0.25,0.6,0.6))
ms_handle = ax.matshow(var_cor, vmin=-1, vmax=1, cmap='bwr' )
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
ax.set_xticklabels(['']+task_vars, Rotation=45)
ax.set_yticklabels(['']+task_vars)
fi.colorbar(ms_handle, label='correlation coefficient')
ax.set_title('Correlation of task variable labels', fontweight='bold')


#%%
corrs = np.zeros([rank, label_mat.shape[1]])*np.nan
for k in range(label_mat.shape[1]):
    for n in range(rank):
        rr = np.corrcoef(tensor_components[1][:,n], label_mat[:,k])
        corrs[n,k] = rr[0,1]
        
        
        
        
        
        
        
        
#%%--------Code to decompose the b-weight matrix from the logistic regression


decoders = ['C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_DemonInitFixation.h5',
            'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_PlayStimulus.h5',
            'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_DemonWaitForResponse.h5',
            'C:/data/LO032/20220215_114758/decoding/correct_side_1_trials_back_balanced_by_response_side_1_trials_back_aligned_to_OutcomePresentation.h5']

coef_list =[]
accuracy_list = []
x_vect = []

for n in decoders:
    tmp = pd.read_hdf(n,'/Data')
    co = [] 
    ac = []
    for k in tmp['models']:
        co.append(np.mean(k['model_coefficients'], axis=0))
        ac.append(np.mean(k['model_accuracy'], axis=0))
        
    # coefficients = np.squeeze(np.array(co))
    # coef_mat = np.hstack((coef_mat, np.squeeze(np.array(ac))))
    # accuracy_vect = np.hstack((accurcy_vect, np.squeeze(np.array(ac))))
    coef_list.append(np.squeeze(np.array(co)))
    accuracy_list.append(np.squeeze(np.array(ac)))
    x_vect.append(np.array(tmp['frame_from_alignment'] * np.floor(average_interval)/1000)) 

coef_mat = np.vstack(coef_list)
accuracy_vect = np.hstack(accuracy_list)     
        
coefficient_variance = np.std(coef_mat, axis=1)**2
accuracy_variance_correlation = np.corrcoef(accuracy_vect, coefficient_variance)

pca_obj = PCA(10)
pca_obj.fit(np.transpose(coef_mat))
components = pca_obj.components_
loadings = pca_obj.transform(np.transpose(coef_mat))
explained_variance = pca_obj.explained_variance_

