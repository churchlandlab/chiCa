# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:24:59 2022

@author: Lukas Oesch
"""
for k in range(C_interpolated.shape[0]):
    
    neuronNum = n
    average_interval_seconds = np.floor(average_interval)/1000
    
    idx_correct = []
    idx_wrong= []
    
    
    half_window = 30
    
    for k in range(len(trialdata)):
        if np.isnan(trialdata['DemonReward'][k][0]) == 0:
            idx_correct.append(k)
        elif np.isnan(trialdata['DemonWrongChoice'][k][0]) == 0:
            idx_wrong.append(k)
          
    C_rewarded= np.full((len(idx_correct),2*half_window+1), np.nan)
    for n in range(len(idx_correct)):
        if np.isnan(state_start_frame_index[idx_correct[n]]) == 0:
            C_rewarded[n,:] = C_interpolated[neuronNum, state_start_frame_index[idx_correct[n]] - half_window : state_start_frame_index[idx_correct[n]] + half_window+1]
        
    C_punished= np.full((len(idx_wrong),2*half_window+1), np.nan)
    for n in range(len(idx_wrong)):
        if np.isnan(state_start_frame_index[idx_wrong[n]]) == 0:
            C_punished[n,:] = C_interpolated[neuronNum, state_start_frame_index[idx_wrong[n]] - half_window : state_start_frame_index[idx_wrong[n]] + half_window+1]  
    
    
    fi = plt.figure()
    fi.suptitle(f'Cell number {neuronNum} aligned to {state_name}')
    ax1 = fi.add_subplot(121)
    ax2 = fi.add_subplot(122)
    
    ax1.axvline(x=half_window+1, color='k',linestyle='--', linewidth=0.3)
    ax1.plot(C_rewarded[trialdata['response_side'][idx_correct] == 1,:].transpose(),linestyle='--',color='#FC7659',linewidth=0.3) #F is the red
    ax1.plot(np.nanmean(C_rewarded[trialdata['response_side'][idx_correct] == 1,:],axis = 0),color='#A2002F',linewidth=2) #F is the red
    ax1.plot(C_rewarded[trialdata['response_side'][idx_correct] == 0,:].transpose(),linestyle='--',color='#ABBAFF',linewidth=0.3)#The blue
    ax1.plot(np.nanmean(C_rewarded[trialdata['response_side'][idx_correct] == 0,:],axis = 0),color='#2800A2',linewidth=2) #F is the red
    ax1.set_xlabel('Time from state onset')
    ax1.set_title('Reward')
    ax1.set_ylabel('Fluorescence intensity (A.U.)')
    ax1.set_xticks(np.arange(1,half_window*2+2,half_window/2))
    ax1.set_xticklabels(np.arange(-half_window*average_interval_seconds,half_window*average_interval_seconds + half_window/2*average_interval_seconds, half_window/2*average_interval_seconds))
    
    ax2.axvline(x=half_window+1, color='k',linestyle='--', linewidth=0.3)
    ax2.plot(C_punished[trialdata['response_side'][idx_wrong] == 1,:].transpose(),linestyle='--',color='#FC7659',linewidth=0.3) #F is the red
    ax2.plot(np.nanmean(C_punished[trialdata['response_side'][idx_wrong] == 1,:], axis=0),color='#A2002F',linewidth=2) #F is the red
    ax2.plot(C_punished[trialdata['response_side'][idx_wrong] == 0,:].transpose(),linestyle='--',color='#ABBAFF',linewidth=0.3)#The blue
    ax2.plot(np.nanmean(C_punished[trialdata['response_side'][idx_wrong] == 0,:],axis = 0),color='#2800A2',linewidth=2) #F is the red
    ax2.set_xlabel('Time from state onset')
    ax2.set_title('Wrong choice')
    ax2.set_ylabel('Fluorescence intensity (A.U.)')
    ax2.set_xticks(np.arange(1,half_window*2+2,half_window/2))
    ax2.set_xticklabels(np.arange(-half_window*average_interval_seconds,half_window*average_interval_seconds + half_window/2*average_interval_seconds, half_window/2*average_interval_seconds))
   