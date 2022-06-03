# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 00:11:01 2022

@author: Lukas Oesch
"""

from PIL import Image

COI = [26, 89, 132]
n = 132

m = int(assignments[assignments[:,1]==n,0][0])

neuron_shape = np.zeros([A1.shape[0], A1.shape[1],3])

neuron_shape[:,:,1] = A1[:,:,m]/np.max([A1[:,:,m]]) #Fill in the green channel with early session

neuron_shape[:,:,0] = A2[:,:,n]/np.max([A2[:,:,n]]) #Magenta for late
neuron_shape[:,:,2] = A2[:,:,n]/np.max([A2[:,:,n]])

z = neuron_shape

plt.figure()
ax1 = plt.axes(frameon=False)
ax1.imshow(neuron_shape)

crop_dims = 50 #as square
top_left = [70, 172]


neuron_zoom = neuron_shape[top_left[0]:top_left[0]+crop_dims, top_left[1]:top_left[1]+crop_dims, :]

plt.figure()
plt.imshow(neuron_zoom)


#%%----------


signal = C2


neuronNum = 132
#neuronNum = int(assignments[assignments[:,1]==89,0][0])

average_interval_seconds = np.floor(average_interval)/1000

    
half_window = 30
    
take_idx = np.where(np.isnan(np.array(state_start_frame))==0)[0]         
C_rewarded= np.full((take_idx.shape[0],2*half_window+1), np.nan)
for n in range(take_idx.shape[0]):
            C_rewarded[n,:] = signal[neuronNum, state_start_frame[take_idx[n]] - half_window : state_start_frame[take_idx[n]] + half_window+1]
  
fi = plt.figure()
fi.suptitle(f'Cell number {neuronNum} aligned to {state_name}')
ax1 = fi.add_subplot(111)
    
ax1.axvline(x=half_window+1, color='k',linestyle='--', linewidth=0.3)
ax1.plot(C_rewarded[trialdata['response_side'][take_idx] == 1,:].transpose(),linestyle='--',color='#FC7659',linewidth=0.3) #F is the red
ax1.plot(np.nanmean(C_rewarded[trialdata['response_side'][take_idx] == 1,:],axis = 0),color='#A2002F',linewidth=2) #F is the red
ax1.plot(C_rewarded[trialdata['response_side'][take_idx] == 0,:].transpose(),linestyle='--',color='#ABBAFF',linewidth=0.3)#The blue
ax1.plot(np.nanmean(C_rewarded[trialdata['response_side'][take_idx] == 0,:],axis = 0),color='#2800A2',linewidth=2) #F is the red
ax1.set_xlabel('Time from state onset')
ax1.set_ylabel('Fluorescence intensity (A.U.)')
ax1.set_xticks(np.arange(1,half_window*2+2,half_window/2))
ax1.set_ylim([0, 120])
ax1.set_xticklabels(np.arange(-half_window*average_interval_seconds,half_window*average_interval_seconds + half_window/2*average_interval_seconds, half_window/2*average_interval_seconds))
    

