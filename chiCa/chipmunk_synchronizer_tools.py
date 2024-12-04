# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:19:33 2022

@author: Lukas Oesch
"""

import matplotlib.pyplot as plt
# off_frame = [None] * len(trialdata)
half_window = 5
LED_resp = np.empty((len(trialdata)-1,2*half_window+1))

for n in range(len(trialdata)-1):
      # frame_time = np.arange(0,trialdata["trial_start_frame_index"][n+1] - trialdata["trial_start_frame_index"][n]) * average_interval/1000
      # #This line assumes that the trial just starts with the new frame, when in reality it has started somewhat earlier
    
      # tmp = frame_time - trialdata["PlayStimulus"][n][0] #Calculate the time differenec
      # off_frame[n] = int(np.where(tmp > 0)[0][0] + trialdata["trial_start_frame_index"][n] + 1)
      
      LED_resp[n,:] = average_frame_intensity[state_start_frame[n] - (half_window):state_start_frame[n]+(half_window) + 1]
      
      
fi = plt.figure(figsize = [5, 9])
axOb = fi.add_subplot(111)
matLED = axOb.matshow(LED_resp, aspect='auto')
col_bar = fi.colorbar(matLED, ax=axOb)
axOb.set_xticks(np.arange(0,11))
axOb.set_xticklabels(np.arange(-5,6))
axOb.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
axOb.set_title('Frame alignment to poke LED off')
axOb.set_xlabel('Frames from Bpod state switching LED off')
axOb.set_ylabel("Trial number")
col_bar.set_label("Average frame intensity (A.U.)")


fu = plt.figure()
saxOb = fu.add_subplot(111)
saxOb.scatter(state_time_covered[0:LED_resp.shape[0]],LED_resp[:,5])
saxOb.set_title('Average frame intensity by LED off time at frame covering LED off')
saxOb.set_xlabel('LED off time covered by the frame (s)')
saxOb.set_ylabel("Average frame intensity (A.U.)")


fa = plt.figure()
baxOb = fa.add_subplot(111)
baxOb.scatter(state_time_covered[0:LED_resp.shape[0]],LED_resp[:,6])
baxOb.set_title('Average frame intensity by LED off time at frame covering LED off')
baxOb.set_xlabel('LED off time covered by the frame at t -1 (s)')
baxOb.set_ylabel("Average frame intensity (A.U.)")

      
#%%
import cv2
import numpy as np

video_file = camlog_file[0][0:len(camlog_file[0])-6] + 'avi' #Remove camlog extension and replace with avi

cap = cv2.VideoCapture(video_file)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video = np.zeros((frame_height, frame_width, frame_num), 'uint8')
for n in range(frame_num):
        success, f = cap.read()
        video[:,:,n] = f[:,:,0]
        
        
poke_LED = video[439,481,:]
poke_LED_resp = np.empty((len(trialdata)-1,2*half_window+1))
half_window = 5
for n in range(len(trialdata)-1):
      # frame_time = np.arange(0,trialdata["trial_start_frame_index"][n+1] - trialdata["trial_start_frame_index"][n]) * average_interval/1000
      # #This line assumes that the trial just starts with the new frame, when in reality it has started somewhat earlier
    
      # tmp = frame_time - trialdata["PlayStimulus"][n][0] #Calculate the time differenec
      # off_frame[n] = int(np.where(tmp > 0)[0][0] + trialdata["trial_start_frame_index"][n] + 1)
      
      poke_LED_resp[n,:] = poke_LED[state_start_video_frame[n] - (half_window):state_start_video_frame[n]+(half_window) + 1]
      
      
fi = plt.figure(figsize = [5, 9])
axOb = fi.add_subplot(111)
matLED = axOb.matshow(poke_LED_resp, aspect='auto')
col_bar = fi.colorbar(matLED, ax=axOb)
axOb.set_xticks(np.arange(0,11))
axOb.set_xticklabels(np.arange(-5,6))
axOb.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
axOb.set_title('Video frame alignment to poke LED off')
axOb.set_xlabel('Frames from Bpod state switching LED off (30 fps)')
axOb.set_ylabel("Trial number")
col_bar.set_label("LED pixel intensity (A.U.)")

cropped_video = video[400:500,420:540,:]
average_video_actual = np.mean(cropped_video[:,:,np.asarray(state_start_video_frame)[:]], axis = 2)
average_video_oneMore = np.mean(cropped_video[:,:,np.asarray(state_start_video_frame)[:]+1], axis = 2)
average_video_twoMore = np.mean(cropped_video[:,:,np.asarray(state_start_video_frame)[:]+2], axis = 2)

plt.figure()
plt.imshow(average_video_actual,cmap='gray')
plt.title('Average frame of Bpod state switching LED off')
plt.axis('off')

plt.figure()
plt.imshow(average_video_oneMore,cmap='gray')
plt.title('Average frame + 1 of Bpod state switching LED off')
plt.axis('off')

plt.figure()
plt.imshow(average_video_twoMore,cmap='gray')
plt.title('Average frame + 2 of Bpod state switching LED off')
plt.axis('off')



#Look at a video from a behaving mouse
video_file = camlog_file[0][0:len(camlog_file[0])-6] + 'avi' #Remove camlog extension and replace with avi

cap = cv2.VideoCapture(video_file)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if frame_num == 0:
    frame_num = video_tracking[0].shape[0]
    
scaling_factor = 0.25

binned_video = np.zeros([int(frame_height*scaling_factor), int(frame_width*scaling_factor), frame_num],'uint8')

for n in range(frame_num):
    success, f = cap.read()
    binned_video[:,:,n] = np.uintc(cv2.resize(f[:,:,0], (int(scaling_factor*frame_width), int(scaling_factor*frame_height))))
    
port2in_onePlus = np.zeros([frame_height, frame_width, len(state_start_video_frame)],'uint8')

for n in range(len(state_start_video_frame)):
    cap.set(1, state_start_video_frame[n]+1)
    success, f = cap.read()
    port2in_onePlus[:,:,n] = f[:,:,0]

import time

for n in range(len(state_start_video_frame)):
               cv2.imshow('Frame',port2in_actual[:,:,n])
             