# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:40:51 2022

@author: Anne
"""

import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import time

#%% Let user select the video files to bin and concatenate

tk.Tk().withdraw()
file_names = filedialog.askopenfiles(filetypes = [('Avi Files', '*.avi')])

#%% Load, bin and concatenate the videos

scaling_factor = 0.25 #Bin 4 times

#Read first frame in order to determine the frame width and hight
cap = cv2.VideoCapture(file_names[0].name)
success, f = cap.read()
frame = cv2.resize(np.squeeze(f[:,:,0]), (int(scaling_factor*f.shape[1]), int(scaling_factor*f.shape[0])))
cap.release()
original_dimensions = [f.shape[1], f.shape[0]]

binned_video = np.array([]) #Probably obsolete

#Read in all the user-specified videos and do the conversions
for k in range(len(file_names)):
    cap = cv2.VideoCapture(file_names[k].name)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    binned =  np.empty([int(scaling_factor*original_dimensions[0]), int(scaling_factor*original_dimensions[1]), frame_num], dtype = 'uint8')
    for n in range(frame_num):
        success, f = cap.read()
        #video[:,:,n] = f[:,:,0]
        binned[:,:,n] = np.uintc(cv2.resize(f[:,:,0], (int(scaling_factor*original_dimensions[0]), int(scaling_factor*original_dimensions[1]))))
        print(f'{n}')
        
    if k == 0:
        binned_video = np.array(binned, dtype = 'uint8')
    else:
        binned_video = np.append(binned_video, binned, axis = 2)
    
    cap.release()    
    del binned
    
    print(f"Processed movie {k}")
    print('--------------------')
        
#%% Playing a subsection of the binned video --- does not work yet
# play_between = [150000, 151000]
# fps = 20

# for k in range(play_between[0], play_between[1]):
#     cv2.imshow('frame', binned_video[:,:,k])
#     time.sleep(1/fps)
    
#%% Save the binned movie to the same folder as the input videos

out_file_name = os.path.split(file_names[0].name)[0] + "/binned_movie.avi"
fps = 20 #Assume 20 fps here, this is will determine playing speed in a player but won't be a problem when reading the data in to python again

vid_out = cv2.VideoWriter(out_file_name,  cv2.VideoWriter_fourcc(*'mp4v'), fps, (binned_video.shape[1], binned_video.shape[0]), False)
#Arguments for VideoWriter are the file name, the codec, frame rate, the single frame width and height and whether it is color or not.
for k in range(binned_video.shape[2]):
    vid_out.write(binned_video[:,:,k])
    print(f'Wrote frame {k}')
vid_out.release()

#%% Auxiliary functionality for the miniscope synchronizer experiment

average_intensity = np.empty([binned_video.shape[2]]) * np.nan
for k in range(binned_video.shape[2]):
    average_intensity[k] = np.mean(np.mean(binned_video[:,:,k], axis = 0))
    
np.save(os.path.split(file_names[0].name)[0] + "/average_frame_intensity.npy",average_intensity)
    


