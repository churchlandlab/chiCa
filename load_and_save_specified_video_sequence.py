# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:55:48 2022

@author: Lukas Oesch
"""


import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import time
import matplotlib.pyplot as plt
#%% Let user select the video files to bin and concatenate

tk.Tk().withdraw()
file_names = filedialog.askopenfiles(filetypes = [('Avi Files', '*.avi')])

#%% Load movies

start_frame = 21600 #Assumes the movies are synchronized
frames_to_load = 900
videos = [None] * len(file_names)
for s in range(len(file_names)):

    #Read first frame in order to determine the frame width and hight
    cap = cv2.VideoCapture(file_names[s].name)
    success, f = cap.read()
    frame = f[:,:,1]
    original_dimensions = [f.shape[1], f.shape[0]]
    
    movie = np.zeros([frame.shape[0], frame.shape[1], frames_to_load])
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);
    print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    for k in range(900):
        success, f = cap.read()
        movie[:,:,k] = f[:,:,1]
        
    cap.release()
    videos[s] = movie

#%%----Dirty and simple: just smush the movies together horizintally with a little separation
horizontal_separation = np.zeros([videos[0].shape[0], 50, frames_to_load])

concatenated_video = np.uint8(np.hstack((videos[0], horizontal_separation, videos[1])))

#%%--- Concatenated frames to video
out_file_name = os.path.split(file_names[0].name)[0] + '/concatenated_video.avi'
fps = 30
vid_out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (concatenated_video.shape[1], concatenated_video.shape[0]), False)
for k in range(concatenated_video.shape[2]):
    vid_out.write(concatenated_video[:,:,k])
    print(f'Wrote frame {k}')
vid_out.release()

