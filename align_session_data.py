# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:04:00 2021

@author: Lukas Oesch
"""
import numpy as np
from scipy.io import loadmat
from tkinter import Tk #For interactive selection
import tkinter.filedialog as filedialog
import glob #Search files in directory
import matplotlib.pyplot as plt
import warnings #Throw warning for unmatched drops
import os #To create a directory


#%%--------- Sort out the location of the files

# Select the session directory
Tk().withdraw() #Don't show the tiny confirmation window
directory_name = filedialog.askdirectory()


#Assume standard data structure
mscope_log_file = glob.glob(directory_name + '/miniscope/*.mscopelog')
mscope_tStamps_file = directory_name + '/miniscope/timeStamps.csv'

chipmunk_file = glob.glob(directory_name + '/chipmunk/*.mat')
camlog_file = glob.glob(directory_name + '/chipmunk/*.camlog')
#%%--- all the loading here

mscope_log = np.loadtxt(mscope_log_file[0], delimiter = ',', skiprows = 2) #Make sure to skip over the header lines and use comma as delimiter
mscope_time_stamps = np.loadtxt(mscope_tStamps_file, delimiter = ',', skiprows = 1)

temp = loadmat(chipmunk_file[0], squeeze_me=True, struct_as_record=True)
chipmunk_data = temp['SessionData']

video_tracking =  np.loadtxt(camlog_file[0], delimiter = ',', skiprows = 6, comments='#')
# State here the comments explicitly again

#%%-------Align trial start times
# Remember that channel 23 is the Bpod input and channel 2 the miniscope frames,
# first column is the channel number, second column the event number, third column
# the teensy time stamp.

trial_starts  = np.where(mscope_log[:,0] == 23) #Find all trial starts
trial_start_frames = mscope_log[trial_starts[0]+1,1] #Set the frame just after
# trial start as the start frame, the trial start takes place in the exposure time of the frame.

# Check whether the number of trials is as expected
trial_number_matches = trial_start_frames.shape[0] == (int(chipmunk_data['nTrials']) + 1)
# Adding one to the number of trials is necessary becuase nTrials measures completed trials

#%%------Check for dropped frames-----
teensy_frames_collected = np.where(mscope_log[:,0] == 2)[0]
teensy_frames = np.transpose(mscope_log[teensy_frames_collected,2])
teensy_frames = teensy_frames - teensy_frames[0] #Zero the start

miniscope_frames = mscope_time_stamps[:,1]
miniscope_frames = miniscope_frames - miniscope_frames[0]

num_dropped = teensy_frames.shape[0] - miniscope_frames.shape[0]


#%%-----match the teensy frames with the actually recorded ones and detect position of dropped

average_interval = np.mean(np.diff(teensy_frames, axis = 0), axis = 0)

teensy_cumulative_time = np.cumsum(teensy_frames) #Generate a continuous time line
miniscope_cumulative_time = np.cumsum(miniscope_frames)

acquired_frame_num = miniscope_frames.shape[0]
clock_time_difference = miniscope_frames - teensy_frames[0:acquired_frame_num]

clock_sync_slope = (miniscope_frames[-1] - teensy_frames[-1])/acquired_frame_num
#This variable is a rough estimate of the slope of the desync between the 
# CPU clock and the teensy clock.
intercept_term = 0 #This is expected to be 0 in the initial segment of the line

# Generate line and subtrct from the data to obtain "rectified" version of the data
line_estimate = (np.arange(0,acquired_frame_num) - 1) * clock_sync_slope + intercept_term
residuals = clock_time_difference - line_estimate

# Compute the first derivative of the residuals
diff_residuals = np.diff(residuals,axis=0)

#Find possible frame drop events
candidate_drops = np.array(np.where(diff_residuals > average_interval))[0,:] #Mysteriously one gets a tuple of indices and zeros

frame_drop_event = []
window = 100
#Go through the candidates and assign as an event if the following frames are all shifted more than one expected frame duration (with 10% safety margin)
for k in candidate_drops:
    if (np.mean(residuals[k+1:k + window]) - np.mean(residuals[k - window:k])) > (average_interval - (average_interval*0.1)):
        frame_drop_event.append(k)

#Estimate the number of frames dropped per 
segments = list(frame_drop_event)
segments.insert(0,0)
segments.append(miniscope_frames.shape[0])

dropped_per_event = [None] * len(frame_drop_event) #Estimated dropped frames per event
rounding_error = [None] * len(frame_drop_event) #Estimates confidence in precise number of dropped
jump_size = [None] * len(frame_drop_event) #The estimated shift in time differencebetween the teensz
# and the miniscope log when a frame is dropped

for k in range(len(frame_drop_event)):
    jump_size = np.mean(residuals[segments[k+1]+1:segments[k+2]+1]) - np.mean(residuals[segments[k]+1:segments[k+1]+1])
    
    #Divide the observed jump in the signal by the expected interval
    approx_loss_per_event = jump_size / average_interval
    dropped_per_event[k] = round(approx_loss_per_event)
    if dropped_per_event[k] > 0:
        rounding_error[k] = (dropped_per_event[k] - approx_loss_per_event)/dropped_per_event[k]
    else:
        rounding_error[k] = dropped_per_event[k] - approx_loss_per_event
    
#Verify the number of dropped frames
if np.sum(dropped_per_event) == num_dropped:
    print(f"Matched {num_dropped} dropped frames in the recording")
    print("-----------------------------------------------------")
    
    if 0 in dropped_per_event: #Print an additional message if one of the drop events is empty
        warnings.warn("One or more empty drops have been detected. Please check all the outputs carefully")

else: 
    print("Dropped frames could not be localized inside the recording.")
    print("Please check manualy.")
    #return teensy_frames, miniscope_frames
    
    
#%%-----Plot summary of the matching
fi = plt.figure()
plt.plot(residuals)
plt.scatter(frame_drop_event, residuals[frame_drop_event], s=100, facecolors='none', edgecolors='r')
plt.legend(["Rectified frame time difference", "Frame drop event"])
plt.title("Detected dropping of single or multiple frames")
plt.xlabel("Frames")
plt.ylabel("Clock time difference between teensy and CPU (ms)")        
    
#%%-------- Save the obtained results to an easily python-readable file

#Generate a new directory if needed
if os.path.isdir(directory_name + "/trial_alignment") == False:
    os.mkdir(directory_name + "/trial_alignment")
    
    
#Get some info on session date and animal id    
temp = os.path.split(directory_name)
session_date = temp[1]
animalID = os.path.split(temp[0])[1]    

#Set the output file name and save relevant variables
output_file = directory_name + "/trial_alignment/" + animalID + "_" + session_date + "_" + "trial_alignment"    
np.savez(output_file, trial_start_frames = trial_start_frames, num_dropped = num_dropped,
        frame_drop_event = frame_drop_event, dropped_per_event = dropped_per_event,
        jump_size = jump_size, rounding_error = rounding_error, average_interval = average_interval)
