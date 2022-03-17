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


#Assume standard data structure with the one unique file with the desired file
#extension in the respective folder.
mscope_log_file = glob.glob(directory_name + '/miniscope/*.mscopelog') #Use glob to select all instances of files defined by the file extension.
mscope_tStamps_file = directory_name + '/miniscope/timeStamps.csv'

chipmunk_file = glob.glob(directory_name + '/chipmunk/*.mat')
if len(chipmunk_file) == 0: #This is the case when it is an obsmat file, for instance
    chipmunk_file = glob.glob(directory_name + '/chipmunk/*.obsmat')
    if  len(chipmunk_file) == 0: #Not copied
        print("It looks like the chipmunk behavior file has not yet been copied to this folder")
        
    
camlog_file = glob.glob(directory_name + '/chipmunk/*.camlog')

#%%--- all the loading here

mscope_log = np.loadtxt(mscope_log_file[0], delimiter = ',', skiprows = 2) #Make sure to skip over the header lines and use comma as delimiter
mscope_time_stamps = np.loadtxt(mscope_tStamps_file, delimiter = ',', skiprows = 1)

temp = loadmat(chipmunk_file[0], squeeze_me=True, struct_as_record=True) #Load the behavioral data file 
chipmunk_data = temp['SessionData']

video_tracking = []
camera_name = []
for k in camlog_file:
    video_tracking.append(np.loadtxt(k, delimiter = ',', skiprows = 6, comments='#'))
    #In the log file certain lines are preceded by #, meaning that these lines
    #comments that should be disregarded in the alignment.
    with open(k) as f:
        first_line = f.readline() #Read the first line only to retrieve the camera 
    camera_name.append(str.split(first_line, sep=' ')[2])

#%%-------Align trial start times for the miniscope 
# Remember that channel 23 is the Bpod input and channel 2 the miniscope frames,
# first column is the channel number, second column the event number, third column
# the teensy time stamp.

#Make sure the teensy file starts with miniscope frames and not trial info
if mscope_log[0, 0] == 23:
     raise ValueError(f"Possibly the behavior was started before the imaging. Please check {mscope_log_file[0]}.")

#In some of the log files large gaps were seen between the first and the second
#catured frame. This happens when the logging start automatically already but then
#the scope DAQ has to be unplugged from the computer to properly start the scope 
#acquisition. Remove these instances.
#If first two events are putative frame captures but the second one happens more than a second after the first there is a problem
if ((mscope_log[0, 0] == 2) & (mscope_log[1, 0] == 2)) & (mscope_log[1,2] - mscope_log[0,2] > 1000): 
  #TODO: check for multiple replugging events and raise error if not enough frames are left...
    mscope_log = mscope_log[1:mscope_log.shape[0],:] #Remove first timestamp that can't correspond to frame
    warnings.warn(f"The first putative frame in the mscopelog file occurs 1 s before the next one. It will be cut out to align the data. Please check {mscope_log_file[0]}")

trial_starts  = np.where(mscope_log[:,0] == 23) #Find all trial starts
trial_start_frames = mscope_log[trial_starts[0]+1,1] #Set the frame just after
# trial start as the start frame, the trial start takes place in the exposure time of the frame.
trial_start_time_covered = mscope_log[trial_starts[0]+1,2] - mscope_log[trial_starts[0],2] 
# Keep the trial time that was covered by the frame acquisition
# Check whether the number of trials is as expected
trial_number_matches = trial_start_frames.shape[0] == (int(chipmunk_data['nTrials']) + 1)
# Adding one to the number of trials is necessary becuase nTrials measures completed trials

#%%-----Find the trial starts on the video tracking 

trial_start_video_frame = [] #The frames that captured the start of a trial
average_video_frame_interval = []#The average interval between the frames 

for n in range(len(video_tracking)):
    #Since the camera channels that record the trial TTL and the frame acquisition
    #are not defined we identify them based on their frequency of occurence
    channel_identifiers = list(set(video_tracking[n][:,2])) #Find unique channel identifiers (there should only be two!)
    if len(channel_identifiers) > 2:
          warnings.warn("More than two TTL signals were detected on the camera log. Please check the log file.")
          
    count_input_TTL = [0] * 2 
    count_input_TTL[0] = np.where(video_tracking[n][:,2]==channel_identifiers[0])[0].shape[0]
    #Very complicated term. Finds the occurences of the specified TTL isentity and accesses the 
    #the shape of the array where they are strored in within the tuple output from np.where.
    count_input_TTL[1] = np.where(video_tracking[n][:,2]==channel_identifiers[1])[0].shape[0]

    #Assign the TTL pulse identity to the camera or Bpod
    if count_input_TTL[0] > count_input_TTL[1]:
       camera_channel = int(channel_identifiers[0])
       trial_channel = int(channel_identifiers[1])
    else:
        camera_channel = int(channel_identifiers[1])
        trial_channel = int(channel_identifiers[0])  
    print(f"Aligning camera {camera_name[n]} frames on channel {camera_channel} and trial TTLs on {trial_channel}")

    #Here a loop might be more readable
    trial_start = []
    for k in range(video_tracking[n].shape[0]-1):
        if video_tracking[n][k,2] == camera_channel and video_tracking[n][k+1,2] == trial_channel:
            trial_start.append(k)

    #Append the existing lists for these variables
    trial_start_video_frame.append(trial_start)
    average_video_frame_interval.append(np.mean(np.diff(video_tracking[n][:,1])))

#%%------Check for dropped frames-----
teensy_frames_collected = np.where(mscope_log[:,0] == 2)[0]
#These are the frames that the miniscope DAQ acknowledges as recorded.

teensy_frames = np.transpose(mscope_log[teensy_frames_collected,2])
teensy_frames = teensy_frames - teensy_frames[0] #Zero the start

miniscope_frames = mscope_time_stamps[:,1] #These are the frames have been received by the computer and written to the HD.
miniscope_frames = miniscope_frames - miniscope_frames[0]

num_dropped = teensy_frames.shape[0] - miniscope_frames.shape[0]


#%%-----match the teensy frames with the actually recorded ones and detect position of dropped

average_interval = np.mean(np.diff(teensy_frames, axis = 0), axis = 0)
#Determine th expected interval between the frames from the teensy output.

teensy_cumulative_time = np.cumsum(teensy_frames) #Generate a continuous time line
miniscope_cumulative_time = np.cumsum(miniscope_frames)

acquired_frame_num = miniscope_frames.shape[0]
clock_time_difference = miniscope_frames - teensy_frames[0:acquired_frame_num]
#Map the frame times 1:1 here first to already reveal unexpected jumps in the time difference

clock_sync_slope = (miniscope_frames[-1] - teensy_frames[-1])/acquired_frame_num
#This variable is a rough estimate of the slope of the desync between the 
# CPU clock and the teensy clock.
intercept_term = 0 #This is expected to be 0 in the initial segment of the line

# Generate line and subtract from the data to obtain "rectified" version of the data
line_estimate = (np.arange(0,acquired_frame_num) - 1) * clock_sync_slope + intercept_term
residuals = clock_time_difference - line_estimate

# Compute the first derivative of the residuals
diff_residuals = np.diff(residuals,axis=0)

#Find possible frame drop events
#candidate_drops = [0] + np.array(np.where(diff_residuals > 0.5*average_interval))[0,:].tolist() + [residuals.shape[0]] #Mysteriously one gets a tuple of indices and zeros
candidate_drops = [0] + np.array(np.where(diff_residuals > (average_interval - average_interval*0.1)))[0,:].tolist() + [residuals.shape[0]] #Mysteriously one gets a tuple of indices and zeros
#Turn into list and add the 0 in front and the teensy signal in the end to make looping easier.

frame_drop_event = []
last_frame_drop = 0 #Keep a pointer to the last frame drop that occured to iteratively lengthen the window to average over
#Go through the candidates and assign as an event if the following frames are
#all shifted more than one expected frame duration (with 20% safety margin).
#Whenever a candidate event does not qualify as a frame drop merge the residuals
#after that event into the current evaluation process.
for k in range(1,len(candidate_drops)-1):
    if  (np.mean(residuals[candidate_drops[k]+1:candidate_drops[k+1]]+1) - np.mean(residuals[last_frame_drop + 1:candidate_drops[k]]+1)) > (average_interval - (average_interval*0.2)):
    #if (np.mean(residuals[k+1:k + window]) - np.mean(residuals[k - window:k])) > (average_interval - (average_interval*0.1)):
        #Adding two here seeks to avoid some of the overshoot that can be seen when the computer is trying to catch up.
        frame_drop_event.append(candidate_drops[k])
        last_frame_drop = candidate_drops[k]

#Estimate the number of frames dropped per 
segments = [0] + frame_drop_event + [residuals.shape[0]]

dropped_per_event = [None] * len(frame_drop_event) #Estimated dropped frames per event
rounding_error = [None] * len(frame_drop_event) #Estimates confidence in precise number of dropped
jump_size = [None] * len(frame_drop_event) #The estimated shift in time differencebetween the teensz
# and the miniscope log when a frame is dropped

for k in range(len(frame_drop_event)):
    jump_size[k] = np.mean(residuals[segments[k+1]+1:segments[k+2]+1]) - np.mean(residuals[segments[k]+1:segments[k+1]+1])
    
    #Divide the observed jump in the signal by the expected interval
    approx_loss_per_event = jump_size[k] / average_interval
    dropped_per_event[k] = round(approx_loss_per_event)
    if dropped_per_event[k] > 0:
        rounding_error[k] = (dropped_per_event[k] - approx_loss_per_event)/dropped_per_event[k]
    else:
        rounding_error[k] = dropped_per_event[k] - approx_loss_per_event
    
#Verify the number of dropped frames
if np.sum(dropped_per_event) == num_dropped:
    print(f"Matched {num_dropped} dropped frames in the miniscope recording")
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
        jump_size = jump_size, rounding_error = rounding_error, average_interval = average_interval,
        acquired_frame_num = acquired_frame_num,
        trial_start_time_covered = trial_start_time_covered, trial_start_video_frame = trial_start_video_frame,
        average_video_frame_interval = average_video_frame_interval, camera_name = camera_name)
