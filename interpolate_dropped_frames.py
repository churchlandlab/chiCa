# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:44:50 2021

@author: Lukas Oesch
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tkinter import Tk #For interactive selection, this part is only used to withdraw() the little selection window once the selection is done.
import tkinter.filedialog as filedialog
import glob #Search files in directory

import sys #To apppend the python path for the selection GUI
sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa') #Include the path to the functions
import load_cnmfe_outputs

#%%-----Select the session folder and load the cnmfe outputs and the alignment file
    
# Select the session directory
Tk().withdraw() #Don't show the tiny confirmation window
directory_name = filedialog.askdirectory()

data_source = directory_name + "/caiman/secondRound.hdf5" #If the directory name is still there we can agian exploit the similar naming 
# Load  retrieve the data and load the memory-mapped movie
A, C, S, image_dims, frame_rate, neuron_num, recording_length, movie_file, spatial_spMat = load_cnmfe_outputs.load_data(data_source)

alignment_file = glob.glob(directory_name + '/trial_alignment/*.npz')
#In this implementation we create a separate folder to hold the alignment information,
#we can therefore look for the npz file in this folder. Later in datajoint we might just pull down the fed in data

for s in range(len(alignment_file)):
    alignment_directory = np.load(alignment_file[s]) #Fill in later
    for key,val in alignment_directory.items(): #Retrieve all the entries 
           exec(key + '=val')

#%%----Check whether frames were lost and interpolate if necessary

if num_dropped > 0: #The case with dropped frames
    
    #Generate a vector with the time stamps matching the acquired frames (leaky)
    time_vect = np.arange(acquired_frame_num + num_dropped) #The actual recording length.

    leaky_time = np.array(time_vect) #initialize a copy where time stamps will be removed
    for k in range(len(frame_drop_event)): #Iteratively remove the missing timestamps 
        if dropped_per_event[k] > 0:
            leaky_time = np.delete(leaky_time, np.arange(frame_drop_event[k],frame_drop_event[k] + dropped_per_event[k]))
            #Make sure to iterate to increasing indices, so that the correct position
            #is found
        
        #---Calculate the cubic spline on the denoised calcium signal and apply a 
        #---linear interpolation to the deconvolved spiking activity    

    cubic_spline_interpolation = CubicSpline(leaky_time, C, axis=1)
    C_interpolated = cubic_spline_interpolation(time_vect)

    linear_interpolation = interp1d(leaky_time, S, axis=1)
    S_interpolated = linear_interpolation(time_vect)
    
    print(f'Successfully interpolated {num_dropped} dropped frames')
    
else:
    C_interpolated = C
    S_interpolated = S

#%%---Save these two traces to a file inside the trial_alignment folder in this implementation
output_file = directory_name + "/trial_alignment/interpolated_calcium_traces.npz"
np.savez(output_file, C_interpolated = C_interpolated, S_interpolated = S_interpolated)

#%%----auxiliary visualization tools

# look_at_before = leaky_time[(leaky_time > 11150) & (leaky_time < 11200)].tolist()
# look_at_after = time_vect[(time_vect > 11150) & (time_vect < 11200)].tolist()

# plt.figure()
# for k in range(denoised_interpolated.shape[0]):
#     be = plt.plot(leaky_time[look_at_before], C[k, look_at_before])
#     af = plt.plot(time_vect[look_at_after], denoised_interpolated[k, look_at_after])
#     plt.waitforbuttonpress()
#     be[0].remove()
#     af[0].remove()