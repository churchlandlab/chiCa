# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:44:50 2021

@author: Lukas Oesch
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys #To apppend the python path for the selection GUI
sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa') #Include the path to the functions
import load_cnmfe_outputs

#%%------load the cnmfe outputs and the alignment file

data_source = "C:/Users/Lukas Oesch/Documents/ChurchlandLab/TestDataChipmunk/TestMiniscopeAlignment/LO012/20210824_112833/caiman/secondRound.hdf5"
A, C, S, image_dims, frame_rate, neuron_num, recording_length, movie_file, spatial_spMat = load_cnmfe_outputs.load_data(data_source)
    


alignment_file = "C:/Users/Lukas Oesch/Documents/ChurchlandLab/TestDataChipmunk/TestMiniscopeAlignment/LO012/20210824_112833/trial_alignment/LO012_20210824_112833_trial_alignment.npz"

alignment_directory = np.load(alignment_file) #Fill in later
for key,val in alignment_directory.items(): #Retrieve all the entries 
        exec(key + '=val')

#%%-----Generate a vector with the time stamps matching the acquired frames (leaky)
time_vect = np.arange(acquired_frame_num + num_dropped) #The actual recording length.

leaky_time = np.array(time_vect) #initialize a copy where time stamps will be removed
for k in range(len(frame_drop_event)): #Iteratively remove the missing timestamps 
    if dropped_per_event[k] > 0:
        leaky_time = np.delete(leaky_time, np.arange(frame_drop_event[k],frame_drop_event[k] + dropped_per_event[k]))
        #Make sure to iterate to increasing indices, so that the correct position is
        #is found
        
#%%----Calculate the cubic spline on the denoised calcium signal and apply a 
#  ----linear interpolation to the deconvolved spiking activity    

cubic_spline_interpolation = CubicSpline(leaky_time, C, axis=1)
denoised_interpolated = cubic_spline_interpolation(time_vect)

linear_interpolation = interp1d(leaky_time, S, axis=1)
deconvolved_interpolated = linear_interpolation(time_vect)


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