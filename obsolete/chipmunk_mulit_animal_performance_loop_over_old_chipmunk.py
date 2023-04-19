# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:03:09 2022

@author: Lukas Oesch
"""

import numpy as np
import os 
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tkinter import Tk #For interactive selection, this part is only used to withdraw() the little selection window once the selection is done.
import tkinter.filedialog as filedialog
import glob #To pick files of a specified type
import pandas as pd
import datetime

#%%
# Tk().withdraw() #Don't show the tiny confirmation window
# directory_name = filedialog.askopenfiles()
data_dir = 'C:/Users/Lukas Oesch/Documents/ChurchlandLab/TestDataChipmunk/Data'

#Specify animals that go into the analysis
animal_IDs = ['LO010', 'LO011', 'LO014', 'LO029', 'LO030']
animal_data = [None] * len(animal_IDs)
date_motif = '_20'


#Start finding the files and loading the data
for s in range(len(animal_IDs)):
    proto_frame = dict()
    proto_frame['session_file'] = glob.glob(os.path.join(data_dir, animal_IDs[s], '*.mat')) #Grab all the sessions
    proto_frame['session_date'] = [None] * len(proto_frame['session_file'])
    proto_frame['performance'] = [None] * len(proto_frame['session_file'])
    proto_frame['wait_time'] = [None] * len(proto_frame['session_file'])
    proto_frame['early_withdrawal'] = [None] * len(proto_frame['session_file'])
    proto_frame['reaction_time'] = [None] * len(proto_frame['session_file'])
    
    animal_data[s] = pd.DataFrame(proto_frame)
    for k in range(len(animal_data[s])):
            #get the date and time of the session
            idx = animal_data[s]['session_file'][k].find(date_motif)
            date_and_time = str.split(animal_data[s]['session_file'][k][idx+1:idx+16], sep='_') #Retrieve the data and time
            animal_data[s]['session_date'][k] = datetime.datetime.strptime(date_and_time[0] + date_and_time[1], '%Y%m%d%H%M%S')
            
            #Start loading the files and retrieving their information
            try:
                sesdata = loadmat(animal_data[s]['session_file'][k], squeeze_me=True,
                                  struct_as_record=True)['SessionData']
                
                #This is the new chipmunk stuff
                try:
                    #get the performance on easiest only
                    stim_rate = np.array(sesdata['StimulusRate'].tolist()) #Get the stimulus rates
                    modality = np.where(np.isnan(stim_rate[0,:]) == 0)[0] #Find the modality for the session
                    easiest_stim = np.logical_or(stim_rate[:,modality]==4, stim_rate[:,modality]==20)
                    correct_resp = np.array(sesdata['CorrectResponse'].tolist())
                    valid_trials = np.array(sesdata['ValidTrials'].tolist())
                    animal_data[s]['performance'][k] = np.sum(correct_resp[easiest_stim[:,0]])/np.sum(valid_trials[easiest_stim[:,0]])
                    
                    
                    animal_data[s]['early_withdrawal'][k] = np.sum(np.array(sesdata['EarlyWithdrawal'].tolist())) / (sesdata['nTrials'] - np.sum(np.array(sesdata['DidNotInitiate'].tolist())))
                    animal_data[s]['wait_time'][k] = np.nanmedian(np.array(sesdata['ActualWaitTime'].tolist()))
                
                    tmp = np.array(sesdata['ActualWaitTime'].tolist()) - (np.array(sesdata['SetWaitTime'].tolist()) + np.array(sesdata['PreStimDelay'].tolist()) )
                    animal_data[s]['reaction_time'][k] = np.median(tmp[np.array(sesdata['ValidTrials'].tolist())==1])
                except:
                    pass
            except: 
                pass
    #Finallly sort by the session data by date
    animal_data[s] = animal_data[s].sort_values('session_date').reset_index(drop = True)
    #Reset index drop = True will force pandas to drop the indices and insted recreate them   
    
#%% ----plot performance

include_mice = [0, 1, 2, 3, 4]
sessions_back = 15
cover_bin = 0.4 
vect = np.arange(-cover_bin , cover_bin , (cover_bin*2)/sessions_back )


#---------
fi = plt.figure()
ax = fi.add_subplot(111)
plot_lines = [None] * len(include_mice)
for s in range(len(include_mice)):
    frame_size = len(animal_data[s])
    tmp = animal_data[s]['performance'][frame_size - sessions_back : frame_size]
    vectX = vect+s
    
    plot_lines[s] = ax.plot(vectX, tmp, label = animal_IDs[include_mice[s]])
  

ax.set_ylim(0.3,1)
ax.set_xticks(np.arange(len(include_mice)))
ax.set_xticklabels([animal_IDs[i] for i in include_mice])
  
ax.set_xlabel('Animal ID')
ax.set_ylabel('Fraction correct on easiest')
ax.set_title(f'Performance over the last {sessions_back} sessions', fontweight='bold')


#----
fi = plt.figure()
ax = fi.add_subplot(111)
plot_lines = [None] * len(include_mice)
for s in range(len(include_mice)):
    frame_size = len(animal_data[s])
    tmp = animal_data[s]['wait_time'][frame_size - sessions_back : frame_size]
    vectX = vect+s
    
    plot_lines[s] = ax.plot(vectX, tmp, label = animal_IDs[include_mice[s]])
  

ax.set_ylim(0,2)
ax.set_xticks(np.arange(len(include_mice)))
ax.set_xticklabels([animal_IDs[i] for i in include_mice])
  
ax.set_xlabel('Animal ID')
ax.set_ylabel('Median wait time (s)')
ax.set_title(f'Wait time over the last {sessions_back} sessions', fontweight='bold')

#---------------
fi = plt.figure()
ax = fi.add_subplot(111)
plot_lines = [None] * len(include_mice)
for s in range(len(include_mice)):
    frame_size = len(animal_data[s])
    tmp = animal_data[s]['early_withdrawal'][frame_size - sessions_back : frame_size]
    vectX = vect+s
    
    plot_lines[s] = ax.plot(vectX, tmp, label = animal_IDs[include_mice[s]])
  

ax.set_ylim(0,1)
ax.set_xticks(np.arange(len(include_mice)))
ax.set_xticklabels([animal_IDs[i] for i in include_mice])
  
ax.set_xlabel('Animal ID')
ax.set_ylabel('Early withdrawal rate')
ax.set_title(f'Early withdrawal rate over the last {sessions_back} sessions', fontweight='bold')



