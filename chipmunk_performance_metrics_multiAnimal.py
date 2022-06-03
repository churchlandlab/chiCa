# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 12:15:34 2022

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

#%% --Start checking for the files to plot and sort them 


#Set the data directory
data_dir = 'C:/Users/Lukas Oesch/Documents/ChurchlandLab/TestDataChipmunk/Data'

#Specify animals that go into the analysis
animal_IDs = ['LO027', 'LO028', 'LO031', 'LO032', 'LO033', 'LO029', 'LO030']
animal_data = [None] * len(animal_IDs)
date_motif = '_20'
# session_files = [ [] for _ in range(len(animal_IDs)) ] #Generate empty list that will hold
# #of all the session files from the different animals
# session_dates = [ [] for _ in range(len(animal_IDs)) ] #Retrieve the dates of the sessions and store 
# date_motif = '_20' #All dates are separated from the previous information by an
# #underscore and start with a 20. This does not interfer with the mouse ids because
# #their 20s are not preceded by an underscore but by LO

#Start finding the files and loading the data
for s in range(len(animal_IDs)):
    proto_frame = dict()
    proto_frame['session_file'] = glob.glob(os.path.join(data_dir, animal_IDs[s] , '*.mat')) #Grab all the sessions
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
        sesdata = loadmat(animal_data[s]['session_file'][k], squeeze_me=True,
                          struct_as_record=True)['SessionData']
        
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
    
    #Finallly sort by the session data by date
    animal_data[s] = animal_data[s].sort_values('session_date').reset_index(drop = True)
    #Reset index drop = True will force pandas to drop the indices and insted recreate them   
    
#%%------Average over days if multiple sessions were performed and deal with all the dates

#Plotting performance over learning
include_mice = [0, 1, 2, 3, 4]
per_day = True
break_criterion = 5

date = [None] * len(include_mice) #Dates of the respective sessions
performance = [None] * len(include_mice) #Performance during the session with the corresponding date

for s in range(len(include_mice)):
    date_all_sessions = []
    

    for k in range(len(animal_data[include_mice[s]])):
        date_all_sessions.append(animal_data[include_mice[s]]['session_date'][k].date())
            
    date_all_sessions = np.array(date_all_sessions)
    date_individual = np.unique(np.array(date_all_sessions))
    performance_individual = np.zeros(date_individual.shape[0])
    
    for n in range(date_individual.shape[0]):
                idx = np.where(date_all_sessions == date_individual[n])[0]
                performance_individual[n] = np.nanmean(animal_data[include_mice[s]]['performance'][idx])

    #Find breaks that are longer than 7d and insert nan to split the line later
    t_delta = np.diff(date_individual)
    training_breaks = [-1]
    for n in range(len(t_delta)):
        if t_delta[n].days > break_criterion :
            training_breaks.append(n)

    if not training_breaks:
        date_plot  = date_individual
        performance_plot = performance_individual
    else:
        date_plot = []
        performance_plot = []
        for n in range(len(training_breaks)-1):
            date_plot = date_plot + date_individual[training_breaks[n]+1:training_breaks[n+1]+1].tolist() + [date_individual[training_breaks[n+1]] + datetime.timedelta(days=1)]
            performance_plot = performance_plot + performance_individual[training_breaks[n]+1:training_breaks[n+1]+1].tolist() + [np.nan]
        
    date_plot = date_plot + date_individual[training_breaks[-1]+1:date_individual.shape[0]].tolist()
    performance_plot = performance_plot + performance_individual[training_breaks[-1]+1:date_individual.shape[0]].tolist()
    
    date[include_mice[s]] = np.array(date_plot)
    performance[include_mice[s]] = np.array(performance_plot)
#%% ----Now plotting

fi = plt.figure(figsize=(12,6))
plt.rc('font', size=15)
ax = fi.add_subplot(111)
perf_ax = [None] * len(include_mice)

for s in range(len(performance)):
    perf_ax[s] = ax.plot(date[s],performance[s], linewidth = 2, label = animal_IDs[include_mice[s]])
    
ax.set_ylim(0.3, 1)
ax.grid()
ax.legend(loc = 'lower right')

tmp = ax.get_xticks()
tmp = tmp[0:-1:2]
ax.set_xticks(tmp)

ax.set_xlabel('Session date')
ax.set_ylabel('Fraction correct on easiest')
ax.set_title('Animal performance over days', fontweight='bold')

#%%----Scatter summary of last sessions

sessions_back = 12

fi = plt.figure()
ax = fi.add_subplot(111)
scatter_dots = [None] * len(include_mice)
average_line = [None] * len(include_mice)

for s in range(len(include_mice)):
    frame_size = len(performance[s])
    tmp = performance[s][frame_size - sessions_back : frame_size]
    tmp_mean = np.nanmean(tmp)
    jitter = np.random.normal(s, 0.1, size=len(tmp))
    
    scatter_dots[s] = ax.scatter(jitter, tmp, label = animal_IDs[include_mice[s]])
    average_line[s] = ax.hlines(tmp_mean,s-0.2, s+0.2, color='k')
    

ax.set_ylim(0.3,1)
ax.set_xticks(np.arange(len(include_mice)))
ax.set_xticklabels([animal_IDs[i] for i in include_mice])
  
ax.set_xlabel('Animal ID')
ax.set_ylabel('Fraction correct on easiest')
ax.set_title(f'Average performance over the last {sessions_back} sessions', fontweight='bold')

#%% ---- wait time 
sessions_back = 12

fi = plt.figure()
ax = fi.add_subplot(111)
scatter_dots = [None] * len(include_mice)
average_line = [None] * len(include_mice)

for s in range(len(include_mice)):
    frame_size = len(animal_data[include_mice[s]])
    tmp = animal_data[include_mice[s]]['wait_time'][frame_size - sessions_back : frame_size]
    tmp_mean = np.mean(tmp)
    jitter = np.random.normal(s, 0.1, size=len(tmp))
    
    scatter_dots[s] = ax.scatter(jitter, tmp, label = animal_IDs[include_mice[s]])
    average_line[s] = ax.hlines(tmp_mean,s-0.2, s+0.2, color='k')
    

ax.set_ylim(0,1.5)
ax.set_xticks(np.arange(len(include_mice)))
ax.set_xticklabels([animal_IDs[i] for i in include_mice])
  
ax.set_xlabel('Animal ID')
ax.set_ylabel('Wait time (s)')
ax.set_title(f'Median wait time over the last {sessions_back} sessions', fontweight='bold')


#%%----- Early withdrawal rate
sessions_back = 12

fi = plt.figure()
ax = fi.add_subplot(111)
scatter_dots = [None] * len(include_mice)
average_line = [None] * len(include_mice)

for s in range(len(include_mice)):
    frame_size = len(animal_data[include_mice[s]])
    tmp = animal_data[include_mice[s]]['early_withdrawal'][frame_size - sessions_back : frame_size]
    tmp_mean = np.mean(tmp)
    jitter = np.random.normal(s, 0.1, size=len(tmp))
    
    scatter_dots[s] = ax.scatter(jitter, tmp, label = animal_IDs[include_mice[s]])
    average_line[s] = ax.hlines(tmp_mean,s-0.2, s+0.2, color='k')
    

ax.set_ylim(0,1)
ax.set_xticks(np.arange(len(include_mice)))
ax.set_xticklabels([animal_IDs[i] for i in include_mice])
  
ax.set_xlabel('Animal ID')
ax.set_ylabel('Early withdrawal rate')
ax.set_title(f'Early withdrawal rate over the last {sessions_back} sessions', fontweight='bold')

#%%------Reaction time
sessions_back = 12

fi = plt.figure()
ax = fi.add_subplot(111)
scatter_dots = [None] * len(include_mice)
average_line = [None] * len(include_mice)

for s in range(len(include_mice)):
    frame_size = len(animal_data[include_mice[s]])
    tmp = animal_data[include_mice[s]]['reaction_time'][frame_size - sessions_back : frame_size]
    tmp_mean = np.mean(tmp)
    jitter = np.random.normal(s, 0.1, size=len(tmp))
    
    scatter_dots[s] = ax.scatter(jitter, tmp, label = animal_IDs[include_mice[s]])
    average_line[s] = ax.hlines(tmp_mean,s-0.2, s+0.2, color='k')
    

ax.set_ylim(0,0.6)
ax.set_xticks(np.arange(len(include_mice)))
ax.set_xticklabels([animal_IDs[i] for i in include_mice])
  
ax.set_xlabel('Animal ID')
ax.set_ylabel('Go cue reaction time (s)')
ax.set_title(f'Median reaction time over the last {sessions_back} sessions', fontweight='bold')