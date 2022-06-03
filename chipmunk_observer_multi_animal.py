# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 20:25:59 2022

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
animal_IDs = ['LO022', 'LO033']
animal_data = [None] * len(animal_IDs)
date_motif = '_20'


#Start finding the files and loading the data
for s in range(len(animal_IDs)):
    proto_frame = dict()
    proto_frame['session_file'] = glob.glob(os.path.join(data_dir, animal_IDs[s], '*.obsmat')) #Grab all the sessions
    proto_frame['session_date'] = [None] * len(proto_frame['session_file'])
    proto_frame['wait_time'] = [None] * len(proto_frame['session_file'])
    proto_frame['early_withdrawal'] = [None] * len(proto_frame['session_file'])
    proto_frame['reaction_time'] = [None] * len(proto_frame['session_file'])
    proto_frame['initiation_delay'] = [None] * len(proto_frame['session_file'])
    proto_frame['valid_trials'] = [None] * len(proto_frame['session_file'])
    
    animal_data[s] = pd.DataFrame(proto_frame)
    for k in range(len(animal_data[s])):
            #get the date and time of the session
            idx = animal_data[s]['session_file'][k].find(date_motif)
            date_and_time = str.split(animal_data[s]['session_file'][k][idx+1:idx+16], sep='_') #Retrieve the data and time
            animal_data[s]['session_date'][k] = datetime.datetime.strptime(date_and_time[0] + date_and_time[1], '%Y%m%d%H%M%S')
            
            #Start loading the files and retrieving their information
            sesdata = loadmat(animal_data[s]['session_file'][k], squeeze_me=True,
                                  struct_as_record=True)['SessionData']
                
            try: 
                animal_data[s]['valid_trials'][k] = np.array(sesdata['ObsDidNotInitiate'].tolist()) == 0
                animal_data[s]['early_withdrawal'][k] = np.array(sesdata['ObsEarlyWithdrawal'].tolist())
                animal_data[s]['wait_time'][k] = np.array(sesdata['ObsActualWaitTime'].tolist())
                animal_data[s]['reaction_time'][k] = np.array(sesdata['ObsActualWaitTime'].tolist()) - np.array(sesdata['ObsSetWaitTime'].tolist())
                
                intermediate = []
                tmp = sesdata['RawEvents'].tolist()
                tmp = tmp['Trial'].tolist()
                trialstates = [None] * tmp.shape[0]
                xx = []
                a = {'DemonInitFixation':None, 'ObsInitFixation':None}
                for t in range(tmp.shape[0]):
                    w = []
                    w = tmp[t]['States'].tolist()
                    a['DemonInitFixation'] = w['DemonInitFixation'].tolist()
                    a['ObsInitFixation'] = w['ObsInitFixation'].tolist()
                    trialstates[t] = a
                    print(f'{trialstates[t]}')
                    xx.append(a)
                    
                trialstates = pd.DataFrame(trialstates)
                
                for n in range(len(trialstates)-1):
                    if animal_data[s]['valid_trials'][k][n]:                       
                       intermediate.append(trialstates['DemonInitFixation'][n][1] - trialstates['ObsInitFixation'][n][0])
                animal_data[s]['initiation_delay'][k] = intermediate
             
            except: 
                pass
    #Finallly sort by the session data by date
    animal_data[s] = animal_data[s].sort_values('session_date').reset_index(drop = True)
    #Reset index drop = True will force pandas to drop the indices and insted recreate them   
    