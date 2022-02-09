import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import math
#%%
dataFrame=pd.read_hdf("C:/Users/Letizia/Desktop/LY Scripts/Chipmunk Scripts/trialdata_20210825_164317.h5","/Data")
print(dataFrame.columns)
#%%
calcium_traces_file = np.load("C:/Users/Letizia/Desktop/LY Scripts/Chipmunk Scripts/20210825_164317/calcium_traces_interpolated.npz") #This will create an npz file object
for key,val in calcium_traces_file.items(): #Retrieve all the entries and create variables with the respective name, here, C and S and the average #interval between frames, average_interval, which is 1/framerate.

        exec(key + '=val')
#%%
#Z-Score data and calculate average trace

zScoreC = stats.zscore(C)
avgTrace = zScoreC.mean(0)
#%% From Lukas' script to find start frames, modified
def find_state_start_frame(state_name, trialdata, average_interval, trial_start_time_covered):
    '''Locate the frame during which a certain state in the chipmunk task has
    started. Requires state_name (string with the name of the state of interest)
    trialdata (a pandas dataframe with the trial start frames
    and the state timers) and average interval (the average frame interval as 
    as recorded by teensy).'''
    
    state_start_frame = [None] * len(trialdata) #The frame that covers the start of
    state_time_covered = np.zeros([len(trialdata)]) #The of the side that has been covered by the frame
    
    for n in range(len(trialdata)-1): #Subtract one here because the last trial is unfinished
          if np.isnan(trialdata[state_name][n][0]) == 0: #The state has been visited
              frame_time = np.arange(trial_start_time_covered[n]/1000, (trialdata["trial_start_frame_index"][n+1] - trialdata["trial_start_frame_index"][n]) * average_interval/1000,  average_interval/1000)
              #Generate frame times starting the first frame at the end of its coverage of trial inforamtion

              tmp = frame_time - trialdata[state_name][n][0] #Calculate the time difference
              state_start_frame[n] = int(np.where(tmp > 0)[0][0] + trialdata["trial_start_frame_index"][n])
              #np.where returns a tuple where the first element are the indices that fulfill the condition.
              #Inside the array of indices retrieve the first one that is positive, therefore the first
              #frame that caputres some information.
         
              state_time_covered[n] =  tmp[tmp > 0][0] #Retrieve the time that was covered by the frame
          else:
              state_start_frame[n] = np.nan
              state_time_covered[n] = np.nan
          
    return state_start_frame, state_time_covered
#%% Set the IDX
for eventName in dataFrame.columns[9:-1]:

    temp=find_state_start_frame(eventName, dataFrame, average_interval, dataFrame['trial_start_time_covered'])[0]  
    tempOnset = [x for x in temp if pd.isnull(x) == False]
    tempIDX=[]
    for i in range(0,len(tempOnset)):
        tempIDX.append(temp.index(tempOnset[i]))
    
    exec(eventName + 'Onset=[list(x) for x in zip(tempIDX, tempOnset)]')
#%%
for eventName in dataFrame.columns[0:9]:

    temp=find_state_start_frame(eventName, dataFrame, average_interval, dataFrame['trial_start_time_covered'])[0]  
    tempOnset = [x for x in temp if pd.isnull(x) == False]
    tempIDX=[]
    for i in range(0,len(tempOnset)):
        tempIDX.append(temp.index(tempOnset[i]))
    
    exec(eventName + 'Onset=[list(x) for x in zip(tempIDX, tempOnset)]')
