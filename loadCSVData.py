import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import math
#%%

csvData=pd.read_csv("C:/Users/Letizia/Desktop/LY Scripts/Chipmunk Scripts/20210825_164317/trialdata.csv")  #Slashes required, backslash doesn’t work unless you use the path library

calcium_traces_file = np.load("C:/Users/Letizia/Desktop/LY Scripts/Chipmunk Scripts/20210825_164317/calcium_traces_interpolated.npz") #This will create an npz file object

for key,val in calcium_traces_file.items(): #Retrieve all the entries and create variables with the respective name, here, C and S and the average #interval between frames, average_interval, which is 1/framerate.

        exec(key + '=val')
#%%
#Z-Score data and calculate average trace

zScoreC = stats.zscore(C)
avgTrace = zScoreC.mean(0)
