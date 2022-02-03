dataFrame=pd.read_hdf("C:/Users/Letizia/Desktop/LY Scripts/Chipmunk Scripts/trialdata_20210824_112833.h5","/Data")
print(dataFrame.columns)
#%%
calcium_traces_file = np.load("C:/Users/Letizia/Desktop/LY Scripts/Chipmunk Scripts/20210825_164317/calcium_traces_interpolated.npz") #This will create an npz file object

for key,val in calcium_traces_file.items(): #Retrieve all the entries and create variables with the respective name, here, C and S and the average #interval between frames, average_interval, which is 1/framerate.

        exec(key + '=val')
#%%
#Z-Score data and calculate average trace

zScoreC = stats.zscore(C)
avgTrace = zScoreC.mean(0)
