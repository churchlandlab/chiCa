
#%%
window=2/(average_interval/1000) #define event window
eventTrace=np.zeros(shape=(len(rightChoice), math.floor(window*2))) #trial x time array
#for one cell, plot trial x time heatmap
neuronNum=89 #set neuron number

for i in range(len(rightChoice)): #for every right reward trial
    
    startIDX=int(rewardOnsetFrame[i]-window) #set startIDX
    endIDX=int(rewardOnsetFrame[i]+window) #set endIDX
    
    tempTrace=zScoreC[neuronNum][startIDX:endIDX]
    
    if len(tempTrace) == math.floor(window*2):            
        eventTrace[i,:]=tempTrace
    else:
        eventTrace[i,:]=tempTrace[:math.floor(window*2)]
plt.figure()
plt.imshow(eventTrace,aspect='auto')
plt.axvline(x = window, color='w', linestyle='--', linewidth=0.5) 
plt.xlabel('Frame')
plt.ylabel('Trial')
plt.title("Right Selective Cell during Reward")