#set event window in seconds, set target neuron of interest
window=2/(average_interval/1000) #define event window
neuronNum=89 
eventNames= ['DemonDidNotInitiate','DemonInitFixation','DemonEarlyWithdrawal',
             'PlayStimulus', 'DemonGoCue','DemonReward', 'DemonWrongChoice']
#%%
plt.figure() 
plt.suptitle('Single-Cell Response to Task Events')
for j in range(0,len(eventNames)):
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    eventTrace=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #trial x time array
    
    for i in range(1,len(tempEvent)): #for every reward trial
        
        startIDX=int(tempEvent[i][1]-window) #set startIDX
        endIDX=int(tempEvent[i][1]+window) #set endIDX
        
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
        
        if len(tempTrace) == math.floor(window*2):            
            eventTrace[i,:]=tempTrace
        else:
            eventTrace[i,:]=tempTrace[:math.floor(window*2)]
    ax = plt.subplot(2, 4,j+1)
    ax.imshow(eventTrace, aspect='auto')
    ax.axvline(x = window, color='w', linestyle='--', linewidth=0.5)
    ax.set_title(eventNames[j])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Trial')