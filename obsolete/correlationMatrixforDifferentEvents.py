#%% Set window and neuron
window=1/(average_interval/1000) #define event window
eventNames= ['DemonDidNotInitiate','DemonInitFixation','DemonEarlyWithdrawal',
             'PlayStimulus', 'DemonGoCue','DemonReward', 'DemonWrongChoice']
#%%
plt.figure() 
plt.suptitle('Sample Single-Cell Response to Task Events')
for j in range(0,len(eventNames)): #for each task event
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    eventTrace=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #create trial x time array
    
    for i in range(0,len(tempEvent)): #for every reward trial
        
        startIDX=int(tempEvent[i][1]-window*0.5) #set startIDX
        endIDX=int(tempEvent[i][1]+window*1.5) #set endIDX
        
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
        
        if len(tempTrace) == math.floor(window*2):            
            eventTrace[i,:]=tempTrace
        else:
            eventTrace[i,:]=tempTrace[:math.floor(window*2)]
            
    covarMatrix = np.cov(eventTrace)
    ax = plt.subplot(2, 4,j+1)
    ax.imshow(covarMatrix, aspect='auto')
    ax.set_title(eventNames[j])
