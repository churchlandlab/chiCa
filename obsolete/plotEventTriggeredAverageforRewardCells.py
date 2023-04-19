#%% event triggered avg activity of all cells to look at population representation
#%% Set window and event labels
window=2/(average_interval/1000) #define event window
eventNames= ['DemonDidNotInitiate','DemonInitFixation','DemonEarlyWithdrawal',
             'PlayStimulus', 'DemonGoCue','DemonReward', 'DemonWrongChoice']
neuronIDX= sortedROCindices[0:50]
#%%
plt.figure() 
plt.suptitle('Task-triggered Average of Top 50 ROC-Reward Cells')

for j in range(0,len(eventNames)):
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    avgTrace=np.zeros(shape=(len(neuronIDX),math.floor(window*2)))    #create an empty arraw that is numNeuron x window size   
    for neuronNum in range(0,len(neuronIDX)): #for each neuron        
        eventTrace=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #trial x time array
        
        for i in range(0,len(tempEvent)): #for every task trial
            
            startIDX=int(tempEvent[i][1]-window*0.5) #set startIDX
            endIDX=int(tempEvent[i][1]+window*1.5) #set endIDX
            
            tempTrace=zScoreC[neuronIDX[neuronNum]][startIDX:endIDX]
            
            if len(tempTrace) == math.floor(window*2):            
                eventTrace[i,:]=tempTrace
            else:
                eventTrace[i,:]=tempTrace[:math.floor(window*2)]        
                
        avgTrace[neuronNum]=eventTrace.mean(0) #take the mean of all the event so you get a vector of one cell x event
    
    ax = plt.subplot(2, 4,j+1)
    plt.imshow(avgTrace, aspect='auto')
    plt.axvline(x = window*0.5, color='w', linestyle='--', linewidth=0.5)
    ax.set_title(eventNames[j])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Neuron #')
