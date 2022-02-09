#%% look at single cell responses to poke in vs withdraw responses to Left, Center, and Right ports:
    
#%%set event window in seconds, set target neuron of interest
sideChoice=dataFrame['response_side']
rightRewardIDX=[]
leftRewardIDX=[]
rewardIDX=[]
for i in range(0,len(DemonRewardOnset)):
    rewardIDX.append(DemonRewardOnset[i][0])

for i in range(0, len(rewardIDX)):
    if sideChoice[rewardIDX[i]] == 0.0:
        leftRewardIDX.append(i)
    if sideChoice[rewardIDX[i]] == 1.0:
        rightRewardIDX.append(i) 

print(rightRewardIDX)
print(leftRewardIDX)
#%% create binary vector for punishment trials
rightPunishIDX=[]
leftPunishIDX=[]
punishIDX=[]
for i in range(0,len(DemonWrongChoiceOnset)):
    punishIDX.append(DemonWrongChoiceOnset[i][0])

for i in range(0, len(rewardIDX)):
    if sideChoice[punishIDX[i]] == 0.0:
        leftPunishIDX.append(i)
    if sideChoice[punishIDX[i]] == 1.0:
        rightPunishIDX.append(i) 

print(rightPunishIDX)
print(leftPunishIDX)

#%%set event window in seconds, set target neuron of interest
window=1/(average_interval/1000) #define event window
neuronNum=89

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Single-Cell Response to Right, Center, & Left Port Poke Initiation')

eventNames= ['DemonInitFixation']
for j in range(0,len(eventNames)):
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    eventTraceMidPort=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #trial x time array
    
    for i in range(0,len(tempEvent)): #for every reward trial
        
        startIDX=int(tempEvent[i][1]-window*0.5) #set startIDX
        endIDX=int(tempEvent[i][1]+window*1.5) #set endIDX
        
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
        
        if len(tempTrace) == math.floor(window*2):            
            eventTraceMidPort[i,:]=tempTrace
        else:
            eventTraceMidPort[i,:]=tempTrace[:math.floor(window*2)]

eventNames= ['DemonReward'] 
for j in range(0,len(eventNames)):
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    eventTraceReward=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #trial x time array
    
    for i in range(0,len(tempEvent)): #for every reward trial
        
        startIDX=int(tempEvent[i][1]-window*0.5) #set startIDX
        endIDX=int(tempEvent[i][1]+window*1.5) #set endIDX
        
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
        
        if len(tempTrace) == math.floor(window*2):            
            eventTraceReward[i,:]=tempTrace
        else:
            eventTraceReward[i,:]=tempTrace[:math.floor(window*2)]
            
eventNames= ['DemonWrongChoice']
for j in range(0,len(eventNames)):
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    eventTracePunish=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #trial x time array
    
    for i in range(0,len(tempEvent)): #for every reward trial
        
        startIDX=int(tempEvent[i][1]-window*0.5) #set startIDX
        endIDX=int(tempEvent[i][1]+window*1.5) #set endIDX
        
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
        
        if len(tempTrace) == math.floor(window*2):            
            eventTracePunish[i,:]=tempTrace
        else:
            eventTracePunish[i,:]=tempTrace[:math.floor(window*2)]
            

eventTraceRightPort=np.vstack((eventTraceReward[rightRewardIDX],eventTracePunish[rightPunishIDX]))
eventTraceLeftPort=np.vstack((eventTraceReward[leftRewardIDX],eventTracePunish[leftPunishIDX]))

ax1.plot(eventTraceRightPort.transpose(),linestyle='--',color='#FC7659',linewidth=0.3)
ax2.plot(eventTraceMidPort.transpose(),linestyle='--',color='#808080',linewidth=0.3)
ax3.plot(eventTraceLeftPort.transpose(),linestyle='--',color='#ABBAFF',linewidth=0.3)
ax1.plot(eventTraceRightPort.mean(0), color = '#A2002F',linewidth=2)
ax2.plot(eventTraceMidPort.mean(0),color='k',linewidth=2)
ax3.plot(eventTraceLeftPort.mean(0), color='#2800A2',linewidth=2)

ax1.set_title('Right Port')
ax2.set_title('Middle Port')
ax3.set_title('Left Port')
ax1.set_ylabel('Z-Score')

fig.canvas.draw()
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = np.arange(-1,5,0.5)
ax1.set_xticklabels(labels)

labels = [item.get_text() for item in ax2.get_xticklabels()]
labels = np.arange(-1,5,0.5)
ax2.set_xticklabels(labels)

labels = [item.get_text() for item in ax3.get_xticklabels()]
labels = np.arange(-1,5,0.5)
ax3.set_xticklabels(labels)

for ax in fig.get_axes():
    ax.axvline(x = window*0.5, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Onset (s)')
    ax.set_ylim([-0.5, 15])
plt.show()