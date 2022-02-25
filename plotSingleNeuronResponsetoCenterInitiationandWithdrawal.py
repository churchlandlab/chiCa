#%% look at single cell responses to poke in vs withdraw responses to Left, Center, and Right ports:
    
#%%set event window in seconds, set target neuron of interest
window=1/(average_interval/1000) #define event window
neuronNum=89

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Single-Cell Response Center Port Poke Initiation & Withdrawal')

eventNames= ['DemonInitFixation']
for j in range(0,len(eventNames)):
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    eventTraceMidPort_init=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #trial x time array
    
    for i in range(0,len(tempEvent)): #for every reward trial
        
        startIDX=int(tempEvent[i][1]-window*0.5) #set startIDX
        endIDX=int(tempEvent[i][1]+window*1.5) #set endIDX
        
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
        
        if len(tempTrace) == math.floor(window*2):            
            eventTraceMidPort_init[i,:]=tempTrace
        else:
            eventTraceMidPort_init[i,:]=tempTrace[:math.floor(window*2)]
            
eventNames= ['DemonWaitForResponse']
for j in range(0,len(eventNames)):
    exec('tempEvent ='+ eventNames[j] + 'Onset')
    eventTraceMidPort_withdraw=np.zeros(shape=(len(tempEvent), math.floor(window*2))) #trial x time array
    
    for i in range(0,len(tempEvent)): #for every reward trial
        
        startIDX=int(tempEvent[i][1]-window*0.5) #set startIDX
        endIDX=int(tempEvent[i][1]+window*1.5) #set endIDX
        
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
        
        if len(tempTrace) == math.floor(window*2):            
            eventTraceMidPort_withdraw[i,:]=tempTrace
        else:
            eventTraceMidPort_withdraw[i,:]=tempTrace[:math.floor(window*2)]            

ax1.plot(eventTraceMidPort_init.transpose(),linestyle='--',color='#D3D3D3',linewidth=0.3)
ax2.plot(eventTraceMidPort_withdraw.transpose(),linestyle='--',color='#D3D3D3',linewidth=0.3)
ax1.plot(eventTraceMidPort_init.mean(0), color = 'k',linewidth=2)
ax2.plot(eventTraceMidPort_withdraw.mean(0),color='k',linewidth=2)

ax1.set_title('Initiation')
ax2.set_title('Withdrawal')
ax1.set_ylabel('Z-Score')

fig.canvas.draw()
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = np.arange(-1,5,0.5)
ax1.set_xticklabels(labels)

labels = [item.get_text() for item in ax2.get_xticklabels()]
labels = np.arange(-1,5,0.5)
ax2.set_xticklabels(labels)


for ax in fig.get_axes():
    ax.axvline(x = window*0.5, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Onset (s)')
    ax.set_ylim([-0.5, 15])
plt.show()