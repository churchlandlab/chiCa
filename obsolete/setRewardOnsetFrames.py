#%%
#grab every trial start frame
trialStartFrame=csvData["trial_start_frame_index"].tolist()
rewardVar=csvData["DemonReward"] 

rewardTrialIDX=[]
rewardTrialStartFrame=[]
rewardDeliveryinSeconds = []
rewardDeliveryinFrames=[]

for i in range(len(trialStartFrame)): #for each trial
    
    if rewardVar[i] != '[nan nan]': #for entries that are not NaN
        rewardTrialIDX.append(i) #store the IDX numbers of the trials that have reward
        rewardTrialStartFrame.append(int(trialStartFrame[i]))
        
        for j in range(len(rewardVar[i])): #iterate through each string character
            if rewardVar[i][j].isspace(): #if the position of the space is found
                break
        temp = float(rewardVar[i][1:j])
        temp = temp/(average_interval/1000)
        rewardDeliveryinFrames.append(temp) #convert str to float and store this value in rewardDeliveryinSeconds
        
rewardTrialTimestamps=list(zip(rewardTrialIDX,rewardTrialStartFrame,rewardDeliveryinFrames))        
print(len(rewardTrialIDX)) 
print(rewardTrialTimestamps)       

#%% set reward onset frames
rewardOnsetFrame=[]

for i in range(len(rewardTrialTimestamps)):
    x=np.ceil((rewardTrialTimestamps[i][1]+rewardTrialTimestamps[i][2]))+1 #add one to mark frame that captures reward info after start
    rewardOnsetFrame.append(x)
print(rewardOnsetFrame)
#%% Check difference with correct frames VERIFY
rewardOnsetFrameCorrected=[4582,7344,9786,10023,12876,13048,13322,15495,15674,16103,16397,18197,18331,18753,19146,19782,20059,20324,20719,23309,23633,24045,25142,25951,26582,27992,28211,28670,29619,29966,30258,30914,31518,31879,34413,37101,37681,39283,40900,41189,43799,44689,45033,45864,49001,49489,49655,49845,51056,56339,56595,58679,59910,60477,61030,62815,64688,65076,74158,76944,87766,87915,89086,101642,101831,105541,105852,108392]
frameDiff=[]
for i in range(len(rewardOnsetFrame)):
    frameDiff.append(rewardOnsetFrameCorrected[i]-rewardOnsetFrame[i])
    
print(frameDiff)
   

#%% plot trial x reward event trace heatmap for individual neurons
window=1/(average_interval/1000) #define event window
eventTrace=np.zeros(shape=(len(rewardOnsetFrame), math.floor(window*2))) #trial x time array
#for one cell, plot trial x time heatmap
neuronNum=429 #set neuron number

for i in range(len(rewardTrialTimestamps)): #for every reward trial
    
    startIDX=int(rewardOnsetFrame[i]) #set startIDX
    endIDX=int(rewardOnsetFrame[i]+2*window) #set endIDX
    
    tempTrace=zScoreC[neuronNum][startIDX:endIDX]
    
    if len(tempTrace) == math.floor(window*2):            
        eventTrace[i,:]=tempTrace
    else:
        eventTrace[i,:]=tempTrace[:math.floor(window*2)]
#%%   

plt.figure()
plt.imshow(eventTrace.transpose(), aspect='auto')
plt.xlabel('Trial Number')
plt.ylabel('Frame #')
plt.title("Top Right Preferential Cell Response to Reward")

#%% plot trial x reward event trace for individual neurons

plt.figure()
for i in range(0,10): #for first 5 traces
    plt.plot(eventTrace[i] + (i+1)*5,'k')
plt.plot(eventTrace.mean(0)-5, 'r')
plt.axvline(x = window, color='k', linestyle='--', linewidth=0.5) 
plt.xlabel('Frame #')
plt.ylabel('dF/F')
plt.title("Sample Traces of Top ROC Selected Cell during Reward")
#%%
plt.figure()
plt.imshow(eventTrace,aspect='auto')
# plt.axvline(x = window, color='w', linestyle='--', linewidth=0.5) 
plt.xlabel('Frame')
plt.ylabel('Trial')
plt.title("Top ROC Selected Cell during Reward")
