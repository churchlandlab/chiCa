#%% time permute function for each neuron
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#%%create binary vector for reward delivery
rewardVector=[0] * zScoreC.shape[1]
windowSize=int(1/(average_interval/1000))

for i in range(len(rewardOnsetFrame)):
    rewardVector[int(rewardOnsetFrame[i]):int(rewardOnsetFrame[i]+windowSize)] = [1] * windowSize
#%%

for n in [114]:

        # #calculate ROC score on each trace
        # fpr, tpr, thresholds = roc_curve(rewardVector, zScoreC[n])
        # neuronROC = auc(fpr, tpr)
        
        # #bootstrap with time permuted trace
        permutedMatrix = np.zeros((1000,zScoreC.shape[1]))
        for i in range(0,1000): #loop 1000 times
            permTrace= [0]*zScoreC.shape[1]
            randIDX=random.choice(range(zScoreC.shape[1]))       #choose random index number  
            permTrace[randIDX+1:]=zScoreC[n][0:randIDX]
            permTrace[0:randIDX]=zScoreC[n][randIDX+1:]
            permutedMatrix[i][:]=permTrace
        auROCScoreRand=[]
        for i in range(len(permutedMatrix)):
            fpr, tpr, thresholds = roc_curve(rewardVector, permutedMatrix[i])
            roc_auc = auc(fpr, tpr)
            auROCScoreRand.append(roc_auc)
        
        #sort all ROC scores
        auROCScoreRand.sort()        


#%%
excCellIDX=[]
excCellScore=[]
excCellData=[]

for n in range(0,len(neuronROCVal)):
    if neuronROCVal[n][1] > auROCScoreRand[990]:
        excCellIDX.append(neuronROCVal[n][0])
        excCellScore.append(neuronROCVal[n][1])
        
excCellData=list(zip(excCellIDX,excCellScore))
print(len(excCellData)) #132 cells
#%%
inhCellIDX=[]
inhCellScore=[]
inhCellData=[] 
for n in range(0,len(neuronROCVal)):           
    if neuronROCVal[n][1] < auROCScoreRand[10]:
        inhCellIDX.append(neuronROCVal[n][0])
        inhCellScore.append(neuronROCVal[n][1])
           
inhCellData=list(zip(inhCellIDX,inhCellScore))
print(len(inhCellData)) #65 Cells

#%% plot observed trace vs shuffled trace to check
plt.figure()

plt.plot(zScoreC[neuronNum][:])
plt.plot(permTrace,'r')
plt.axvline(x = randIDX, color='k', linestyle='--')
