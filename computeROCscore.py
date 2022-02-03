from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#%%create binary vector for reward delivery
rewardVector=[0] * zScoreC.shape[1]
windowSize=int(1/(average_interval/1000))

for i in range(len(rewardOnsetFrame)):
    rewardVector[int(rewardOnsetFrame[i]):int(rewardOnsetFrame[i]+windowSize)] = [1] * windowSize

plt.figure()    
plt.plot(rewardVector) #plotting for sanity check
#%% Compute fpr, tpr, thresholds and roc auc for every cell
auROCScore=[]
neuronNum=[]
for i in range(len(zScoreC)):
    fpr, tpr, thresholds = roc_curve(rewardVector, zScoreC[i])
    roc_auc = auc(fpr, tpr)
    auROCScore.append(roc_auc)
    neuronNum.append(i)
#%%

sortedROCindices=np.argsort(auROCScore)[::-1]
sortedROCScore=auROCScore.sort()
sortedROCScore=auROCScore[::-1]
neuronROCVal=list(zip(sortedROCindices,sortedROCScore))
excCell=sortedROCindices[0]
inhCell=sortedROCindices[-1]

#%% plot a sample cell


fpr1, tpr1, thresholds = roc_curve(rewardVector, zScoreC[excCell])
roc_auc1 = auc(fpr1, tpr1)
fpr2, tpr2, thresholds = roc_curve(rewardVector, zScoreC[inhCell])
roc_auc2 = auc(fpr2, tpr2)

plt.figure()
plt.plot(fpr1, tpr1, label='ROC curve (area = %0.3f)' % roc_auc1, color ='r')
plt.plot(fpr2, tpr2, label='ROC curve (area = %0.3f)' % roc_auc2, color ='b')
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Sample Neuron ROC Curve')
plt.legend(loc="lower right")
#%%
plt.figure()

plt.imshow(avgTrace[sortedROCindices[0:10]],aspect='auto')
plt.axvline(x = window, color='w', linestyle='--', linewidth=0.5)
plt.colorbar()
plt.xlabel('Frame #')
plt.ylabel('Neuron #')
plt.title('Reward Triggered Average (Sorted by Most Active Cells)')
