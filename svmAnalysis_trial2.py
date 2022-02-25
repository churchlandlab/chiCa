import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn import metrics
#%% formatting Y_outcome trials binary vector of right or left choice
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
#%%
rightRewardVector=np.zeros((len(DemonRewardOnset),1))
rightRewardVector[rightRewardIDX]=1
rightRewardVector=np.ravel(rightRewardVector)
#%% set start and end IDX from stimulus play to go cue
stimplayIDX=[]
for i in range(0,len(DemonRewardOnset)): #for every index number in reward
    for j in range(0,len(DemonGoCueOnset)): #for every go cue onset
        if column(DemonRewardOnset,0)[i] == column(DemonGoCueOnset,0)[j]: #if reward index number equals gocue index number
            stimplayIDX.append(column(PlayStimulusOnset,1)[j]) #append the frame number of the play stimulus onset
Y_event_vector = rightRewardVector #set to right choice
#%% format X_neural activity vector: 423 neuron x 68 observations

avgTrace=np.zeros(shape=(len(zScoreC),len(stimplayIDX)))    #create an empty arraw that is numNeuron x window size   

for neuronNum in range(0,len(zScoreC)): #for each neuron        
    eventTrace=np.zeros(((len(stimplayIDX), 10))) #trial x time array
    for i in range(0,len(stimplayIDX)): #for every task trial
                   
        startIDX=int(stimplayIDX[i]) #set startIDX
        endIDX=int(stimplayIDX[i]+10) #set endIDX
            
        tempTrace=zScoreC[neuronNum][startIDX:endIDX]
            
        if len(tempTrace) == 10:            
            eventTrace[i,:]=tempTrace
        else:
            eventTrace[i,:]=tempTrace[:10]        
                
    avgTrace[neuronNum]=eventTrace.mean(1) # 432 cell/feature X 68 outcome trial matrix
     
plt.imshow(avgTrace)
plt.xlabel('Outcome Trials')
plt.ylabel('Cells')
#%% run the svm
X_neuron_trace = avgTrace
X_neuron_trace=X_neuron_trace.transpose()
#%%
clf_svm = SVC(kernel='linear')
b_coeff_mat=[]
y_pred_acc=[]
numFolds=10
kf = KFold(n_splits=numFolds)


for train, test in kf.split(X_neuron_trace):
    X_train, X_test, y_train, y_test = X_neuron_trace[train], X_neuron_trace[test], Y_event_vector[train], Y_event_vector[test]
    
    clf_svm.fit(X_train, y_train)
    y_pred_acc.append(clf_svm.score(X_test, y_test))
    b_coeff_mat.append(clf_svm.coef_)
    #print(clf_svm.predict(X_test))
    #print(test)

#print(np.mean(b_coeff_mat,0)) #10-Fold Averaged B-Coefficients of Cells
#print(y_pred_acc) #10-Fold Accuracy Scores for Each Fold
print('Model Accuracy:', np.mean(y_pred_acc)) #Averaged Model performance over 10 folds
#%%
for Model in [Ridge, Lasso]:
    model = Model()
    print('%s: %s' % (Model.__name__, cross_val_score(model, X_neuron_trace, Y_event_vector,cv=10).mean()))
#%% using Lasso Regression:
    
lasso_model = Lasso()
y_pred_lasso_acc=[]
b_coeff_lasso=[]
kf = KFold(n_splits=10)

for train, test in kf.split(X_neuron_trace):
    X_train, X_test, y_train, y_test = X_neuron_trace[train], X_neuron_trace[test], Y_event_vector[train], Y_event_vector[test]
    
    lasso_model.fit(X_train, y_train)
    #print(lasso_model.predict(X_test))
    #print(y_test)
    y_pred_lasso_acc.append(lasso_model.score(X_test,y_test))
    b_coeff_lasso.append(lasso_model.coef_)

print(np.mean(b_coeff_lasso,0))
print(y_pred_lasso_acc)
print(np.mean(y_pred_lasso_acc))