import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge, Lasso
from sklearn import metrics
#%% formatting Y_outcome trials binary vector of right or left choice
sideChoice=dataFrame['response_side']
outcomeIDX=[]
for i in range(0, len(sideChoice)):
    if sideChoice[i] == 0.0:
        outcomeIDX.append([i,0])
    if sideChoice[i] == 1.0:
        outcomeIDX.append([i,1])
PlayStimulusOutcomeIDX=[]
for i in range(len(column(PlayStimulusOnset,0))):
    if column(PlayStimulusOnset,0)[i] in column(outcomeIDX,0):
        PlayStimulusOutcomeIDX.append(PlayStimulusOnset[i])

frameDiff=[]
for i in range(len(PlayStimulusOutcomeIDX)):
    frameDiff.append(column(DemonGoCueOnset,1)[i]-column(PlayStimulusOutcomeIDX,1)[i])
    
print(frameDiff)
#%% set start and end IDX from stimulus play to go cue
Y_labels = np.array(column(outcomeIDX,1))
print(Y_labels)
#%% format X_neural activity vector: design matrix

sliding_window_performance=[]
shuffled_performance=[]


for sliding_window in np.arange(-5,30):
    averageTrialActivity=np.zeros(shape=(len(zScoreC),len(PlayStimulusOutcomeIDX)))  
    Y_labels_shuffle=shuffle(Y_labels)
    
    for neuronNum in range(len(zScoreC)): #for each neuron        
        singleTrialActivity=np.empty((len(PlayStimulusOutcomeIDX), 30)) #trial x time array
        singleTrialActivity[:] = np.nan
        
        for i in range(len(PlayStimulusOutcomeIDX)): #for every task trial
                           
            startIDX=int(column(PlayStimulusOutcomeIDX,1)[i]+sliding_window) #set startIDX
            endIDX=int(column(DemonGoCueOnset,1)[i]+sliding_window) #set endIDX
    
            singleTrialActivity[i,:endIDX-startIDX]=zScoreC[neuronNum][startIDX:endIDX]
        averageTrialActivity[neuronNum]=np.nanmean(singleTrialActivity,1) # design matrix
             
    # plt.imshow(averageTrialActivity)
    # plt.xlabel('Outcome Trials')
    # plt.ylabel('Cells')

    X_design_matrix = averageTrialActivity
    X_design_matrix=X_design_matrix.transpose()
    
    clf_svm = SVC(kernel='linear')
    clf_svm_shuffle = SVC(kernel='linear')
    b_coeff_mat=[]
    b_coeff_mat_shuffle=[]
    y_pred_acc=[]
    y_pred_acc_shuffle=[]
    numFolds=10
    kf = KFold(n_splits=numFolds)
    
        
    for train, test in kf.split(X_design_matrix): #OBSERVED
        X_train, X_test, y_train, y_test = X_design_matrix[train], X_design_matrix[test], Y_labels[train], Y_labels[test]
        
        clf_svm.fit(X_train, y_train)
        
        y_pred_acc.append(clf_svm.score(X_test, y_test))
        b_coeff_mat.append(clf_svm.coef_)
        #print(clf_svm.predict(X_test))
        #print(y_test)
        #print(np.mean(b_coeff_mat,0)) #10-Fold Averaged B-Coefficients of Cells
        #print('Test Accuracy:', clf_svm.score(X_test, y_test)) #10-Fold Accuracy Scores for Each Fold
    for train, test in kf.split(X_design_matrix): #OBSERVED
        X_train, X_test, y_train, y_test = X_design_matrix[train], X_design_matrix[test], Y_labels_shuffle[train], Y_labels_shuffle[test]
        
        clf_svm_shuffle.fit(X_train, y_train)
        
        y_pred_acc_shuffle.append(clf_svm_shuffle.score(X_test, y_test))
        b_coeff_mat_shuffle.append(clf_svm_shuffle.coef_)
    
    print('Overall Model Accuracy:', np.mean(y_pred_acc)) #Averaged Model performance over 10 folds
    sliding_window_performance.append(np.mean(y_pred_acc))
    shuffled_performance.append(np.mean(y_pred_acc_shuffle))

#%% plotting
fig = plt.figure()
plt.plot(sliding_window_performance, 'k', linewidth=2,label='Observed')
plt.plot(shuffled_performance,color='#ABBAFF',linewidth=0.5,label='Shuffled')
plt.axvline(x = 5, color='k', linestyle='--', linewidth=0.7)
plt.axvline(x = 25, color='k', linestyle='--',linewidth=0.7)
plt.legend(loc='upper right')
plt.title('Decoder Performance')
plt.ylabel('Model Performance')
plt.xlabel('Stimulus Onset Time (ms)') #1 frame ~50 ms
plt.xticks(np.arange(0, 36, step=5))

#%%
plt.figure()

plt.hist(np.mean(b_coeff_mat,0))
plt.show() 