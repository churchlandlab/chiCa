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
playstim_outcomeIDX=[]
for i in range(len(column(PlayStimulusOnset,0))):
    if column(PlayStimulusOnset,0)[i] in column(outcomeIDX,0):
        playstim_outcomeIDX.append(PlayStimulusOnset[i])
frameDiff=[]
for i in range(len(playstim_outcomeIDX)):
    frameDiff.append(column(DemonGoCueOnset,1)[i]-column(playstim_outcomeIDX,1)[i])
    
print(frameDiff)
#%% set start and end IDX from stimulus play to go cue
Y_event_vector = np.array(column(outcomeIDX,1))
#%% format X_neural activity vector: design matrix
sliding_window_performance=[]
shuffled_performance=[]

for sliding_window in np.arange(0,10):
    avgTrace=np.zeros(shape=(len(zScoreC),len(playstim_outcomeIDX)))    #create an empty arraw that is numNeuron x window size   
    Y_event_vector_shuffle=shuffle(Y_event_vector)
    
    for neuronNum in range(len(zScoreC)): #for each neuron        
        eventTrace=np.empty((len(playstim_outcomeIDX), 30)) #trial x time array
        eventTrace[:] = np.nan
        for i in range(len(playstim_outcomeIDX)): #for every task trial
                       
            startIDX=int(column(playstim_outcomeIDX,1)[i]+sliding_window) #set startIDX
            endIDX=int(column(DemonGoCueOnset,1)[i]+sliding_window) #set endIDX
                
            tempTrace=zScoreC[neuronNum][startIDX:endIDX]
                
            eventTrace[i,:endIDX-startIDX]=tempTrace
                    
        avgTrace[neuronNum]=np.nanmean(eventTrace,1) # 432 cell/feature X 68 outcome trial matrix
         
    # plt.imshow(avgTrace)
    # plt.xlabel('Outcome Trials')
    # plt.ylabel('Cells')

    X_neuron_trace = avgTrace
    X_neuron_trace=X_neuron_trace.transpose()

    clf_svm = SVC(kernel='linear')
    clf_svm_shuffle = SVC(kernel='linear')
    b_coeff_mat=[]
    b_coeff_mat_shuffle=[]
    y_pred_acc=[]
    y_pred_acc_shuffle=[]
    numFolds=10
    kf = KFold(n_splits=numFolds)
    
    
    for train, test in kf.split(X_neuron_trace): #OBSERVED
        X_train, X_test, y_train, y_test = X_neuron_trace[train], X_neuron_trace[test], Y_event_vector[train], Y_event_vector[test]
        
        clf_svm.fit(X_train, y_train)
        y_pred_acc.append(clf_svm.score(X_test, y_test))
        b_coeff_mat.append(clf_svm.coef_)
        #print(clf_svm.predict(X_test))
        #print(test)
    for train, test in kf.split(X_neuron_trace): #SHUFFLE
        X_train, X_test, y_train, y_test = X_neuron_trace[train], X_neuron_trace[test], Y_event_vector_shuffle[train], Y_event_vector_shuffle[test]
        
        clf_svm_shuffle.fit(X_train, y_train)
        y_pred_acc_shuffle.append(clf_svm_shuffle.score(X_test, y_test))
        b_coeff_mat_shuffle.append(clf_svm_shuffle.coef_)
        #print(clf_svm.predict(X_test))
        #print(test)
    
    #print(np.mean(b_coeff_mat,0)) #10-Fold Averaged B-Coefficients of Cells
    #print(y_pred_acc) #10-Fold Accuracy Scores for Each Fold
    sliding_window_performance.append(np.mean(y_pred_acc))
    shuffled_performance.append(np.mean(y_pred_acc_shuffle))
    print('Model Accuracy:', np.mean(y_pred_acc)) #Averaged Model performance over 10 folds
    #print('Shuffle Model Accuracy:', np.mean(y_pred_acc_shuffle))
#%% plot the sliding window scores:

plt.figure()
plt.plot(sliding_window_performance)
#plt.plot(shuffled_performance,linewidth=0.5)
plt.xlabel('Play Stimulus Onset')
plt.axvline(x = 5, color='k', linestyle='--')
plt.ylabel('Model Accuracy')

