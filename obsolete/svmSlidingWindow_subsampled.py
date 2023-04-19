import sklearn
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge, Lasso
from sklearn import metrics
from collections import Counter
from sklearn.datasets import make_classification

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
    
#%% Checking for frame difference 
frameDiff=[]
for i in range(len(PlayStimulusOutcomeIDX)):
    frameDiff.append(column(DemonGoCueOnset,1)[i]-column(PlayStimulusOutcomeIDX,1)[i])
    
print(frameDiff)
#%% set start and end IDX from stimulus play to go cue
Y_labels = np.array(column(outcomeIDX,1))
print(Y_labels)
#%% Set minority vs majority class labels:
print(Counter(Y_labels))

if Y_labels.tolist().count(0)<Y_labels.tolist().count(1):
    min_class=0
    maj_class=1
else:
    min_class=1
    maj_class=0

min_class_indices = [index for index, element in enumerate(Y_labels.tolist()) if element == min_class]
maj_class_indices = [index for index, element in enumerate(Y_labels.tolist()) if element == maj_class]

#%% format X_neural activity vector: design matrix
sliding_window_performance=[]
shuffled_performance=[]

for sliding_window in np.arange(-5,25):
    averageTrialActivity=np.zeros(shape=(len(zScoreC),len(PlayStimulusOutcomeIDX)))  
    Y_event_vector_shuffle=shuffle(Y_event_vector)
    
    for neuronNum in range(len(zScoreC)): #for each neuron        
        eventTrace=np.empty((len(playstim_outcomeIDX), 30)) #trial x time array
        eventTrace[:] = np.nan
        for i in range(len(playstim_outcomeIDX)): #for every task trial
                       
            startIDX=int(column(playstim_outcomeIDX,1)[i]+sliding_window) #set startIDX
            endIDX=int(column(DemonGoCueOnset,1)[i]+sliding_window) #set endIDX
                
            tempTrace=zScoreC[neuronNum][startIDX:endIDX]
                
            eventTrace[i,:endIDX-startIDX]=tempTrace
                    
        averageTrialActivity[neuronNum]=np.nanmean(eventTrace,1) # design mat
    #Making design matrix
    
    X_design_matrix = averageTrialActivity
    X_design_matrix=X_design_matrix.transpose()
    
    #Build model for 1000 subsampled iterations
    subsampled_acc=[]
    subsampled_b_coeff_mat=[]
    subsampled_acc_shuffle=[]
    subsampled_b_coeff_mat_shuffle=[]
    
    for i in range(100):
        subsampled_maj_class_indices=random.sample(maj_class_indices,len(min_class_indices))
        subsampled_indices=sorted(min_class_indices+subsampled_maj_class_indices)
    
        Y_labels_subsampled=Y_labels[subsampled_indices]
        X_design_matrix_subsampled=X_design_matrix[subsampled_indices]
    
        clf_svm = SVC(kernel='linear')
        clf_svm_shuffle = SVC(kernel='linear')
        b_coeff_mat=[]
        b_coeff_mat_shuffle=[]
        y_pred_acc=[]
        y_pred_acc_shuffle=[]
        numFolds=10
        kf = KFold(n_splits=numFolds)
        
        for train, test in kf.split(X_design_matrix_subsampled): #SHUFFLE
            X_train, X_test, y_train, y_test = X_design_matrix_subsampled[train], X_design_matrix_subsampled[test], Y_labels_subsampled[train], Y_labels_subsampled[test]
            
            clf_svm_shuffle.fit(X_train, y_train)
            
            y_pred_acc_shuffle.append(clf_svm_shuffle.score(X_test, y_test))
            b_coeff_mat_shuffle.append(clf_svm_shuffle.coef_)
            
            subsampled_acc_shuffle.append(np.mean(y_pred_acc_shuffle))
            subsampled_b_coeff_mat_shuffle=np.mean(b_coeff_mat_shuffle,0)
            
        for train, test in kf.split(X_design_matrix_subsampled): #OBSERVED
            X_train, X_test, y_train, y_test = X_design_matrix_subsampled[train], X_design_matrix_subsampled[test], Y_labels_subsampled[train], Y_labels_subsampled[test]
            
            clf_svm.fit(X_train, y_train)
            
            y_pred_acc.append(clf_svm.score(X_test, y_test))
            b_coeff_mat.append(clf_svm.coef_)
            
            subsampled_acc.append(np.mean(y_pred_acc))
            subsampled_b_coeff_mat=np.mean(b_coeff_mat,0)
            # print(clf_svm.predict(X_test))
            # print(y_test)
            #print(np.mean(b_coeff_mat,0)) #10-Fold Averaged B-Coefficients of Cells
            #print('Test Accuracy:', clf_svm.score(X_test, y_test)) #10-Fold Accuracy Scores for Each Fold
            #print('Overall Model Accuracy:', np.mean(y_pred_acc)) #Averaged Model performance over 10 folds
    sliding_window_performance.append(np.mean(subsampled_acc))
    shuffled_performance.append(np.mean(subsampled_acc_shuffle))    
    print('The Average Subsampled Model Performance is:', np.mean(subsampled_acc))
#%%
plt.figure()
plt.plot(sliding_window_performance, label='Observed')
plt.plot(shuffled_performance,linewidth=0.5, color='gray', label='Shuffle')
plt.xlabel('Play Stimulus Onset')
plt.axvline(x = 5, color='k', linestyle='--')
plt.axvline(x = 25, color='k', linestyle='--')
plt.ylabel('Model Accuracy')

#%%
fig1, ax1 = plt.subplots()
ax1.set_title('Distribution of Subsampled Model Performance Scores (1000-Fold)')
ax1.boxplot(np.mean(subsampled_b_coeff_mat,0))
ax1.xticks([])
ax1.ylabel('Model Performance')
#%%
plt.figure()
plt.hist(np.mean(subsampled_b_coeff_mat,0), bins=50,color='gray')
plt.title('Distribution of Model Weights')
plt.xlabel('Weights')
plt.ylabel('Frequency')