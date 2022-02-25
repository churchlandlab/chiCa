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

frameDiff=[]
for i in range(len(PlayStimulusOutcomeIDX)):
    frameDiff.append(column(DemonGoCueOnset,1)[i]-column(PlayStimulusOutcomeIDX,1)[i])
    
print(frameDiff)
#%% set start and end IDX from stimulus play to go cue
Y_labels = np.array(column(outcomeIDX,1))
print(Y_labels)
#%% format X_neural activity vector: design matrix

averageTrialActivity=np.zeros(shape=(len(zScoreC),len(PlayStimulusOutcomeIDX)))  

for neuronNum in range(len(zScoreC)): #for each neuron        
    singleTrialActivity=np.empty((len(PlayStimulusOutcomeIDX), 30)) #trial x time array
    singleTrialActivity[:] = np.nan
    
    for i in range(len(PlayStimulusOutcomeIDX)): #for every task trial
                       
        startIDX=int(column(PlayStimulusOutcomeIDX,1)[i]) #set startIDX
        endIDX=int(column(DemonGoCueOnset,1)[i]) #set endIDX

        singleTrialActivity[i,:endIDX-startIDX]=zScoreC[neuronNum][startIDX:endIDX]
    averageTrialActivity[neuronNum]=np.nanmean(singleTrialActivity,1) # design matrix
         
# plt.imshow(averageTrialActivity)
# plt.xlabel('Outcome Trials')
# plt.ylabel('Cells')
#%%
X_design_matrix = averageTrialActivity
X_design_matrix=X_design_matrix.transpose()
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

#%% Build model for 1000 subsampled iterations
subsampled_acc=[]
subsampled_b_coeff_mat=[]
for i in range(100):
    subsampled_maj_class_indices=random.sample(maj_class_indices,len(min_class_indices))
    subsampled_indices=sorted(min_class_indices+subsampled_maj_class_indices)

    Y_labels_subsampled=Y_labels[subsampled_indices]
    X_design_matrix_subsampled=X_design_matrix[subsampled_indices]

    lasso_model = Lasso()
    b_coeff_mat=[]
    y_pred_acc=[]
    numFolds=10
    kf = KFold(n_splits=numFolds)
    
        
    for train, test in kf.split(X_design_matrix_subsampled): #OBSERVED
        X_train, X_test, y_train, y_test = X_design_matrix_subsampled[train], X_design_matrix_subsampled[test], Y_labels_subsampled[train], Y_labels_subsampled[test]
        
        lasso_model.fit(X_train, y_train)
        
        y_pred_acc.append(lasso_model.score(X_test, y_test))
        b_coeff_mat.append(lasso_model.coef_)
        subsampled_acc.append(np.mean(y_pred_acc))
        subsampled_b_coeff_mat=np.mean(b_coeff_mat,0)
        # print(lasso_model.predict(X_test))
        # print(y_test)
        #print(np.mean(b_coeff_mat,0)) #10-Fold Averaged B-Coefficients of Cells
        #print('Test Accuracy:', lasso_model.score(X_test, y_test)) #10-Fold Accuracy Scores for Each Fold
        #print('Overall Model Accuracy:', np.mean(y_pred_acc)) #Averaged Model performance over 10 folds
    
print('The Average Subsampled Model Performance is:', np.mean(subsampled_acc))
#%%
fig1, ax1 = plt.subplots()
ax1.set_title('Distribution of Subsampled Model Performance Scores (1000-Fold)')
ax1.boxplot(np.mean(subsampled_b_coeff_mat,0))
ax1.xticks([])
ax1.ylabel('Model Performance')
#%% plot distribution of B-weights:
plt.figure()
a=np.histogram(np.mean(subsampled_b_coeff_mat,0), bins=200);    
plt.hist(a)
