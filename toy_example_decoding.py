# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:27:58 2022

@author: Lukas Oesch
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from find_behavior_event_imaging import find_state_start_frame_imaging

#%%
#Generate two slightly overlapping clusters
labels = np.zeros([1000])
labels[500:1000] = 1
data = np.ones([1000,2])
for k in range(data.shape[0]):
    data[k,0] = 0.75 + ((0.5-np.random.rand())/1.6)
    data[k,1] = 0.25 + ((0.5-np.random.rand())/1.6)
data[500:1000,:] = np.transpose(np.vstack((data[500:1000,1],data[500:1000,0])))

plt.figure()
plt.scatter(data[0:500,0],data[0:500,1], c='#FC7659')
plt.scatter(data[500:1000,0],data[500:1000,1], c='#ABBAFF')

#%%
skf = StratifiedKFold(n_splits=10) #Use stratified cross-validation to make sure 
#that the folds are balanced themselves and that the training and validation is 
#stable.
skf.get_n_splits(data, labels)

c = 0.1 #Regularization parameter

def train_validate_logReg(skf, c, data, labels):
    for train_index, test_index in skf.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = LogisticRegression(penalty='l1', C=c, solver='liblinear').fit(X_train,y_train)
        print('Training accuracy:', clf.score(X_test, y_test))




#%%

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)

# Create a scaler object
sc = StandardScaler()
# Fit the scaler to the training data and transform
X_train_std = sc.fit_transform(X_train)
# Apply the scaler to the test data
X_test_std = sc.transform(X_test)

#%%

C = [10, 1, .1, .001]

for c in C:
    clf = LogisticRegression(penalty='l1', C=c, solver='liblinear')
    clf.fit(X_train, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train_std, y_train))
    print('Test accuracy:', clf.score(X_test_std, y_test))
    print('')


#%% Define the function to parallelize

#----------The function you want to run with
#multiprocess has to be defined outside of the
#script of function you want to parallel-process
#and it has to be imported.

# def dummy_parallel(data, cols):
#     col_sums = np.zeros([1,cols])
#     for k in range(cols):
#         col_sums[0,k] = np.sum(data[:,k])
#     return col_sums
#%%-------Generate a radom dataset and run the function
from parallel_processing_dummy import dummy_parallel


data_3d = [np.random.rand(1000,100) for i in range(50)]
par_pool = mp.Pool(mp.cpu_count())
co = par_pool.starmap(dummy_parallel,[(data, cols) for data in data_3d])
par_pool.close()

 
        
