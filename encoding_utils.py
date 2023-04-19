# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:22:46 2022

@author: Lukas Oesch
"""

def find_best_splits(all_train_splits, all_test_splits, Yt):
    '''Returns the split of the provided inputs that minimizes the
    distance of the means of the training and testing sets to the overall
    mean. This helps to mitigate effects of inadequate sampling inside the 
    folds for cross-validation
    
    Parameters
    ----------
    all_train_splits: list of lists with arrays holding the sample indices for
                      selecting a training dataset. The dimensions are: 
                      number of splits -> folds -> number of indices
                      dataset.
    all_test_splits: same as above for test set.
    Yt: numpy array of shape trials x number of neurons holding the calcium data
        for the timepoint t.
        
    Returns
    -------
    training: list containing the best training splits for each neuron
    testing: same as above for test
    minimum_deviation: the distance of the selected training and testing set
                       from the overall average.
    '''
    
    import numpy as np
    from time import time
    
    training = [] #Initialite the splits for each time point
    testing = []
    minimum_deviation = []
    
    timer = time()
    for n in range(Yt.shape[1]):       
        abs_deviation = []
        for k in range(len(all_train_splits)):
                
                tmp_dev = []
                
                for draw_num in range(len(all_train_splits[k])):
                    average = np.mean(Yt[:,n])
                    y_train, y_test = Yt[all_train_splits[k][draw_num],n], Yt[all_test_splits[k][draw_num],n]
                    mean_train = np.mean(y_train, axis=0)
                    mean_test = np.mean(y_test, axis=0)
                    
                    tmp_dev.append(np.mean([np.abs(average - mean_train), np.abs(average - mean_test)]))
                    
                abs_deviation.append(tmp_dev)
        
       
        split_deviation = np.mean(abs_deviation)
        min_dev = np.min(split_deviation)
        split_idx = np.where(split_deviation == min_dev)[0][0]
        
        training.append(all_train_splits[split_idx])
        testing.append(all_test_splits[split_idx])
        minimum_deviation.append(min_dev)
    
    print(f'Found best splits in {time() - timer} seconds')
    print('---')
    
    training, testing, minimum_deviation

#%%--------Define the function that muliprocessing is going to need to use
# def encoding_model_individual_cell(Xt, Yt, reg, training_t, testing_t, keys_list):
#     '''xxxx'''
    
#     from decoding_utils import train_ridge_model 
    
#     #Only add to the arrays that are different for individual neurons
#     append_keys =[x for x in keys_list if not ((x == 'best_alpha') or (x == 'number_of_samples'))]  
    
#     for n in range(Yt.shape[1]):
#         for  draw_num in range(len(training_t[n])):
#             train_index = training_t[n][draw_num] #Get the training indices for that once cell
#             test_index = testing_t[n][draw_num]
            
#             #Retrieve training and testing data from splits
#             x_train, x_test = Xt[train_index,:][:,reg], Xt[test_index,:][:,reg]
#             y_train, y_test = Yt[train_index,:][:,n], Yt[test_index,:][:,n]
            
#             #For the single variable models the arrays have to be reshaped into 2d
#             if x_train.ndim == 1:
#                 x_train = x_train.reshape(-1,1)
#                 x_test = x_test.reshape(-1,1)
            
#             #Start the model fitting
#             raw_model = train_ridge_model(x_train, y_train, 1, alpha = regularization_strengths, fit_intercept = fit_intercept)      
            
#             #---Assess model performance
#             y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
#             shuffle_y_hat = np.zeros(y_test.shape).T * np.nan #The reconstructed value
            
#             #Reconstruct the signal, q are trials
#             for q in range(y_test.shape[0]):
#                 y_hat[:,q] = np.sum(x_test[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
#                 shuffle_y_hat[:,q] = np.sum(x_test[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
           
#             #Asses the fit for each neuron individually ---- redun
#             rsq = 1 - (np.sum((y_test[:,0] - y_hat[0,:])**2) / (np.sum((y_test[:,0] - np.mean(y_test[:,0]))**2)))
#             shuffle_rsq = 1 - (np.sum((y_test[:,0] - shuffle_y_hat[0,:])**2) / (np.sum((y_test[:,0] - np.mean(y_test[:,0]))**2)))
                  
#             #For validation also reconstruct the training R2 for all the neurons
#             train_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
#             train_shuffle_y_hat = np.zeros(y_train.shape).T * np.nan #The reconstructed value
            
#             for q in range(y_train.shape[0]):
#                 train_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['model_coefficients'][0], axis=1) + raw_model['model_intercept'][0]
#                 train_shuffle_y_hat[:,q] = np.sum(x_train[q,:] * raw_model['shuffle_coefficients'][0], axis=1) + raw_model['shuffle_intercept'][0]
           
#             #Asses the fit for each neuron individually
#             train_rsq = 1 - (np.sum((y_train[:,0] - train_y_hat[0,:])**2) / (np.sum((y_train[:,0] - np.mean(y_train[:,0]))**2)))
#             train_shuffle_rsq = 1 - (np.sum((y_train[:,0] - train_shuffle_y_hat[0,:])**2) / (np.sum((y_train[:,0] - np.mean(y_train[:,0]))**2)))
                  
#             #Fill into data frame
#             raw_model.insert(0, 'neuron_r_squared', [rsq])
#             raw_model.insert(0,'shuffle_neuron_r_squared', [shuffle_rsq])
            
#             raw_model.insert(0, 'train_neuron_r_squared', [train_rsq])
#             raw_model.insert(0,'train_shuffle_neuron_r_squared', [train_shuffle_rsq])
         
#             #concatenate the dataframes to achieve similar structure to other models
#             if draw_num == 0:
#                 model_df = raw_model
#             else: 
#                 model_df = pd.concat([model_df, raw_model], axis=0, ignore_index = True )
            
#         #Now the weirdness: Assing a temporary data frame to append to.
#         if n==0:
#             neuron_df = model_df
#         else:
#             for k in append_keys():
#                 neuron_df[k] = np.stack((neuron_df[k], model_df[k]),axis=1)
    
#     output_dict = dict()
#     for k in keys_list:
#         output_dict[k] = neuron_df[k]
    
#     return output_dict

# #----------------------------------------------------------------------