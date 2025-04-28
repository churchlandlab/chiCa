# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:22:46 2022

@author: Lukas Oesch
"""
#%%----Compund with a set of functions to fit a ridge model using the Karabastos algorithm
def ridge_MML(Y, X, recenter = True, L = None, regress = True):
    """
    This is an implementation of Ridge regression with the Ridge parameter
    lambda determined using the fast algorithm of Karabatsos 2017 (see
    below). I also made some improvements, described below.

    Inputs are Y (the outcome variables) and X (the design matrix, aka the
    regressors). Y may be a matrix. X is a matrix with as many rows as Y, and
    should *not* include a column of ones.

    A separate value of lambda will be found for each column of Y.

    Outputs are the lambdas (the Ridge parameters, one per column of Y); the
    betas (the regression coefficients, again with columns corresponding to
    columns of Y); and a vector of logicals telling you whether fminbnd
    failed to converge for each column of y (this happens frequently).

    If recenter is True (default), the columns of X and Y will be recentered at
    0. betas will be of size:  np.size(X, 1) x np.size(Y, 1)
    To reconstruct the recentered Y, use:
    y_recentered_hat = (X - np.mean(X, 0)) * betas

    If recenter is False, the columns of X and Y will not be recentered. betas
    will be of size:  size(X, 2)+1 x size(Y, 2)
    The first row corresponds to the intercept term.
    To reconstruct Y, you may therefore use either:
    y_hat = [np.ones((np.size(X, 0), 1)), X] * betas
    or
    y_hat = X * betas[1:, :] + betas[0, :]

    If lambdas is supplied, the optimization step is skipped and the betas
    are computed immediately. This obviously speeds things up a lot.


    TECHNICAL DETAILS:

    To allow for un-centered X and Y, it turns out you can simply avoid
    penalizing the intercept when performing the regression. However, no
    matter what it is important to put the columns of X on the same scale
    before performing the regression (though Matlab's ridge.m does not do
    this if you choose not to recenter). This rescaling step is undone in the
    betas that are returned, so the user does not have to worry about it. But
    this step substantially improves reconstruction quality.

    Improvements to the Karabatsos algorithm: as lambda gets large, local
    optima occur frequently. To combat this, I use two strategies. First,
    once we've passed lambda = 25, we stop using the fixed step size of 1/4
    and start using an adaptive step size: 1% of the current lambda. (This
    also speeds up execution time greatly for large lambdas.) Second, we add
    boxcar smoothing to our likelihood values, with a width of 7. We still
    end up hitting local minima, but the lambdas we find are much bigger and
    closer to the global optimum.

    Source: "Marginal maximum likelihood estimation methods for the tuning
    parameters of ridge, power ridge, and generalized ridge regression" by G
    Karabatsos, Communications in Statistics -- Simulation and Computation,
    2017. Page 6.
    http://www.tandfonline.com/doi/pdf/10.1080/03610918.2017.1321119

    Written by Matt Kaufman, 2018. mattkaufman@uchicago.edu
    
    Adapted to Python by Michael Sokoletsky, 2021
    """
    
    ## Optional arguments
    import numpy as np

    if L is None:
        compute_L = True
    else:
        compute_L = False

    ## If design matrix is a DataFrame, convert to a matrix
    X = np.array(X)

    ## Error checking
    if np.size(Y, 0) != np.size(X, 0):
        raise IndexError('Size mismatch')

    ## Ensure Y is zero-mean
    # This is needed to estimate lambdas, but if recenter = 0, the mean will be
    # restored later for the beta estimation
    pY = np.size(Y, 1)

    if compute_L or recenter:
        X[np.isnan(X)] = 0

    ## Optimize lambda
    if compute_L:

        ## SVD the predictors
        U, d, VH = np.linalg.svd(X, full_matrices=False)
        S = np.diag(d)
        V = VH.T.conj()

        ## Find the valid singular values of X, compute d and alpha
        n = np.size(X, 0)  # Observations
        p = np.size(V, 1)  # Predictors

        # Find the number of good singular values. Ensure numerical stability.
        q = np.sum(d.T > abs(np.spacing(U[0,0])) * np.arange(1,p+1))
        d2 = d ** 2

        # Equation 1
        # Eliminated the diag(1 ./ d2) term: it gets cancelled later and only adds
        # numerical instability (since later elements of d may be tiny).
        # alph = V' * X' * Y
        alph = S @ U.T @ Y
        alpha2 = alph ** 2

        ## Compute variance of y
        # In Equation 19, this is shown as y'y

        Y_var = np.sum(Y ** 2, 0)

        ## Compute the lambdasnp.

        L = np.full(pY,np.nan)

        convergence_failures = np.empty(pY, dtype=int)
        
        for i in range(pY):
            
            L[i], flag = ridge_MML_one_Y(q, d2, n, Y_var[i], alpha2[:, i])
            convergence_failures[i] = flag
        
    else:
        p = np.size(X, 1)




    # If requested, perform the actual regression

    if regress:


        betas = np.full((p, pY), np.nan)

        # You would think you could compute X'X more efficiently as VSSV', but
        # this is numerically unstable and can alter results slightly. Oh well.
        # XTX = V * bsxfun(@times, V', d2)

        XTX = X.T @ X

        # Prep penalty matrix
        ep = np.identity(p)


        # Compute X' * Y all at once, again for speed
        XTY = X.T @ Y

        # Compute betas for renormed X
        if hasattr(L, "__len__"):
            for i in range(0, pY):
                betas[:, i] = np.linalg.solve(XTX + L[i] * ep, XTY[:, i])
        else:
            betas = np.linalg.solve(XTX + L * ep, XTY)

        betas[np.isnan(betas)] = 0



    ## Display fminbnd failures
    
    if compute_L and sum(convergence_failures) > 0:
        print(f'fminbnd failed to converge {sum(convergence_failures)}/{pY} times')
    
    if compute_L and regress:
        return L, betas
    if compute_L:
        return L
    return betas
    

def ridge_MML_one_Y(q, d2, n, Y_var, alpha2):
    
    # Compute the lambda for one column of Y
    import numpy as np
    from scipy import optimize
    # Width of smoothing kernel to use when dealing with large lambda
    
    smooth = 7

    # Value of lambda at which to switch from step size 1/4 to step size L/stepDenom.
    # Value of stepSwitch must be >= smooth/4, and stepSwitch/stepDenom should
    # be >= 1/4.
    step_switch = 25
    step_denom = 100
    
    ## Set up smoothing

    # These rolling buffers will hold the last few values, to average for smoothing
    sm_buffer = np.full(smooth, np.nan)
    test_vals_L = np.full(smooth, np.nan)

    # Initialize index of the buffers where we'll write the next value
    sm_buffer_I = 0
                
    
    # Evaluate the log likelihood of the data for increasing values of lambda
    # This is step 1 of the two-step algorithm at the bottom of page 6.
    # Basically, increment until we pass the peak. Here, I've added trying
    # small steps as normal, then switching over to using larger steps and
    # smoothing to combat local minima.
    
    ## Mint the negative log-likelihood function
    NLL_func = mint_NLL_func(q, d2, n, Y_var, alpha2)

    # Loop through first few values of k before you apply smoothing.
    # Step size 1/4, as recommended by Karabatsos

    done = False
    NLL = np.inf
    for k in range(step_switch * 4+1):
        sm_buffer_I = sm_buffer_I % smooth +1
        prev_NLL = NLL

      # Compute negative log likelihood of the data for this value of lambda
        NLL = NLL_func(k / 4)
        
      # Add to smoothing buffer
        sm_buffer[int(sm_buffer_I-1)] = NLL
        test_vals_L[int(sm_buffer_I-1)] = k / 4

      # Check if we've passed the minimum
        if NLL > prev_NLL:
            # Compute limits for L
            min_L = (k - 2) / 4
            max_L = k / 4
            done = True
            break
                        
    # If we haven't already hit the max likelihood, continue increasing lambda,
    # but now apply smoothing to try to reduce the impact of local minima that
    # occur when lambda is large

    # Also increase step size from 1/4 to L/stepDenom, for speed and robustness
    # to local minima
    
    if not done:
        
        L = k / 4
        NLL = np.mean(sm_buffer)
        iteration = 0
        
        while not done:
            L += L / step_denom
            sm_buffer_I = sm_buffer_I % smooth + 1
            prev_NLL = NLL
            iteration += 1
            # Compute negative log likelihood of the data for this value of lambda,
            # overwrite oldest value in the smoothing buffer
            sm_buffer[int(sm_buffer_I-1)] = NLL_func(L)
            test_vals_L[int(sm_buffer_I-1)] = L
            NLL = np.mean(sm_buffer)
            
            # Check if we've passed the minimum or hit NaN NLL (L passed double-precision maximum)
            
            if NLL>prev_NLL:
                # Adjust for smoothing kernel (walk back by half the kernel)
                sm_buffer_I -= (smooth - 1) / 2
                sm_buffer_I += smooth * (sm_buffer_I < 0) # wrap around
                
        
                max_L = test_vals_L[int(sm_buffer_I-1)]
            
                # Walk back by two more steps to find min bound
                sm_buffer_I -= 2
                sm_buffer_I += smooth * (sm_buffer_I < 0) # wrap around
                min_L = test_vals_L[int(sm_buffer_I-1)]

                passed_min = True
                done = True

            elif np.isnan(NLL):

                passed_min = False
                done = True
                
    else:

        passed_min = True

 
    ## Bounded optimization of lambda
    # This is step 2 of the two-step algorithm at the bottom of page 6. Note
    # that Karabatsos made a mistake when describing the indexing relative to
    # k*, which is fixed here (we need to go from k*-2 to k*, not k*-1 to k*+1)

    if passed_min:
        L, _, flag, _ = optimize.fminbound(NLL_func, max(0, min_L), max_L, xtol=1e-04, full_output=1, disp=0)
    else:
        flag = 1 # if the above loop could not find the minimum, return failed-to-converge flag
    
    return L, flag


def  mint_NLL_func(q, d2, n, Y_var, alpha2):
    '''
    Mint an anonymous function with L as the only input parameter, with all
    the other terms determined by the data.
    We've modified the math here to eliminate the d^2 term from both alpha
    (Equation 1, in main function) and here (Equation 19), because they
    cancel out and add numerical instability.
    '''
    import numpy as np
    NLL_func = lambda L: - (q * np.log(L) - np.sum(np.log(L + d2[:q])) \
                - n * np.log(Y_var - np.sum( np.divide(alpha2[:q],(L + d2[:q])))))
    return NLL_func





#%%----
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
#%%---Function to loop through different shuffles of the encoding model

def fit_encoding_model_shuffles(X, Y, shuffle_indices, standardize_variables, training, testing):
    '''Fit linear encoding model using the ridge_MML function. This function also
    takes inputs to shuffle or z-score specified columns of the design matrix.
    
    Parameters
    ----------
    X: array, k x r design matrix where k are the number of observations and r 
       the number of regressors.
    Y: array, k x n response matrix where k are observations and n are neurons.
       Note: Each column of Y will be z-scored to the training data.
    shuffle_indices: array, regressor indices that need to be shuffled.
    standardize_variables: array, regressor indices that need to be z-scored
    training: list of arrays, the indices of observations to include for training
    testing: list of arrays, the indices of observations to include for testing
    
    Returns
    -------
    alphas: array, best ridge penality for each neuron
    betas: array, the regressor weights for each neuron
    r_squared: array, average of the coefficient of determination calculated
               for the model in each fold
    corr: array, average of the squared correlation coefficient between true
          and predicted neural responses for the model in each fold.
    y_test: list of arrays: The true z-scored neural activity for the test sets
            in each fold. Note: if the same training and testing splits are 
            used for multiple rounds of shuffling different regressors this will
            be the same for each round!
    y_hat: list of arrays: The predicted neural activity from the model fits for
                           for every fold
    
    '''
    
    #-------------------------------------------------------------------------
    import numpy as np
    from time import time

    
    
    #Start timing the fitting
    start_fitting = time()
     
    #Shuffle regressors independently for each time point
    x_shuffle = np.array(X)
    if shuffle_indices is not None: #Only do the shuffling when required
        for k in shuffle_indices:
             x_shuffle[:,k] = np.array(x_shuffle[np.random.permutation(x_shuffle.shape[0]), k]) 
    
    #Initialize the outputs
    tmp_betas = []
    tmp_r_squared = []
    tmp_corr = []
    cum_y_test = []
    cum_y_hat = []
    
    for fold in range(len(training)):      
        y_data = Y[training[fold],:]
        y_std = np.std(y_data, axis=0)
        y_mean = np.mean(y_data, axis=0)
        
        y_train = (y_data - y_mean) / y_std
        y_test = (Y[testing[fold],:] - y_mean) / y_std
        
        x_train = np.array(x_shuffle[training[fold],:]) #Let's make sure this is a new independent array
        x_test = np.array(x_shuffle[testing[fold]])
        if standardize_variables.shape[0] > 0: #Use the mean and std of the training set to scale the test set
            x_std_analog = np.std(x_train[:,standardize_variables], axis=0) 
            assert np.sum(np.isnan(x_std_analog)) == 0, f"Column(s) {standardize_variables[np.where(np.isnan(x_std_analog))[0]]} of the desing matrix have zero standard deviation."
            x_mean_analog = np.mean(x_train[:,standardize_variables], axis=0)
            #The reason this is named analog here is that, although one can standardize
            #dummy variables and kernel regressors, the interpretation of these 
            #standardized variables becomes difficult and so I chose to only z-score
            #the analog regressors. 
        
            x_train[:,standardize_variables] = (x_train[:,standardize_variables] - x_mean_analog) / x_std_analog
            x_test[:,standardize_variables] = (x_test[:,standardize_variables] - x_mean_analog) / x_std_analog
        
        if fold == 0: #Estiate the regularization strength on the first round
            alphas, t_betas = ridge_MML(y_train, x_train, regress=True)
        else:
            t_betas = ridge_MML(y_train, x_train, L = alphas)
        tmp_betas.append(t_betas)
        
        #Get the predictions from the model
        y_hat = np.matmul(x_test,t_betas) #Order matters here!
       
        #Calculate the coefficient of detemination
        ss_results = np.sum((y_test - y_hat)**2, axis = 0)
        ss_total = np.sum((y_test - np.mean(y_test, axis=0))**2, axis = 0)
        rsq = 1 - (ss_results / ss_total) 
        tmp_r_squared.append(rsq)
        
        #Also compute squared pearon correlation
        t_corr = np.zeros([y_test.shape[1]])*np.nan
        for q in range(y_test.shape[1]):
            t_corr[q] = np.corrcoef(y_test[:,q], y_hat[:,q])[0,1]**2 #It is the squared correlation coefficient here
        tmp_corr.append(t_corr)
        
        cum_y_test.append(y_test)
        cum_y_hat.append(y_hat)
        
    betas = np.mean(tmp_betas, axis=0)
    r_squared = np.mean(tmp_r_squared, axis=0)
    corr = np.mean(tmp_corr, axis=0)
    
    stop_fitting = time()
    print(f'Finished run in {stop_fitting - start_fitting}')
    
    return alphas, betas, r_squared, corr, cum_y_test, cum_y_hat
#------------------------------------------------------------------------------
#%%---------------------------------------------------------------------------

def r_squared_timecourse(y_test, y_hat, testing, trial_duration):
    '''Reconstruct the coefficient of determination for the different trial times
    using the predictions from k folds and the corresponding true activity'''
    
    import numpy as np
    
    #Put all the list elements together as an array
    y_t = y_test[0]
    y_h = y_hat[0]
    test_indices = testing[0]
    for k in range(1,len(testing)):
        y_t = np.vstack((y_t, y_test[k]))
        y_h = np.vstack((y_h, y_hat[k]))
        test_indices = np.hstack((test_indices, testing[k])) #The prefered dimension are columns, so for 0-dimensional arrays append to columns..
    
    #Sort the arrays so that one frame occurs after the other one
    sort_idx = np.argsort(test_indices)
    y_t = np.array(y_t[sort_idx,:])
    y_h = np.array(y_h[sort_idx,:])

    #Initialize array
    r_squared = np.zeros([trial_duration, y_t.shape[1]]) * np.nan
    corr = np.zeros([trial_duration, y_t.shape[1]]) * np.nan
    for k in range(trial_duration):
        time_t = np.arange(k, y_t.shape[0], trial_duration)
        
        #Calculate the coefficient of detemination
        ss_results = np.sum((y_t[time_t,:] - y_h[time_t,:])**2, axis = 0)
        ss_total = np.sum((y_t[time_t,:] - np.mean(y_t[time_t,:], axis=0))**2, axis = 0)
        r_squared[k,:] = 1 - (ss_results / ss_total) 
        
         #Also compute squared pearon correlation
        t_corr = np.zeros([y_t.shape[1]])*np.nan
        for q in range(y_t.shape[1]):
            t_corr[q] = np.corrcoef(y_t[time_t,q], y_h[time_t,q])[0,1]**2 #It is the squared correlation coefficient here
        corr[k,:] = t_corr
        
    return r_squared, corr

#-----------------------------------------------------------------------------
#%%----------------------------------------------------------------------------

def assemble_event_trace(aligned_event_timestamps, total_frame_num):
    '''Create a binary event trace from trial aligned timestamps.
    
    Parameters
    ----------
    aligned_event_timestamps: numpy array, vector of timestamps aligned to miniscope
                              or video data.
    total_frame_num: int, the length of the recording in frame number.'
    
    Returns
    ------
    event_trace: numpy array, event trace.
    
    --------------------------------------------------------------------------
    '''
    
    import numpy as np
    
    e_timestamps = np.array(aligned_event_timestamps[np.isnan(aligned_event_timestamps)==0], dtype=int) #Remove remaining nans in the timestamps
    
    event_trace = np.zeros([total_frame_num], dtype=int)
    event_trace[e_timestamps] = 1
    
    return event_trace

#-------------------------------------------------------------------------------
#%%

def shift_regressor(regressor_trace, min_shift, max_shift):
    '''Generate a matrix of shifted traces from an input vector. The range of 
    shifts is from min_shift to max_shift. The shifts are realized using numpy
    roll introducing circular shifts.
    
    Parameters
    ----------
    regressor_trace: numpy array, a vector whose length is matched to the imaging data
    min_shift: int, lower inclusive bound for shifts
    max_shifts: intm, upper exclusive bound for shifts
    
    Returns
    ------
    shift_mat: numpy arry, array with the concatenated time shifted traces, the
               the dimensions are vector.shape[0] x (max_shifts - min_shifts)
               
    ---------------------------------------------------------------------------
    '''
    
    import numpy as np
    
    shift_mat = []
    for k in range(min_shift, max_shift):
        shift_mat.append(np.roll(regressor_trace, k))
        
    shift_mat = np.array(shift_mat).T

    return shift_mat

#------------------------------------------------------------------------------
#%%

def align_frames_miniscope_video(aligned_to, time_frame, trialdata, valid_trials, miniscope_data, video_alignment = None):
    '''Retrieve synchronized timestamps of imaging and video data aligned to behavioral states'''
    
    import numpy as np
    from chiCa import find_state_start_frame_imaging, match_video_to_imaging
    
    
    tmp_imaging = []
    tmp_video = []
    for k in range(len(aligned_to)):
       
        tmp_im = []
        tmp_vi = []
        state_start_frame, state_time_covered = find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
        zero_frame = np.array(state_start_frame[valid_trials] + time_frame[k][0], dtype=int) #The firts frame to consider
        
        for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
            tmp_im.append(zero_frame + add_to)
            
            matching_frames = []
            if video_alignment is not None:
                for q in range(zero_frame.shape[0]): #unfortunately need to loop through the trials, should be improved in the future...
                      tmp = match_video_to_imaging(np.array([zero_frame[q] + add_to]), miniscope_data['trial_starts'][valid_trials[q]],
                           miniscope_data['frame_interval'], video_alignment['trial_starts'][valid_trials[q]], video_alignment['frame_interval'])[0].astype(int)
                      matching_frames.append(tmp)
            tmp_vi.append(np.array(matching_frames))
            
        tmp_imaging.append(tmp_im)
        tmp_video.append(tmp_vi)
        
    imaging_frames = np.sort(np.vstack(tmp_imaging).flatten()).astype(int)
    video_frames = np.sort(np.vstack(tmp_video).flatten()).astype(int)
    
    return imaging_frames, video_frames

#-------------------------------------------------------------
#%%----------------  
    
def fit_ridge_cv_shuffles(X, Y, alpha_range, alpha_per_target, fit_intercept, shuffle_indices, standardize_variables, training, testing):
    '''Fit linear encoding model using the sklearn's RidgeCV and optimize for 
    a set of provided regularization strengths alpha. This hyperparameter is
    separately optimized in each fold. Standardization is applied on training
    and testing data in each fold with the mean and std found from the training
    set to avoid ddata leakage (alpha optimization is performed on the training
    set only for the same purpose). The function allows the user to pass an
    array of indices of columns of the design matrix to be shuffled.
    
    Parameters
    ----------
    X: array, k x r design matrix where k are the number of observations and r 
       the number of regressors.
    Y: array, k x n response matrix where k are observations and n are neurons.
       Note: Each column of Y will be z-scored to the training data.
    alpha_range: list or array, a set of regularization strengths alpha to search
                 through and determine optimal one.
    alpha_per_target: bool, allow a different regularization for each column of y
    fit_intercept: bool, whether to include a global intercept or not
    shuffle_indices: array, regressor indices that need to be shuffled.
    standardize_variables: array, regressor indices that need to be z-scored
    training: list of arrays, the indices of observations to include for training
    testing: list of arrays, the indices of observations to include for testing
    
    Returns
    -------
    alphas: array, best ridge penality for each neuron
    betas: array, the regressor weights for each neuron
    r_squared: array, average of the coefficient of determination calculated
               for the model in each fold
    corr: array, average of the squared correlation coefficient between true
          and predicted neural responses for the model in each fold.
    y_test: list of arrays: The true z-scored neural activity for the test sets
            in each fold. Note: if the same training and testing splits are 
            used for multiple rounds of shuffling different regressors this will
            be the same for each round!
    y_hat: list of arrays: The predicted neural activity from the model fits for
                           for every fold
    
    '''
    
    #-------------------------------------------------------------------------
    import numpy as np
    from time import time
    from sklearn.linear_model import Ridge, RidgeCV
    
    
    #Start timing the fitting
    start_fitting = time()
     
    #Shuffle regressors independently for each time point
    x_shuffle = np.array(X)
    if shuffle_indices is not None: #Only do the shuffling when required
        for k in shuffle_indices:
             x_shuffle[:,k] = np.array(x_shuffle[np.random.permutation(x_shuffle.shape[0]), k]) 
    
    #Initialize the outputs
    tmp_betas = []
    tmp_alphas = []
    tmp_r_squared = []
    tmp_corr = []
    cum_y_test = []
    cum_y_hat = []
    
    # #First, determine alphas on the full dataset
    # y_full = Y - np.mean(Y,axis=0) / np.std(Y, axis=0)
    # y_full[np.isnan(y_full)] = 0
    # y_full[np.isinf(y_full)] = 0
    
    # x_full = np.array(x_shuffle)
    # x_full[:,standardize_variables] = x_full[:,standardize_variables] - np.mean(x_full[:,standardize_variables],axis=0) / np.std(x_full[:,standardize_variables], axis=0)
    # x_full[np.isnan(x_full)] = 0
    # x_full[np.isinf(x_full)] = 0
    
    # ridge_grid_search = RidgeCV(alphas = alpha_range, fit_intercept = fit_intercept, alpha_per_target = alpha_per_target, cv=None, scoring='r2').fit(x_full, y_full)
    # alphas = ridge_grid_search.alpha_
    
    for fold in range(len(training)):      
        y_data = Y[training[fold],:]
        y_std = np.std(y_data, axis=0)
        y_mean = np.mean(y_data, axis=0)
        
        y_train = (y_data - y_mean) / y_std
        y_test = (Y[testing[fold],:] - y_mean) / y_std
        
        #Temporary
        assert np.sum(y_std < 0) == 0, f"Found response variable with zero variance!"
        
        x_train = np.array(x_shuffle[training[fold],:]) #Let's make sure this is a new independent array
        x_test = np.array(x_shuffle[testing[fold]])
        if standardize_variables.shape[0] > 0: #Use the mean and std of the training set to scale the test set
            x_std_analog = np.std(x_train[:,standardize_variables], axis=0) 
            assert np.sum(np.isnan(x_std_analog)) == 0, f"Column(s) {standardize_variables[np.where(np.isnan(x_std_analog))[0]]} of the desing matrix have zero standard deviation."
            x_mean_analog = np.mean(x_train[:,standardize_variables], axis=0)
            #The reason this is named analog here is that, although one can standardize
            #dummy variables and kernel regressors, the interpretation of these 
            #standardized variables becomes difficult and so I chose to only z-score
            #the analog regressors. 
        
            x_train[:,standardize_variables] = (x_train[:,standardize_variables] - x_mean_analog) / x_std_analog
            x_test[:,standardize_variables] = (x_test[:,standardize_variables] - x_mean_analog) / x_std_analog
        
        
        #ridge_model = Ridge(alpha = alphas, fit_intercept = fit_intercept).fit(x_train, y_train)
        ridge_model = RidgeCV(alphas = alpha_range, fit_intercept = fit_intercept, alpha_per_target = alpha_per_target, cv=None, scoring='r2').fit(x_train, y_train)
        tmp_betas.append(ridge_model.coef_)
        tmp_alphas.append(ridge_model.alpha_)
        #Get the predictions from the model
        y_hat = ridge_model.predict(x_test)
       
        #Calculate the coefficient of detemination
        ss_results = np.sum((y_test - y_hat)**2, axis = 0)
        ss_total = np.sum((y_test - np.mean(y_test, axis=0))**2, axis = 0)
        rsq = 1 - (ss_results / ss_total) 
        tmp_r_squared.append(rsq)
        
        #Also compute squared pearon correlation
        t_corr = np.zeros([y_test.shape[1]])*np.nan
        for q in range(y_test.shape[1]):
            t_corr[q] = np.corrcoef(y_test[:,q], y_hat[:,q])[0,1]**2 #It is the squared correlation coefficient here
        tmp_corr.append(t_corr)
        
        cum_y_test.append(y_test)
        cum_y_hat.append(y_hat)
        
    betas = np.mean(tmp_betas, axis=0)
    alphas = 10**(np.mean(np.log10(tmp_alphas))) #Use the average of the exponents for base ten here, because the input grid is on an exponential scale with base 10
    r_squared = np.mean(tmp_r_squared, axis=0)
    corr = np.mean(tmp_corr, axis=0)
    
    stop_fitting = time()
    print(f'Finished run in {stop_fitting - start_fitting}')
    
    return alphas, betas, r_squared, corr, cum_y_test, cum_y_hat
#------------------------------------------------------------------------------