# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:17:43 2021

Read saved data from hdf5 file and return some key values

@author: Lukas Oesch
"""


import numpy as np
import h5py
import scipy
import scipy.sparse
from pathlib import Path
import matplotlib.pyplot as plt

#%%------Define the import function

def load_data(data_source):
   
    
#%%---Extract the variables of interest

# Determine the data source - either hdf5 file or caiman object   
    if isinstance(data_source, str): #Loading the data from HDF5
        
        hf = h5py.File(data_source, 'r') #'r' for reading ability

# Extract the noisy, extracted and deconvolved calcium signals
# Use the same variable naming scheme as inside caiman

        params = hf.get('params') 
        image_dims = np.array(params['data/dims'])
        frame_rate = np.array(params['data/fr'])
        movie_file = hf.get('mmap_file')[()] # Use [()] notation to access the value of a dataset in h5
        movie_file = Path(movie_file) #Convert to path
    
        C = np.array(hf.get('estimates/C'))
        S = np.array(hf.get('estimates/S'))
        
# Get the sparse matrix with the shapes of the individual neurons
        temp = hf.get('estimates/A')
        
# Reconstruct the sparse matrix from the provided indices and data
        spMat = scipy.sparse.csc_matrix((np.array(temp['data']),np.array(temp['indices']),
                            np.array(temp['indptr'])), shape=np.array(temp['shape']))
        
    else: #Directly accessing from caiman object
        image_dims = data_source.dims
        frame_rate = data_source.params.data['fr']
        movie_file = data_source.mmap_file
        movie_file = Path(movie_file) #Convert to path
        
        
        C = data_source.estimates.C
        S = data_source.estimates.S
        
        spMat = data_source.estimates.A
        
# Retrieve other useful info from the shape of the signal 
    neuron_num = C.shape[0]
    recording_length = C.shape[1]


    deMat = np.array(spMat.todense()) # fill the holes and transform to numpy array

# Several important things here: Other than in caiman the output is saved as 
# n x neuron_num matrix. Therefore, the neuron dimension will be the third one.
# Also, it is important to set the order of reshaping to 'F' (Fortran).
    A = deMat.reshape(image_dims[0], image_dims[1], neuron_num, order='F')

#%%--- Temporary sanity check
    # plt.figure
    # plt.imshow(A[:,:,2]) # show some neuron shape...

    # plt.figure()
    # plt.plot(C[2]) # ...and the corresponding trace

    # plt.figure()
    # plt.plot(deMat[:,2]) # plot one column of data from the dense matrix to check 
    # # at what interval non-zero elements repeat

    # b = A.sum(2)
    # plt.figure()
    # plt.imshow(b) # Plot the projection of all neurons 

#%%---Define the outputs
    print('------------------------')
    print('Successfully loaded data')
    return A, C, S, image_dims, frame_rate, neuron_num, recording_length, movie_file

    