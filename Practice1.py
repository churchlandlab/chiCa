# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:41:41 2022

@author: letiz
"""

import numpy as np
import h5py
import scipy
import scipy.sparse
from pathlib import Path
import matplotlib.pyplot as plt
import statistics

#%%------Define the import function

data_source=  "C:/Users/letiz/Desktop/Test/secondRound.hdf5"
    
#%%---Extract the variables of interest

# Determine the data source - either hdf5 file or caiman object   
isinstance(data_source, str) #Loading the data from HDF5
        
hf = h5py.File(data_source, 'r') #'r' for reading ability

# Extract the noisy, extracted and deconvolved calcium signals
# Use the same variable naming scheme as inside caiman

params = hf.get('params') 
image_dims = np.array(params['data/dims'])
frame_rate = np.array(params['data/fr'])
    
C = np.array(hf.get('estimates/C'))
S = np.array(hf.get('estimates/S'))
        
# Get the sparse matrix with the shapes of the individual neurons
temp = hf.get('estimates/A')
        
# Reconstruct the sparse matrix from the provided indices and data
spMat = scipy.sparse.csc_matrix((np.array(temp['data']),np.array(temp['indices']),
               np.array(temp['indptr'])), shape=np.array(temp['shape']))
        
        
# Retrieve other useful info from the shape of the signal 
neuron_num = C.shape[0]
recording_length = C.shape[1]


deMat = np.array(spMat.todense()) # fill the holes and transform to numpy array

# Several important things here: Other than in caiman the output is saved as 
# n x neuron_num matrix. Therefore, the neuron dimension will be the third one.
# Also, it is important to set the order of reshaping to 'F' (Fortran).
A = deMat.reshape(image_dims[0], image_dims[1], neuron_num, order='F')



    