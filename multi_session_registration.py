# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:43:55 2021

@author: Anne
"""
import caiman as cm
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from matplotlib import pyplot as plt
import numpy as np
import os #To resolve the windows path
import sys #To apppend the python path for the selection GUI
sys.path.append('C:/Users/Anne/Documents/chiCa') #Include the path to the functions
import load_cnmfe_outputs
#%% Specify the directory to load the different

#To be done smartly..
session_files = ["D:/data/LO012/20210824_112833/miniscope/binned/secondRound.hdf5",
                 "D:/data/LO012/20210825_164317/miniscope/binned/secondRound.hdf5"]
    
A = [None] * len(session_files)
C = [None] * len(session_files)
S = [None] * len(session_files)
neuron_num = [None] * len(session_files)
recording_length = [None] * len(session_files)
movie_file = [None] * len(session_files)
spatial_spMat = [None] * len(session_files)


for k in range(len(session_files)):
    A[k], C[k], S[k], image_dims, frame_rate, neuron_num[k], recording_length[k], movie_file[k], spatial_spMat[k] = load_cnmfe_outputs.load_data(session_files[k])
    

#%% Compute the correlation image if necessary
corr_image_templates = [None] * len(session_files)

for k in range(len(session_files)):
    Yr, dims, T = cm.load_memmap(str(movie_file[k].resolve()), mode='r')
    images = Yr.T.reshape((T,) + dims, order='F')
    
    #For now
    gSig = (3, 3) 
    corr_image_templates[k], pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
    # Compute a filtered correlation image for all the movies here. This may take a long time. Better do the computation
    # from within the analysis pipeline directly and save the file separately

#%% Register all the sessions to each other using the correlation images
    # and perform the cell matching
    spatial_union, assignments, matchings = register_multisession(A=spatial_spMat, dims=tuple(image_dims), templates=corr_image_templates)
    
    