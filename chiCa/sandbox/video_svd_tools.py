# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:27:01 2023

@author: Lukas Oesch
"""

import sys #To run from the command line

def compute_video_SVDs(file_name):
   '''Command line function to be run in the wfield environment.
   Computes the SVD from a provided video file, generates a motion energy
   movie and does the SVD on this one as well.'''

   from wfield import load_stack, approximate_svd, chunk_indices
   import numpy as np
   import os
   from glob import glob
   from tqdm import tqdm


   print(file_name)

   #Start by obtaining the svd from the full video
   print('Starting video svd...')

   file_dir = os.path.split(file_name)[0]
   dat = load_stack(file_name)
   chunkidx = chunk_indices(len(dat),chunksize=256)
   frame_averages = []
   for on,off in tqdm(chunkidx, desc='Computing average.'):
       frame_averages.append(dat[on:off].mean(axis = 0))
   frames_average = np.stack(frame_averages).mean(axis = 0)
   U,SVT = approximate_svd(dat, frames_average,nframes_per_bin=2)

   if not os.path.isdir(os.path.join(file_dir, "..", "analysis")): #If the analysis folder doesn't exist yet
        os.makedirs(os.path.join(file_dir, "..", "analysis"))
   np.save(os.path.join(file_dir, "..", "analysis", os.path.splitext(os.path.split(file_name)[1])[0] + "_video_SVD.npy"), [U, SVT])

   #Do the svd on motion energy video
   print('Starting the svd for the motion energy video')
   me_file = glob(os.path.join(file_dir, "..", "analysis", "*motion_energy.avi"))[0]
   dat = load_stack(me_file)
   frames_average = np.zeros([dat.shape[1], dat.shape[2], dat.shape[3]])
   U,SVT = approximate_svd(dat, frames_average,nframes_per_bin=2)
   np.save(os.path.join(file_dir, "..", "analysis", os.path.splitext(os.path.split(file_name)[1])[0] + "_motion_energy_SVD.npy"), [U, SVT])

#--------------------------------------------------------------------------------
#%%

if __name__ == "__main__":
    file_name = sys.argv[1] #Collect the input argument
    compute_video_SVDs(file_name) #Run the function
