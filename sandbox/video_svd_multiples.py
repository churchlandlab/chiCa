# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:27:01 2023

@author: Lukas Oesch
"""

import sys #To run from the command line

def compute_video_SVDs(file_names):
   '''Command line function to be run in the wfield environment.
   Computes the SVD from a provided video file(s)'''

   from wfield import load_stack, approximate_svd, chunk_indices
   import numpy as np
   import os
   from glob import glob
   from tqdm import tqdm

   #----
   if isinstance(file_names, list)==False:
       file_names = [file_names]

   for current_file in file_names:
       print(current_file)
       #Start by obtaining the svd from the full video
       print('Starting video svd...')

       file_dir = os.path.split(current_file)[0]
       dat = load_stack(current_file)
       chunkidx = chunk_indices(len(dat),chunksize=256)
       frame_averages = []
       for on,off in tqdm(chunkidx, desc='Computing average.'):
           frame_averages.append(dat[on:off].mean(axis = 0))
       frames_average = np.stack(frame_averages).mean(axis = 0)
       U,SVT = approximate_svd(dat, frames_average,nframes_per_bin=2)
       np.save(os.path.join(file_dir,  os.path.splitext(os.path.split(current_file)[1])[0] + "_video_SVD.npy"), [U, SVT])

#--------------------------------------------------------------------------------
#%%

if __name__ == "__main__":
    file_names = sys.argv[1:] #Collect the input argument
    compute_video_SVDs(file_names) #Run the function
