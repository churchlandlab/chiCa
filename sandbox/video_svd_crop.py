
import sys #To run from the command line

def video_SVD(file_name, crop_position=None):
   '''Command line function to be run in the wfield environment.
   Computes the SVD from a provided video file, generates a motion energy
   movie and does the SVD on this one as well.

   Parameters
   ---------
   file_name: string, the full name of the file.
   crop_position: tuple or list, the coordinates between which to keep the data,
                  this should contain the following elements:
                  [lower bound on x, upper bound on x, lower bound on y,
                  upper bound on y]
    '''

   from wfield import load_stack, approximate_svd, chunk_indices
   import numpy as np
   import os
   from glob import glob
   from tqdm import tqdm


   print(file_name)

   #Start by obtaining the svd from the full video
   print('Starting video svd...')

   file_dir = os.path.split(file_name)[0]
   print(f'{crop_position}')
   dat = load_stack(file_name)
   if crop_position is not None: #Apply cropping for the SVT
        mask = np.zeros([dat.shape[2], dat.shape[3]],dtype=bool)
        mask[crop_position[2]:crop_position[3], crop_position[0]:crop_position[1]] = True
   else:
        mask = None

   chunkidx = chunk_indices(len(dat),chunksize=256)
   frame_averages = []
   for on,off in tqdm(chunkidx, desc='Computing average.'):
       frame_averages.append(dat[on:off].mean(axis = 0))
   frames_average = np.stack(frame_averages).mean(axis = 0)
   U,SVT = approximate_svd(dat, frames_average,nframes_per_bin=2, mask=mask)

   if not os.path.isdir(os.path.join(file_dir, "..", "analysis")): #If the analysis folder doesn't exist yet
        os.makedirs(os.path.join(file_dir, "..", "analysis"))
   np.save(os.path.join(file_dir, "..", "analysis", os.path.splitext(os.path.split(file_name)[1])[0] + "_SVD.npy"), [U, SVT, crop_position])


if __name__ == "__main__":
    file_name = sys.argv[1] #Collect the input argument
    #crop_position = sys.argv[2]
    video_SVD(file_name, [int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]) #Run the function
