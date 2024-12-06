
import sys #To run from the command line

def video_SVD(file_name, is_me, crop_position):
   '''Command line function to be run in the wfield environment.
   Computes the SVD from a provided video file, generates a motion energy
   movie and does the SVD on this one as well.

   Parameters
   ---------
   file_name: string, the full name of the file.
   is_me: bool, whether the input video is a motion energy video or not. If False,
          assumes regular video and performs mean subtraction before running SVD, 
          if True it will not perform mean subtraction.
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
   print(is_me)
   #Start by obtaining the svd from the full video
   if is_me:
       print('Starting SVD on motion energy video...')
   else:
       print('Starting SVD on raw video...')

   file_dir = os.path.split(file_name)[0]
   print(f'{crop_position}')
   dat = load_stack(file_name)
   if crop_position is not None: #Apply cropping for the SVT
        mask = np.zeros([dat.shape[2], dat.shape[3]],dtype=bool)
        mask[crop_position[2]:crop_position[3], crop_position[0]:crop_position[1]] = True
   else:
        mask = None
   
   if is_me:
        frames_average = np.zeros([dat.shape[1], dat.shape[2], dat.shape[3]])
        name_appendix = '_motion_energy_SVD.npy'
   else:
       chunkidx = chunk_indices(len(dat),chunksize=256)
       frame_averages = []
       for on,off in tqdm(chunkidx, desc='Computing average.'):
           frame_averages.append(dat[on:off].mean(axis = 0))
       frames_average = np.stack(frame_averages).mean(axis = 0)
       name_appendix = '_video_SVD.npy'
   U,SVT = approximate_svd(dat, frames_average,nframes_per_bin=2, mask=mask)

   if not os.path.isdir(os.path.join(file_dir, "..", "analysis")): #If the analysis folder doesn't exist yet
        os.makedirs(os.path.join(file_dir, "..", "analysis"))
   np.save(os.path.join(file_dir, "..", "analysis", os.path.splitext(os.path.split(file_name)[1])[0] + name_appendix), [U, SVT, crop_position])
   return

if __name__ == "__main__":
    file_name = sys.argv[1] #Collect the input argument
    is_me = bool(eval(sys.argv[2]))
    if len(sys.argv) > 3:
        crop_position = [int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])]
    else:
        crop_position = None
    video_SVD(file_name, is_me, crop_position) #Run the function
