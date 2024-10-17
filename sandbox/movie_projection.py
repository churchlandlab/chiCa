# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:46:41 2022

@author: Lukas Oesch
"""
def movie_average_projection(movie_file, chunk_size = 2000):
    '''Calculates temporal average frame of a large movie by
    splitting it up into chunks'''
    
    import numpy as np
    import cv2
    import os 
    
    # movie_file = 'C:/data/LO037/20221005_162433/chipmunk/LO037_20221005_162433_chipmunk_DemonstratorAudiTask_DemonstratorBottom_00000000.avi'
    # chunk_size = 10000
    cap = cv2.VideoCapture(movie_file)
    
    #Determine dimensions for specified loading later
    success, f = cap.read()
    frame = f[:,:,1] #The video is gray-scale
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames 
    loops = int(np.ceil(frame_number / chunk_size))
    
    start_f = 0
    stop_f = chunk_size
    
    chunk_average = []
    
    for k in range(loops):
        if k*chunk_size + chunk_size > frame_number:
            stop_f = frame_number - (k  * chunk_size)
        
        mov = np.zeros([chunk_size, 1, frame.shape[0], frame.shape[1]],dtype=int)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f);
        for n in range(stop_f):
            success, f = cap.read()
            mov[n,0,:,:,] = f[:,:,1]
        
        chunk_average.append(np.nanmean(mov,axis=0)) 
    
        start_f = start_f + chunk_size
        print(f'Loop {k} done!')
    
    frame_average = np.mean(np.squeeze([chunk_average], axis=0), axis=0)
    
    np.save(os.path.join(os.path.split(movie_file)[0],'average_frames.npy'), frame_average)
    print('Average projection completed')
    
    return frame_average

#----------------------------------------------------------------------------
#%%
def motion_energy_video(movie_file):
    '''Generates a motion energy video from the provided input.
    assumes input video is uint8! Use the ugly while(True) implementation
    here because some of the video headers might be inaccurate
    This is rather inefficient for now...'''
    
    import numpy as np
    import cv2
    import os 
    import time
    
    cap = cv2.VideoCapture(movie_file)
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames 
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    
    output_file = os.path.join(os.path.split(os.path.split(movie_file)[0])[0], 'analysis', os.path.splitext(os.path.split(movie_file)[1])[0] + '_motion_energy.avi')
    wrt = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, frame_dims)
    previous_frame = np.array([])
    
    timer = time.time()
    sporadic_report = 0
    k = 0
    if frame_number > 0:
        for k in range(frame_number):
        # while(True):
            success, f = cap.read()
            
            if not success:
                print(f'Something_happend at frame {k}. Check the movie!')
                break
            
            frame = np.array(f[:,:,1], dtype=float) #The video is gray-scale, convert to float to be able to record negative values
            
            if previous_frame.shape[0] > 0:
                motion_energy = np.abs(frame -previous_frame).astype(np.uint8)
            else:
                motion_energy = np.zeros([frame_dims[1], frame_dims[0]]).astype(np.uint8)
            
            wrt.write(np.stack((motion_energy, motion_energy, motion_energy), axis=2))
            previous_frame = frame
            
            sporadic_report = sporadic_report + 1
            if sporadic_report == 5000:
                print(f'Wrote {k} frames to file.')
                sporadic_report = 0
            k = k+1   
    else:
        
        while(True):
            success, f = cap.read()
            
            if success:
                frame = np.array(f[:,:,1], dtype=float) #The video is gray-scale, convert to float to be able to record negative values
                
                if previous_frame.shape[0] > 0:
                    motion_energy = np.abs(frame -previous_frame).astype(np.uint8)
                else:
                    motion_energy = np.zeros([frame_dims[1], frame_dims[0]]).astype(np.uint8)
                
                wrt.write(np.stack((motion_energy, motion_energy, motion_energy), axis=2))
                previous_frame = frame
                
                sporadic_report = sporadic_report + 1
                if sporadic_report == 5000:
                    print(f'Wrote {k} frames to file.')
                    sporadic_report = 0
                k = k+1
            else:
                print(f'Something_happend at frame {k}. Check the movie!')
                break
        print(f'Wrote {sporadic_report} frames. Please compare this number with the camlog file.')
        
    wrt.release()
    print(f'Completed in {time.time() - timer} seconds.')
    print('--------------------------------------------')
    return