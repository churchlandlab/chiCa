# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:10:26 2022

@author: Lukas Oesch
"""

class video_snippets:
    '''Create an object for loading, playing and saving video snippets.
    
    Attributes
    ----------
    video_file: string or list, the video or list of video files to load the snippets from.
                When a list is provided the order in the list will taken into account. Currently,
                this class only supports extraction of windows that occur within one or across two 
                movie files.
    time_points: list, a set of frame indices associated with events of interest. 
                 When loading the snippets the function will extract a given amount 
                 of frames before and after these time points.
    video_frame_interval: float, the avergae interval between frames in seconds.
    window: int, duration of the snippet (minus one frame) to be extracted, default = 2.
    time_point_at: str, Start at the given timepoints = 'start' or load symmetrically
                   around the timepoints = 'center', default is 'center'
    snippets: list, list of movie snippets extracted
    
    Methods
    ------
    load: extract movie snippets from the video file
    play: play back specified snippets
    save: save snippet data (under construction)
    -> type video_snippet.method? for more information.
    
    Usage
    -----
    snippet_collection_single_file = video_snippets('video_file.avi', time_points, video_frame_interval, window = 2, time_point_at = 'start')
    snippet_collection_multi_file = video_snippets(['video_one.avi','video_two.avi'], time_points, video_frame_interval, window = 2)
    
    '''
    
    def __init__(self, video_file, time_points, video_frame_interval, window = 2, time_point_at = 'center'):
        self.video_file = video_file
        self.time_points = list(time_points)
        self.video_frame_interval = video_frame_interval
        self.window = window
        self.time_point_at = time_point_at

    def load(self):
        '''Load specified movie snippets from the movie file and adds them as the
        snippets attribute to the object.
        
        Usage
        -----
        snippet_collection.load()
        
        '''
        import numpy as np
        import cv2
        #import decord as de
        from time import time
        
        #Determine whether the movie is stored in one or multiple files
        if isinstance(self.video_file, str): #When the video is stored in one single file
        #Open video
        
            # #----------------------------------------------------------------
            # #----Very incorrect decord search
            # window_frames = int(round(self.window/self.video_frame_interval)) #Round first to not automatically floor for the int
            # vr = de.VideoReader(self.video_file) #The video reader
            # frame_number = vr._num_frame #Get the total number of frames
            
            # #Collect all the frame timesatamps
            # timepoint_collection = []
            # for k in range(len(self.time_points)):
            #     if self.time_point_at == 'center': #Pick half the frames before and half the frames after the provided timepoints
            #         start_frame = self.time_points[k] - round(window_frames/2) # Make sure to keep symmetric window also with uneven frame numbers
            #         stop_frame = self.time_points[k] + round(window_frames/2)
            #     elif self.time_point_at == 'start':
            #         start_frame = self.time_points[k] # Start right at the timepoint
            #         stop_frame = self.time_points[k] + window_frames + 1
            #     timepoint_collection.append(np.arange(start_frame,stop_frame))
            # timepoint_collection = np.hstack(timepoint_collection)
            
            # start_t = time()
            # movie = vr.get_batch([timepoint_collection]).asnumpy()
            # self.snippets = np.split(np.transpose(np.squeeze(movie[:,:,:,0]), [1,2,0]), len(self.time_points), axis=2)
            
            # print(f'Loaded snippets in {time() - start_t} seconds')
            # print('----------------------------------------------')
            
           
            #-------------------------------------------------------------------
            #-------Precise but slow OpenCV version
            cap = cv2.VideoCapture(self.video_file)
        
            #Determine dimensions for specified loading later
            success, f = cap.read()
            frame = f[:,:,1] #The video is gray-scale
            frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames 
            window_frames = int(round(self.window/self.video_frame_interval)) #Round first to not automatically floor for the int
            cap.release()
            
            
            #Collect all the frame timesatamps
            timepoint_collection = []
            snippet_end = [] #Ugly but helps keeping data load smaller
            for k in range(len(self.time_points)):
                if self.time_point_at == 'center': #Pick half the frames before and half the frames after the provided timepoints
                    start_frame = self.time_points[k] - round(window_frames/2) # Make sure to keep symmetric window also with uneven frame numbers
                    stop_frame = self.time_points[k] + round(window_frames/2)
                elif self.time_point_at == 'start':
                    start_frame = self.time_points[k] # Start right at the timepoint
                    stop_frame = self.time_points[k] + window_frames + 1
                timepoint_collection.append(np.arange(start_frame,stop_frame))
                snippet_end.append(stop_frame-1)
            timepoint_collection = np.hstack(timepoint_collection)
            # snippet_end = np.squeeze(snippet_end)
            
            cap = cv2.VideoCapture(self.video_file)
            movie = []
            start_t = time()
            self.snippets = [] #Pre-allocate the snippet data
            for k in range(int(timepoint_collection[-1]+1)): #Only need to loop through the last timepoint, given they are ordered
                  success, f = cap.read()
                  if k in timepoint_collection:
                      movie.append(f[:,:,1])
                      if k in snippet_end:
                          self.snippets.append(np.transpose(np.squeeze(movie), [1,2,0]))
                          movie = []
            #movie = np.transpose(np.reshape(np.squeeze(movie), [len(self.time_points), window_frames + 1, frame.shape[0], frame.shape[1]]), [2,3,1,0])
            # movie = np.transpose(np.squeeze(movie), [1,2,0])
            # self.snippets = np.split(movie, len(self.time_points), axis=2)
            print(f'Loaded snippets in {time() - start_t} seconds')
            print('----------------------------------------------')
        
        elif isinstance(self.video_file, list):
            # Open video
            cap = cv2.VideoCapture(self.video_file[0])
            #Determine dimensions for specified loading later
            success, f = cap.read()
            frame = f[:,:,1] #The video is gray-scale
            cap.set(cv2.CAP_PROP_POS_FRAMES,0) #Reset to 0 position
            
            #Get the frame number of the first movie to navigate through the other ones 
            frames_per_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames per video file in the list, except last that might be shorter              
            window_frames = int(round(self.window/self.video_frame_interval)) #Round first to not automatically floor for the int
        
            #Start loading the snippets
            self.snippets = [] #Pre-allocate the snippet data
            current_video = 0
            
            
            start_t = time()
            for k in range(len(self.time_points)):
                movie = np.zeros([frame.shape[0], frame.shape[0], window_frames], dtype='uint8')
                
                #Make movie reader for correct movie file
                expected_file = int(np.floor(self.time_points[k]/frames_per_file)) #Find the movie file we expect the frame in
                if expected_file != current_video: #When the currently open movie is not the right one
                    cap = cv2.VideoCapture(self.video_file[expected_file])
                    current_video = expected_file
                
                #Find the start position within the right movie
                if self.time_point_at == 'start':
                    expected_position = int(self.time_points[k] - (expected_file*frames_per_file))
                elif self.time_point_at == 'center':
                    expected_position = int(self.time_points[k] - (expected_file*frames_per_file)) - round(window_frames/2)
                              
                # #Sanity check
                print(f'Reading frame {expected_position} from file {self.video_file[expected_file]}')
                
                empty_runs = expected_position - int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #See how many frames have to be read blank before storing them
                
                for n in range(empty_runs): #Read through until reaching the correct location
                    success, f = cap.read()
                
                if frames_per_file - expected_position > window_frames:
                    for n in range(window_frames):
                        success, f = cap.read()
                        movie[:,:,n] = f[:,:,1]
                    self.snippets.append(movie)
                    #print(f'Loaded snippet number {k}.')
                elif frames_per_file - expected_position <= window_frames:
                    first_mov = (frames_per_file - expected_position)
                    for n in range(first_mov):
                        success, f = cap.read()
                        movie[:,:,n] = f[:,:,1]
                        
                    cap = cv2.VideoCapture(self.video_file[current_video + 1])
                    current_video = current_video + 1
                    second_mov = window_frames - first_mov
                    for n in range(second_mov):
                        success, f = cap.read()
                        movie[:,:,first_mov + n] = f[:,:,1]
                    self.snippets.append(movie)
                    #print(f'Loaded snippet number {k}.')
            print(f'Loaded snippets in {time() - start_t} seconds')
            print('----------------------------------------------')
                
            #-----------------------------------------------------------------
            #-------Old imprecise search version
            # #Start loading the snippets
            # self.snippets = [] #Pre-allocate the snippet data
            # for k in range(len(self.time_points)):
            #     if self.time_point_at == 'center': #Pick half the frames before and half the frames after the provided timepoints
            #         start_frame = self.time_points[k] - round(window_frames/2) # Make sure to keep symmetric window also with uneven frame numbers
            #         stop_frame = self.time_points[k] + round(window_frames/2)
            #     elif self.time_point_at == 'start':
            #         start_frame = self.time_points[k] # Start right at the timepoint
            #         stop_frame = self.time_points[k] + window_frames + 1
                    
            #     if start_frame < 0 or stop_frame > frame_number:
            #         self.snippets.append(None)
            #         print(f'Time point number {k} cannot be retrieved because it is out of the video bounds with the current window size, inserted None.')
            #     else:
            #         movie = np.zeros([frame.shape[0], frame.shape[1], window_frames + 1], 'uint8') #The extra frame is the actual event that flanked by two equally sized chunks
            #         cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            #         print(f'Set position to: {cap.get(cv2.CAP_PROP_POS_FRAMES)}')
            #         for n in range(window_frames + 1):
            #             success, f = cap.read()
            #             movie[:,:,n] = f[:,:,1]
            #         self.snippets.append(movie)
            #         print(f'Loaded snippet number {k}.')
                
        # elif isinstance(self.video_file, list):
        #     #Open video
        #     cap = cv2.VideoCapture(self.video_file[0])
        #     #Determine dimensions for specified loading later
        #     success, f = cap.read()
        #     frame = f[:,:,1] #The video is gray-scale
            
        #     #Get the frame number of the first movie to navigate through the other ones 
        #     frames_per_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames per video file in the list
            
        #     #Retrieve number frames on last movie (that may contain less frames) and sum all frames
        #     cap = cv2.VideoCapture(self.video_file[len(self.video_file) -1])
        #     tmp =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames in last video file
        #     frame_number = frames_per_file * (len(self.video_file)-1) + tmp #Add the frame number of the last video to the numbers from all the preceeding ones
            
        #     window_frames = int(round(self.window/self.video_frame_interval)) #Round first to not automatically floor for the int
        
        #     #Start loading the snippets
        #     self.snippets = [] #Pre-allocate the snippet data
        #     for k in range(len(self.time_points)):
        #         start_frame = self.time_points[k] - round(window_frames/2) # Make sure to keep symmetric window also with uneven frame numbers
        #         stop_frame = self.time_points[k] + round(window_frames/2)
                
        #         start_file_idx = int(np.floor(start_frame/frames_per_file)) #Retrieve the index of the movie containing the first frame of the snippet
        #         stop_file_idx = int(np.floor(stop_frame/frames_per_file)) #Retrieve the index of the movie containing the first frame of the snippet
        #         if start_frame < 0 or stop_frame > frame_number: #Make sure the specified window covers frames
        #             self.snippets.append(None)
        #             print(f'Time point number {k} cannot be retrieved because it is out of the video bounds with the current window size, inserted None.')
        #         else:
        #             if start_file_idx == stop_file_idx:
        #                 movie = np.zeros([frame.shape[0], frame.shape[1], window_frames + 1], 'uint8') #The extra frame is the actual event that flanked by two equally sized chunks
        #                 start_in_file = start_frame - start_file_idx * frames_per_file #Go to the corresponding frame inside the file, subtracting the frames accounted for by the other files
                        
        #                 cap = cv2.VideoCapture(self.video_file[start_file_idx]) #Direct the video reader to the correct movie file
        #                 cap.set(cv2.CAP_PROP_POS_FRAMES, start_in_file);
        #                 for n in range(window_frames + 1):
        #                     success, f = cap.read()
        #                     movie[:,:,n] = f[:,:,1]
        #                 self.snippets.append(movie)
        #                 print(f'Loaded snippet number {k}.')
        #             else: #This is when the snippet is contained in two sequential files
        #                 movie = np.zeros([frame.shape[0], frame.shape[1], window_frames + 1], 'uint8') #The extra frame is the actual event that flanked by two equally sized chunks
        #                 start_in_file = [start_frame - start_file_idx * frames_per_file, 0]
        #                 #Go to the corresponding frame inside the file, subtracting the frames accounted for by the previous files, start at the beginning for the second movie
        #                 stop_after = [frames_per_file - start_in_file[0], window_frames - (frames_per_file - start_in_file[0]) + 1] #Add one to include last frame
        #                 #Set after how many frames to stop reading from the respective file
        #                 counter = 0 #Introduce this here to remember the position inside the snippet
        #                 for m in range(start_file_idx, stop_file_idx + 1):
        #                     cap = cv2.VideoCapture(self.video_file[m]) #Direct the video reader to the correct movie file, 
        #                     cap.set(cv2.CAP_PROP_POS_FRAMES, start_in_file[m - start_file_idx]); #Set the first frame to read within the respective video
        #                     for n in range(stop_after[m - start_file_idx]): #Pass the amount of frames to be read
        #                         success, f = cap.read()
        #                         movie[:,:,counter] = f[:,:,1]
        #                         counter = counter + 1
        #                 self.snippets.append(movie)
        #                 print(f'Loaded snippet number {k}.')


    def play(self, snippet_indices = [], separation = 0.3):
      '''Play back a selection of movie snippets.
      
      Parameters
      ----------
      snippet_indices: list, specifier of the snippets to be played, corresponding
                       to the respective index of the time_points. If the list is
                       empty all the snippets are played. Default = []
      separation: float, time between snippet replay in seconds. Default = 0.3 s
      
      Usage
      -----
      snippet_collection.play([], 1)
      
      '''
      import numpy as np
      import cv2
      import time
      
      if not snippet_indices:
          snippet_indices = np.arange(len(self.snippets))
          
      window_title = "Snippet"
      cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE) #Generate window 

      #Find the desired coordinates to put the frame number display (top right corner)
      #and assign other text parameters
      x_coord = 0.9 * self.snippets[-1].shape[1]
      y_coord = 0.1 * self.snippets[-1].shape[0]
          
      org = (int(x_coord), int(y_coord))
      font = cv2.FONT_HERSHEY_PLAIN
      font_scale = 1
      t_color = (0,126,205) #This is BGR, stupid!
      
      #Determine subtraction of frame number if necessary
      window_frames = int(round(self.window/self.video_frame_interval))
      
      for k in snippet_indices:
        cv2.setWindowTitle(window_title, f'Snippet {k}') #Change the name of the window
        for n in range(self.snippets[k].shape[2]):
            if self.time_point_at == 'center':
                disp_time = n - round(window_frames/2)
            elif self.time_point_at == 'start':
                disp_time = n
            
            temp = np.stack((self.snippets[k][:,:,n],self.snippets[k][:,:,n],self.snippets[k][:,:,n]),axis=2)
            image = cv2.putText(temp, f'{disp_time}', org, font, font_scale, t_color, 2)
          #  cv2.imshow(window_title, self.snippets[k][:,:,n]) #Note that the window always retains its original name for plotting but that this name can be changed for display
            cv2.imshow(window_title, image) #Note that the window always retains its original name for plotting but that this name can be changed for display
            cv2.waitKey(round(self.video_frame_interval*1000)) #Requires an ineger amount of ms
    
        time.sleep(separation) #Here the break is in seconds


###############################################################################
