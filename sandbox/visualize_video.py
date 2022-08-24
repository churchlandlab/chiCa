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
    snippets: list, list of movie snippets extracted
    
    Methods
    ------
    load: extract movie snippets from the video file
    play: play back specified snippets
    save: save snippet data (under construction)
    -> type video_snippet.method? for more information.
    
    Usage
    -----
    snippet_collection_single_file = video_snippets('video_file.avi', time_points, video_frame_interval, window = 2)
    snippet_collection_multi_file = video_snippets(['video_one.avi','video_two.avi'], time_points, video_frame_interval, window = 2)
    
    '''
    
    def __init__(self, video_file, time_points, video_frame_interval, window = 2):
        self.video_file = video_file
        self.time_points = time_points
        self.video_frame_interval = video_frame_interval
        self.window = window

    def load(self):
        '''Load specified movie snippets from the movie file and adds them as the
        snippets attribute to the object.
        
        Usage
        -----
        snippet_collection.load()
        
        '''
        import numpy as np
        import cv2
        
        #Determine whether the movie is stored in one or multiple files
        if isinstance(self.video_file, str): #When the video is stored in one single file
        #Open video
            cap = cv2.VideoCapture(self.video_file)
        
            #Determine dimensions for specified loading later
            success, f = cap.read()
            frame = f[:,:,1] #The video is gray-scale
            frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames 
            window_frames = int(round(self.window/self.video_frame_interval)) #Round first to not automatically floor for the int
        
            #Start loading the snippets
            self.snippets = [] #Pre-allocate the snippet data
            for k in range(len(self.time_points)):
                start_frame = self.time_points[k] - round(window_frames/2) # Make sure to keep symmetric window also with uneven frame numbers
                stop_frame = self.time_points[k] + round(window_frames/2)
                if start_frame < 0 or stop_frame > frame_number:
                    self.snippets.append(None)
                    print(f'Time point number {k} cannot be retrieved because it is out of the video bounds with the current window size, inserted None.')
                else:
                    movie = np.zeros([frame.shape[0], frame.shape[1], window_frames + 1], 'uint8') #The extra frame is the actual event that flanked by two equally sized chunks
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);
                    for n in range(window_frames + 1):
                        success, f = cap.read()
                        movie[:,:,n] = f[:,:,1]
                    self.snippets.append(movie)
                    print(f'Loaded snippet number {k}.')
                
        elif isinstance(self.video_file, list):
            #Open video
            cap = cv2.VideoCapture(self.video_file[0])
            #Determine dimensions for specified loading later
            success, f = cap.read()
            frame = f[:,:,1] #The video is gray-scale
            
            #Get the frame number of the first movie to navigate through the other ones 
            frames_per_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames per video file in the list
            
            #Retrieve number frames on last movie (that may contain less frames) and sum all frames
            cap = cv2.VideoCapture(self.video_file[len(self.video_file) -1])
            tmp =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames in last video file
            frame_number = frames_per_file * (len(self.video_file)-1) + tmp #Add the frame number of the last video to the numbers from all the preceeding ones
            
            window_frames = int(round(self.window/self.video_frame_interval)) #Round first to not automatically floor for the int
        
            #Start loading the snippets
            self.snippets = [] #Pre-allocate the snippet data
            for k in range(len(self.time_points)):
                start_frame = self.time_points[k] - round(window_frames/2) # Make sure to keep symmetric window also with uneven frame numbers
                stop_frame = self.time_points[k] + round(window_frames/2)
                
                start_file_idx = int(np.floor(start_frame/frames_per_file)) #Retrieve the index of the movie containing the first frame of the snippet
                stop_file_idx = int(np.floor(stop_frame/frames_per_file)) #Retrieve the index of the movie containing the first frame of the snippet
                if start_frame < 0 or stop_frame > frame_number: #Make sure the specified window covers frames
                    self.snippets.append(None)
                    print(f'Time point number {k} cannot be retrieved because it is out of the video bounds with the current window size, inserted None.')
                else:
                    if start_file_idx == stop_file_idx:
                        movie = np.zeros([frame.shape[0], frame.shape[1], window_frames + 1], 'uint8') #The extra frame is the actual event that flanked by two equally sized chunks
                        start_in_file = start_frame - start_file_idx * frames_per_file #Go to the corresponding frame inside the file, subtracting the frames accounted for by the other files
                        
                        cap = cv2.VideoCapture(self.video_file[start_file_idx]) #Direct the video reader to the correct movie file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_in_file);
                        for n in range(window_frames + 1):
                            success, f = cap.read()
                            movie[:,:,n] = f[:,:,1]
                        self.snippets.append(movie)
                        print(f'Loaded snippet number {k}.')
                    else: #This is when the snippet is contained in two sequential files
                        movie = np.zeros([frame.shape[0], frame.shape[1], window_frames + 1], 'uint8') #The extra frame is the actual event that flanked by two equally sized chunks
                        start_in_file = [start_frame - start_file_idx * frames_per_file, 0]
                        #Go to the corresponding frame inside the file, subtracting the frames accounted for by the previous files, start at the beginning for the second movie
                        stop_after = [frames_per_file - start_in_file[0], window_frames - (frames_per_file - start_in_file[0]) + 1] #Add one to include last frame
                        #Set after how many frames to stop reading from the respective file
                        counter = 0 #Introduce this here to remember the position inside the snippet
                        for m in range(start_file_idx, stop_file_idx + 1):
                            cap = cv2.VideoCapture(self.video_file[m]) #Direct the video reader to the correct movie file, 
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_in_file[m - start_file_idx]); #Set the first frame to read within the respective video
                            for n in range(stop_after[m - start_file_idx]): #Pass the amount of frames to be read
                                success, f = cap.read()
                                movie[:,:,counter] = f[:,:,1]
                                counter = counter + 1
                        self.snippets.append(movie)
                        print(f'Loaded snippet number {k}.')


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

      for k in snippet_indices:
        cv2.setWindowTitle(window_title, f'Snippet {k}') #Change the name of the window
        for n in range(self.snippets[k].shape[2]):
            cv2.imshow(window_title, self.snippets[k][:,:,n]) #Note that the window always retains its original name for plotting but that this name can be changed for display
            cv2.waitKey(round(self.video_frame_interval*1000)) #Requires an ineger amount of ms
    
        time.sleep(separation) #Here the break is in seconds


###############################################################################
