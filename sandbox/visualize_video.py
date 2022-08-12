# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:10:26 2022

@author: Lukas Oesch
"""

class video_snippets:
    '''Create an object for loading, playing and saving video snippets.
    
    Attributes
    ----------
    video_file: string, the video to load the snippets from
    time_points: list, aa set of frame indices associated with events of interest. 
                 When loading the snippets the function will extract a given amount 
                 of frames before and after these time points.
    video_frame_interval: float, the avergae interval between frames in s
    window: int, duration of the snippet (minus one frame) to be extracted, default = 2.
    snippets: list, list of movie snippets extracted
    
    Methods
    ------
    load: extract movie snippets from the video file
    play: play back specified snippets
    save: save snippet data
    -> type video_snippet.method? for more information.
    
    Usage
    -----
    snippet_collection = video_snippets(video_file, time_points, video_frame_interval, window = 2)
    
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
            start_frame = self.time_points[k] - self.window/2
            stop_frame = self.time_points[k] + self.window/2
            if start_frame < 0 or stop_frame > frame_number:
                print(f'Time point number {k} cannot be retrieved because it is out of the video bounds with the current window size.')
            else:
                movie = np.zeros([frame.shape[0], frame.shape[1], window_frames + 1], 'uint8') #The extra frame is the actual event that flanked by two equally sized chunks
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);
                for n in range(window_frames + 1):
                    success, f = cap.read()
                    movie[:,:,n] = f[:,:,1]
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
