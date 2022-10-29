# -*- coding: utf-8 -*-
"""
Interactive plotting of calcium imaging data aligned to some task state and 
grouped by task variable combination


Created on Thu Oct 20 23:41:37 2022

@author: Lukas Oesch
"""

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button #Import required widgets
    from matplotlib.widgets import TextBox
    from matplotlib.gridspec import GridSpec #To create a custom grid on the figure
    import sys
    sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa')
    import decoding_utils
    
    #%%--Input check and convenience
    
    if 'miniscope_data' in locals():
        frame_interval = miniscope_data['frame_interval']
        trial_starts = miniscope_data['trial_starts']
    
    #%%------Set up the globals for the updating
    
    #Set variables to global that will be updated
    global current_neuron
    global num_input
    
    current_neuron = 0
    
    #%%------ Set the figure up
    
    fi = plt.figure(figsize=(20,8))
    fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    gs_data = GridSpec(nrows=2, ncols=4, top=0.9, bottom=0.2, left=0.05, right=0.95 )
    gs_ui = GridSpec(nrows=1, ncols=4, top=0.1, bottom=0.05, left=0.75, right=0.95)
      
    data_axes = []
    for n in range(gs_data.nrows): #Iteratively fill the grid with the desired subplots
        for k in range(gs_data.ncols):
            data_axes.append(fi.add_subplot(gs_data[n,k]))  
    
    #Add x-label on bottom panels, y-labels on left panels 
    for n in [0,4]:
        data_axes[n].set_ylabel('Intensity (A.U.)')
    
    for n in range(4,8):
        data_axes[n].set_xlabel('Time from state/event onset (s)')
    
    
    lines_on_plots = [[] for i in range(2*4)] #Create list with entry for every subplot holding a list with all the line objects 
    
    textAx = fi.add_subplot(gs_ui[0,2])
    textAx.xaxis.set_visible(False) #Remove the axes where not needed
    textAx.yaxis.set_visible(False)
    
    update_textAx = fi.add_subplot(gs_ui[0,3])
    update_textAx.xaxis.set_visible(False) #Remove the axes where not needed
    update_textAx.yaxis.set_visible(False)
    
    backwardAx = fi.add_subplot(gs_ui[0,0])
    backwardAx.xaxis.set_visible(False) #Remove the axes where not needed
    backwardAx.yaxis.set_visible(False)
    
    forwardAx = fi.add_subplot(gs_ui[0,1])
    forwardAx.xaxis.set_visible(False) #Remove the axes where not needed
    forwardAx.yaxis.set_visible(False)
    
    #%%-----Define basic functions to retrieve data and plot traces
        
    def get_state_start_signal(signal, state_start_frame, frame_interval, window):
        '''Retrieve calcium siganl for the respective event from the corresponding neuron.
        Note that that the signal as input has the dimension: neurons x frames.'''
    
        interval_frames = round(window/frame_interval)
        aligned_signal = np.zeros([interval_frames, len(state_start_frame)])
        for n in range(len(state_start_frame)):
            if np.isnan(state_start_frame[n]) == 0:
               aligned_signal[:, n] = signal[current_neuron, int(state_start_frame[n] -  np.ceil(interval_frames/2)) : int(state_start_frame[n] + np.ceil(interval_frames/2) +1)]
            else: 
               aligned_signal[:,n] = np.nan
        x_vect = np.linspace(-window/2, window/2 , aligned_signal.shape[0]) #Use linspace here to be sure that the vector sizes match
        
        return aligned_signal, x_vect
    #----------------------------------------------------------------------
    
    def plot_signal(axes_object, line_objects, aligned_signal, x_vect, indices, line_labels):
        '''Plot the signal grouped by the list of indices. Currently supports up to 4 groups.
        '''
        
        #Set the color scheme: dark blue, bright blue, bright red, dark red
        color_specs = ['#062999','#ABBAFF','#FC7659','#b20707']
        
        #Remove all the lines and plots on the axes before plotting new ones, only do this if they exist
        if not line_objects == False: #Only if there is an entry in the list
            for q in line_objects:
                if type(q) == list:
                    q[0].remove() #Stupidly, plt.plot generates a list of line objects
                else:
                    q.remove()
        line_objects = [] #Reset the list to empty before appending anew
                
        for k in range(len(indices)):
            av = np.nanmean(aligned_signal[:,indices[k]], axis=1)
            sem = np.nanstd(aligned_signal[:,indices[k]], axis=1)/np.sqrt(np.sum(np.isnan(aligned_signal[0,indices[k]])==0))
            
            line_objects.append(axes_object.fill_between(x_vect, av - sem, av + sem, color = color_specs[k], alpha = 0.2)) 
            line_objects.append(axes_object.plot(x_vect, av, color = color_specs[k], label = line_labels[k]))
            line_objects.append(axes_object.axvline(x=0, color='k',linestyle='--', linewidth = 0.3))    
        
        return line_objects
    #--------------------------------------------------------------------------
    
    def figure_updater(fig_handle, data_axes, lines_on_plots, signal, trialdata, average_interval, window):
        '''The master function for initializing and updating the plots on the
           the figure. The states that are plotted are set inside this function 
           currently but may be passed in a later version.'''
           
        consider_states = ['DemonInitFixation', 'PlayStimulus', 'DemonWaitForResponse', 'outcome_presentation']
        label_states = ['Start center fixation', 'Stimulus onset', 'Movement onset', 'Outcome']
        
        pattern = np.array([[0,0],[0,1],[1,0],[1,1]]) #Find these cases in the data
        
        #Determine the indices of the desired task variable combinations
        choice_category = np.array([trialdata['response_side'], trialdata['correct_side']]).T
        curr_indices = []
        
        p_cho = decoding_utils.determine_prior_variable(trialdata['response_side'], np.ones(trialdata.shape[0]), 1, mode = 'consecutive')
        p_cat = decoding_utils.determine_prior_variable(trialdata['correct_side'], np.ones(trialdata.shape[0]), 1, mode = 'consecutive')
        prior_cho_cat = np.array([p_cho, p_cat]).T
        prior_indices = []
        
        for k in range(4):
            curr_indices.append(np.where((choice_category == pattern[k,:]).all(1))[0])
            #Using .all(1) finds all that match along the specified dimension,
            #Using np.where one retrieves specifically the indices
            prior_indices.append(np.where((prior_cho_cat == pattern[k,:]).all(1))[0])
          
        #Define the line labels including the number of trials per condition
        curr_labels = [f'Correct left choice, {len(curr_indices[0])} trials', f'Incorrect left choice, {len(curr_indices[1])} trials',
                       f'Incorrect right choice, {len(curr_indices[2])} trials', f'Correct right choice, {len(curr_indices[3])} trials']
        
        prior_labels = [f'Prior correct left choice, {len(prior_indices[0])} trials', f'Prior incorrect left choice, {len(prior_indices[1])} trials',
                       f'Prior incorrect right choice, {len(prior_indices[2])} trials', f'Prior correct right choice, {len(prior_indices[3])} trials'] 
        
        #Do the plotting
        for m in range(4):
            state_start_frame, _ = decoding_utils.find_state_start_frame_imaging(consider_states[m], trialdata, frame_interval, trial_starts)
            aligned_signal, x_vect =  get_state_start_signal(signal, state_start_frame, frame_interval, window)
            
            lines_on_plots[m] = plot_signal(data_axes[m], lines_on_plots[m], aligned_signal, x_vect, curr_indices, curr_labels)
            lines_on_plots[m + 4] = plot_signal(data_axes[m + 4], lines_on_plots[m + 4], aligned_signal, x_vect, prior_indices, prior_labels)
        
        
        #Adjust the display
        #First allow python to find optimal axes limits
        for m in range(len(data_axes)):
            data_axes[m].relim()
            data_axes[m].autoscale()
        
        
        #Find the required y limits and make them the same on all plots
        tmp = np.zeros([20,2]) 
        for m in range(len(data_axes)):
            tmp[m,:] = data_axes[m].get_ylim()
        
        y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
        for m in range(len(data_axes)):
            data_axes[m].set_ylim(y_lims)
        del y_lims
        
        #Add the titles with the included states/events
        for m in range(len(consider_states)):
            data_axes[m].set_title(label_states[m])
        
        #Add the legend on the rightmost plot of each column
        data_axes[3].legend(loc='upper left')
        data_axes[7].legend(loc='upper left')
        
    #%%------Prepare callback functions
    
    def move_forward(val):
        global current_neuron
        global num_input
        current_neuron = current_neuron + 1
        figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_interval, window)
        num_input.set_val(str(current_neuron))
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    
    def move_backward(val):
        global current_neuron
        global num_input
        current_neuron = current_neuron - 1
        figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_interval, window)
        num_input.set_val(str(current_neuron))
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    
    def jump_to(val):
        global current_neuron
        global num_input
        current_neuron = int(num_input.text)
        figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_interval, window)
        num_input.set_val(str(current_neuron))
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    
    #%%-------Introduce the updating
    button_next = Button(forwardAx, 'Next')
    button_next.on_clicked(move_forward)
    
    button_previous = Button(backwardAx, 'Previous')
    button_previous.on_clicked(move_backward)
    
    num_input = TextBox(textAx,"", initial=f'{current_neuron}')
    
    update_text_button = Button(update_textAx, 'Jump to')
    update_text_button.on_clicked(jump_to)
    
    figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_interval, window)
    
