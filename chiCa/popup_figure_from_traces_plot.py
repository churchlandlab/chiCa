# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:21:40 2023

@author: Lukas Oesch
"""

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
    # import sys
    # sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/chiCa')
    # import decoding_utils
    from chiCa import *
    
    #%%--Input check and convenience
    
    if 'miniscope_data' in locals():
        frame_rate = miniscope_data['frame_rate']
        trial_starts = miniscope_data['trial_starts']
        frame_interval = miniscope_data['frame_interval']
        
        prior_mode = 'consecutive'
        
    if 'include_trials' not in locals():
        include_trials = None
        
    #%%------Set up the globals for the updating
    
    #Set variables to global that will be updated
    global current_neuron
    global num_input
    global contingency_multiplier
    
    current_neuron = 0
    contingency_multiplier = -1
    
    #%%------ Set the figure up
    
    fi = plt.figure(figsize=(20,8))
    fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    gs_data = GridSpec(nrows=2, ncols=4, top=0.9, bottom=0.2, left=0.05, right=0.95 )
    gs_ui = GridSpec(nrows=1, ncols=4, top=0.1, bottom=0.05, left=0.75, right=0.95)
    gs_popup = GridSpec(nrows=1, ncols=1, top=0.1, bottom=0.05, left=0.05, right=0.12)
    gs_stim = GridSpec(nrows=1, ncols=1, top=0.1, bottom=0.05, left=0.15, right=0.22)  
    gs_past_cur = GridSpec(nrows=1, ncols=1, top=0.1, bottom=0.05, left=0.25, right=0.32)  
    
    data_axes = []
    for n in range(gs_data._nrows): #Iteratively fill the grid with the desired subplots
        for k in range(gs_data._ncols):
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
    
    popupAx = fi.add_subplot(gs_popup[0,0])
    popupAx.xaxis.set_visible(False) #Remove the axes where not needed
    popupAx.yaxis.set_visible(False)
    
    stimAx = fi.add_subplot(gs_stim[0,0])
    stimAx.xaxis.set_visible(False)
    stimAx.yaxis.set_visible(False)
    
    past_cur_Ax = fi.add_subplot(gs_past_cur[0,0])
    past_cur_Ax.xaxis.set_visible(False)
    past_cur_Ax.yaxis.set_visible(False)
    
    #%%-----Define basic functions to retrieve data and plot traces
        
    def get_state_start_signal(signal, state_start_frame, frame_rate, window):
        '''Retrieve calcium siganl for the respective event from the corresponding neuron.
        Note that that the signal as input has the dimension: neurons x frames.'''
    
        interval_frames = round(window * frame_rate)
        aligned_signal = np.zeros([interval_frames, len(state_start_frame)])
        for n in range(len(state_start_frame)):
            if np.isnan(state_start_frame[n]) == 0:
               aligned_signal[:, n] = signal[current_neuron, int(state_start_frame[n] -  np.floor(interval_frames/2)) : int(state_start_frame[n] + np.ceil(interval_frames/2))]
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
           
        consider_states = ['DemonInitFixation', 'DemonWaitForResponse', 'outcome_presentation', 'outcome_end']
        label_states = ['Start center fixation', 'Movement onset', 'Outcome', 'End of outcome']
        
        pattern = np.array([[0,0],[0,1],[1,0],[1,1]]) #Find these cases in the data
        
        #Determine the indices of the desired task variable combinations
        choice_category = np.array([trialdata['response_side'], trialdata['correct_side']]).T
        curr_indices = []
        
        p_cho = determine_prior_variable(trialdata['response_side'], np.ones(trialdata.shape[0]), 1, mode = prior_mode)
        p_cat = determine_prior_variable(trialdata['correct_side'], np.ones(trialdata.shape[0]), 1, mode = prior_mode)
        prior_cho_cat = np.array([p_cho, p_cat]).T
        prior_indices = []
        
        if include_trials is not None:
            tmp = [x for x in np.arange(choice_category.shape[0]) if x not in include_trials] #All the trials that are not desired
            choice_category[tmp,:] = np.array([np.nan, np.nan]) #Set these trials to nan
            prior_cho_cat[tmp,:] = np.array([np.nan, np.nan])
        
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
            aligned_signal, x_vect =  get_state_start_signal(signal, state_start_frame, frame_rate, window)
            
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
        figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_rate, window)
        num_input.set_val(str(current_neuron))
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    
    def move_backward(val):
        global current_neuron
        global num_input
        current_neuron = current_neuron - 1
        figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_rate, window)
        num_input.set_val(str(current_neuron))
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    
    def jump_to(val):
        global current_neuron
        global num_input
        current_neuron = int(num_input.text)
        figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_rate, window)
        num_input.set_val(str(current_neuron))
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
        
    #%%--Define functions to plot out new figures with specific splits    
    def popup(val):
        global current_neuron
        
        #Here, always alagined to stimulus on for now...
        state_start_frame, _ = decoding_utils.find_state_start_frame_imaging('PlayStimulus', trialdata, frame_interval, trial_starts)
        aligned_signal, x_vect =  get_state_start_signal(signal, state_start_frame, frame_rate, window)
        
        color_specs = ['#062999','#ABBAFF','#FC7659','#b20707']
        subplot_titles = ['Prior correct left', 'Prior incorrect left', 'Prior incorrect right', 'Prior correct right']
        line_labels = ['Correct left', 'Incorrect left', 'Incorrect right', 'Correct right']
        x_label_string = 'Time from stim on (s)'
        
        popup_fig = plt.figure(figsize=[15,9])
        pppAx = [None] * 4
        
        
        pattern = np.array([[0,0],[0,1],[1,0],[1,1]]) #Find these cases in the data
        choice_category = np.array([trialdata['response_side'], trialdata['correct_side']]).T
        
        p_cho = decoding_utils.determine_prior_variable(trialdata['response_side'], np.ones(trialdata.shape[0]), 1, mode = prior_mode)
        p_cat = decoding_utils.determine_prior_variable(trialdata['correct_side'], np.ones(trialdata.shape[0]), 1, mode = prior_mode)
        prior_cho_cat = np.array([p_cho, p_cat]).T
        
        #-----------------------------------------------------------------
        # curr_indices_by_prior = []
        # prior_indices = []
        # for k in range(4): #First find the current chocies
        #     curr_idx = []
        #     prior_indices.append(np.where((prior_cho_cat == pattern[k,:]).all(1))[0])
        #     for n in range(4): #Now spot what the previous choice was
        #         curr_idx.append(prior_indices[k][np.where((choice_category[prior_indices[k],:] == pattern[n,:]).all(1))[0]])
        #     curr_indices_by_prior.append(curr_idx)
        
        
        # #Now the entire plotting business  
        # for k in range(pattern.shape[0]):
        #     pppAx[k] = popup_fig.add_subplot(2,2,k+1)
        #     for n in range(len(curr_indices_by_prior[k])):
        #             tmp_mean = np.mean(aligned_signal[:,curr_indices_by_prior[k][n]], axis=1)
        #             tmp_std = np.std(aligned_signal[:,curr_indices_by_prior[k][n]], axis=1)/np.sqrt(len(curr_indices_by_prior[k][n]))
                    
        #             pppAx[k].fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=color_specs[n], alpha=0.2) 
        #             pppAx[k].plot(x_vect, tmp_mean, color=color_specs[n], label= line_labels[n] + f', {len(curr_indices_by_prior[k][n])} trials')  #Right in red
        #             pppAx[k].axvline(0, color='k', linestyle='--')
        #---------------------------------------------------------
        line_labels = ['Prior correct left', 'Prior incorrect left', 'Prior incorrect right', 'Prior correct right']
        subplot_titles = ['Correct left', 'Incorrect left', 'Incorrect right', 'Correct right']
        
        prior_indices_by_curr = []
        curr_indices = []
        for k in range(4): #First find the current chocies
            prior_idx = []
            curr_indices.append(np.where((choice_category == pattern[k,:]).all(1))[0])
            for n in range(4): #Now spot what the previous choice was
                prior_idx.append(curr_indices[k][np.where((prior_cho_cat[curr_indices[k],:] == pattern[n,:]).all(1))[0]])
            prior_indices_by_curr.append(prior_idx)
        
        
        #Now the entire plotting business  
        for k in range(pattern.shape[0]):
            pppAx[k] = popup_fig.add_subplot(2,2,k+1)
            for n in range(len(prior_indices_by_curr[k])):
                    tmp_mean = np.mean(aligned_signal[:,prior_indices_by_curr[k][n]], axis=1)
                    tmp_std = np.std(aligned_signal[:,prior_indices_by_curr[k][n]], axis=1)/np.sqrt(len(prior_indices_by_curr[k][n]))
                    
                    pppAx[k].fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=color_specs[n], alpha=0.2) 
                    pppAx[k].plot(x_vect, tmp_mean, color=color_specs[n], label= line_labels[n] + f', {len(prior_indices_by_curr[k][n])} trials')  #Right in red
                    pppAx[k].axvline(0, color='k', linestyle='--')
                    
            pppAx[k].set_title(subplot_titles[k])
            pppAx[k].legend(loc='best')
            if (k == 0) or (k == 2):    
                pppAx[k].set_ylabel('Fluorescence intensity (A.U.)')
            if (k == 2) or (k == 3):
                pppAx[k].set_xlabel(x_label_string)      
        
        #Adjust y-axis limtis
        tmp = np.zeros([4,2]) 
        for k in range(len(pppAx)):
         tmp[k,:] = pppAx[k].get_ylim()
        
        y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
        for k in range(len(pppAx)):
            pppAx[k].set_ylim(y_lims)
        del y_lims
        plt.tight_layout()
    
    def past_curr_connection(val):
        global current_neuron
        
        #Here, always alagined to stimulus on for now...
        state_start_frame, _ = decoding_utils.find_state_start_frame_imaging('PlayStimulus', trialdata, frame_interval, trial_starts)
        aligned_signal, x_vect =  get_state_start_signal(signal, state_start_frame, frame_rate, window)
        
        color_specs = ['#3989c6','#d03e3e']
        subplot_titles = ['Prior correct left', 'Prior incorrect left', 'Prior incorrect right', 'Prior correct right']
        line_labels = ['Left choice', 'right choice']
        x_label_string = 'Time from stim on (s)'
        
        past_curr_fig = plt.figure(figsize=[15,9])
        pacAx = [None] * 4
        
        
        pattern = np.array([[0,0],[0,1],[1,0],[1,1]]) #Find these cases in the data
        choice_category = np.array([trialdata['response_side'], trialdata['correct_side']]).T
        
        p_cho = decoding_utils.determine_prior_variable(trialdata['response_side'], np.ones(trialdata.shape[0]), 1, mode = prior_mode)
        p_cat = decoding_utils.determine_prior_variable(trialdata['correct_side'], np.ones(trialdata.shape[0]), 1, mode = prior_mode)
        indices = []
        for k in range(len(pattern)):
            tmp = []
            for ch in range(2):
                tmp.append(np.where(((p_cho==pattern[k][0]) & (p_cat==pattern[k][1])) & (trialdata['response_side']==ch))[0])
            indices.append(tmp)
        # prior_cho_cat = np.array([p_cho, p_cat]).T
        
        # # line_labels = ['Prior correct left', 'Prior incorrect left', 'Prior incorrect right', 'Prior correct right']
        # # subplot_titles = ['Correct left', 'Incorrect left', 'Incorrect right', 'Correct right']
        
        # prior_indices_by_curr = []
        # curr_indices = []
        # for k in range(4): #First find the current chocies
        #     prior_idx = []
        #     curr_indices.append(np.where((choice_category == k).all(1))[0])
        #     for n in range(4): #Now spot what the previous choice was
        #         prior_idx.append(curr_indices[k][np.where((prior_cho_cat[curr_indices[k],:] == pattern[n,:]).all(1))[0]])
        #     prior_indices_by_curr.append(prior_idx)
        
        
        #Now the entire plotting business  
        for k in range(pattern.shape[0]):
            pacAx[k] = past_curr_fig.add_subplot(2,2,k+1)
            for n in range(2):
                    tmp_mean = np.mean(aligned_signal[:,indices[k][n]], axis=1)
                    tmp_std = np.std(aligned_signal[:,indices[k][n]], axis=1)/np.sqrt(len(indices[k][n]))
                    
                    pacAx[k].fill_between(x_vect, tmp_mean-tmp_std, tmp_mean+tmp_std , color=color_specs[n], alpha=0.2) 
                    pacAx[k].plot(x_vect, tmp_mean, color=color_specs[n], label= line_labels[n] + f', {len(indices[k][n])} trials')  #Right in red
                    pacAx[k].axvline(0, color='k', linestyle='--')
                    
            pacAx[k].set_title(subplot_titles[k])
            pacAx[k].legend(loc='best')
            if (k == 0) or (k == 2):    
                pacAx[k].set_ylabel('Fluorescence intensity (A.U.)')
            if (k == 2) or (k == 3):
                pacAx[k].set_xlabel(x_label_string)      
        
        #Adjust y-axis limtis
        tmp = np.zeros([4,2]) 
        for k in range(len(pacAx)):
         tmp[k,:] = pacAx[k].get_ylim()
        
        y_lims = np.array([np.min(tmp[:,0]), np.max(tmp[:,1])])
        for k in range(len(pacAx)):
            pacAx[k].set_ylim(y_lims)
        del y_lims
        plt.tight_layout()
    
     #-----------          
        
    def stim_fig(val):
        global current_neuron
        
        stimulus_figure = plt.figure(figsize=[15,9])
        sAx = stimulus_figure.add_subplot(211)
        title_string = 'Stimulus strength by choice'
        x_label_string = 'Time from stim on (s)'
        
        #Start with finding the stim strengths already
        stim_strength = np.zeros([trialdata.shape[0]]) * np.nan
        for k in range(trialdata.shape[0]):
            stim_strength[k] = trialdata['stimulus_event_timestamps'][k].shape[0]
            
        presented = np.unique(stim_strength)
        
        
        #This is to get the desired color scheme for the plotting
        import seaborn as sns
        l_cmap = sns.color_palette('PiYG', presented.shape[0])
        r_cmap = sns.color_palette('PuOr', presented.shape[0])
        
        #Here, always alagined to stimulus on for now...
        state_start_frame, _ = decoding_utils.find_state_start_frame_imaging('outcome_presentation', trialdata, frame_interval, trial_starts)
        aligned_signal, x_vect =  get_state_start_signal(signal, state_start_frame, frame_rate, window)
        
        for k in range(presented.shape[0]):
            av = np.mean(aligned_signal[:,((stim_strength == presented[k]) & (np.array(trialdata['response_side'])==0))], axis=1)
            sAx.plot(x_vect, av, color = np.array(l_cmap[k])**2, label = f'{presented[k]} Hz, left choice')
        for k in range(presented.shape[0]): 
            av = np.mean(aligned_signal[:,((stim_strength == presented[k]) & (np.array(trialdata['response_side'])==1))], axis=1)
            sAx.plot(x_vect, av, color = np.array(r_cmap[k])**2, label = f'{presented[k]} Hz, right choice')
        
        sAx.set_xlabel(x_label_string)
        sAx.set_title(title_string)
        sAx.legend(loc='lower left', bbox_to_anchor= (0, -1), ncol=2,
            borderaxespad=0, frameon=True)

   
        
    #%%-------Introduce the updating
    button_next = Button(forwardAx, 'Next')
    button_next.on_clicked(move_forward)
    
    button_previous = Button(backwardAx, 'Previous')
    button_previous.on_clicked(move_backward)
    
    num_input = TextBox(textAx,"", initial=f'{current_neuron}')
    
    update_text_button = Button(update_textAx, 'Jump to')
    update_text_button.on_clicked(jump_to)
    
    button_popup = Button(popupAx, 'Split by\npast choice')
    button_popup.on_clicked(popup)
    
    button_stim = Button(stimAx, 'Split by\nstim strength')
    button_stim.on_clicked(stim_fig)
    
    button_past_cur = Button(past_cur_Ax, 'Split by\npast and current')
    button_past_cur.on_clicked(past_curr_connection)
    
    figure_updater(fi, data_axes, lines_on_plots, signal, trialdata, frame_rate, window)
    
