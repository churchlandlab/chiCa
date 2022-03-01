# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:55:11 2022

@author: Lukas Oesch
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider #For interactive control of frame visualization
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.gridspec import GridSpec #To create a custom grid on the figure


#%%------Set up the globals for the updating

global current_neuron
global num_input
global window

global trialdata #The dataframe with the states, etc.
global signal #The array of calcium imaging data, columns are cells and rows are frames
global average_interval
global window

current_neuron = 0


#%%------ Set the figure up

fi = plt.figure(figsize=(18,12))
fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
gs_data = GridSpec(nrows=4, ncols=5, top=0.9, bottom=0.2, left=0.05, right=0.95 )
gs_ui = GridSpec(nrows=2, ncols=3, top=0.1, bottom=0.05, left=0.75, right=0.95)
      
data_axes = []
for n in range(gs_data.nrows): #Iteratively fill the grid with the desired subplots
    for k in range(gs_data.ncols):
          data_axes.append(fi.add_subplot(gs_data[n,k]))  

lines_on_plots = [[] for i in range(4*5))] #Create list with entry for every subplot holding a list with all the line objects 

textAx = fi.add_subplot(gs_ui[0,1])
textAx.xaxis.set_visible(False) #Remove the axes where not needed
textAx.yaxis.set_visible(False)

update_textAx = fi.add_subplot(gs_ui[1,1])
update_textAx.xaxis.set_visible(False) #Remove the axes where not needed
update_textAx.yaxis.set_visible(False)
    
backwardAx = fi.add_subplot(gs_ui[0,0])
backwardAx.xaxis.set_visible(False) #Remove the axes where not needed
backwardAx.yaxis.set_visible(False)

forwardAx = fi.add_subplot(gs_ui[0,2])
forwardAx.xaxis.set_visible(False) #Remove the axes where not needed
forwardAx.yaxis.set_visible(False)

#%%-----Define basic functions to retrieve data and plot traces

def state_time_stamps(state_name, trialdata, average_interval):
    '''Locate the frame during which a certain state in the chipmunk task has
    started. This is a similar functio as the one originally defined in align
    behavior event imaging but does not compute the time covered within the frame.
    This may no be necessary for the context of this visualization GUI.'''
    
    state_start_frame = [None] * len(trialdata) #The frame that covers the start of
    
    for n in range(len(trialdata)): #Subtract one here because the last trial is unfinished
          if np.isnan(trialdata[state_name][n][0]) == 0: #The state has been visited
              try:   
                  frame_time = np.arange(trial_start_time_covered[n]/1000, trialdata['FinishTrial'][n][0] - trialdata['Sync'][n][0], average_interval/1000)
                  #Generate frame times starting the first frame at the end of its coverage of trial inforamtion
              except:
                   frame_time = np.arange(trial_start_time_covered[n]/1000, trialdata['FinishTrial'][n][0] - trialdata['ObsInitFixation'][n][0], average_interval/1000)
                   #This is to fit in the previous implementation of chipmunk
              tmp = frame_time - trialdata[state_name][n][0] #Calculate the time difference
              state_start_frame[n] = int(np.where(tmp > 0)[0][0] + trialdata["trial_start_frame_index"][n])
              #np.where returns a tuple where the first element are the indices that fulfill the condition.
              #Inside the array of indices retrieve the first one that is positive, therefore the first
              #frame that caputres some information.
    
          else:
              state_start_frame[n] = np.nan
          
    return state_start_frame 
    

def get_state_start_signal(signal, state_start_frame, average_interval, window):
    '''Retrieve calcium siganl for the respective event from the corresponding neuron.
       CAUTION: The average interval is in ms here, while the window is defined in 
       seconds. This is messy!'''
   
    interval = np.floor(average_interval)/1000 #Short hand for the average interval between frames, assumes scope acquisition is slower than expected
    aligned_signal = np.zeros([int(window/interval +1), len(state_start_frame)])
    for n in range(len(state_start_frame)):
        if np.isnan(state_start_frame[n]) == 0:
            aligned_signal[:, n] = signal[current_neuron, int(state_start_frame[n] - window/interval /2) : int(state_start_frame[n]  + window/interval /2 + 1)]
        else: 
            aligned_signal[:,n] = np.nan
  
    x_vect = np.arange(-window/2, window/2 + interval, interval) #Also return a vector of timestamps in seconds
    
    return aligned_signal, x_vect
    

def determine_prior_variable(trialdata, tracked_variable):
    '''Get the label of the prior task variable or action, mainly choice or 
       correct side.'''
    
    prior_label = np.zeros([trialdata.shape[0]]) * np.nan
    retain_val = np.nan #Keep the last value in memory while looping
    
    for n in range(trialdata.shape[0]):
        if np.isnan(trialdata[tracked_variable][n]) == 0:
            prior_label[n] = retain_val
            retain_val = trialdata[tracked_variable][n]
    
    return prior_label
       
    
    
def plot_signal(axes_object, line_objects, aligned_signal, x_vect, grouping_variable):
    '''Plot the signal grouped by some binary variable to the respective axes'''
    
    #Calculate the average and the standard error of the mean, so we don't need to plot even more lines
    average_left = np.mean(aligned_signal[:,grouping_variable==0], axis=1)
    sem_left = np.std(aligned_signal[:,grouping_variable==0], axis=1)/np.sqrt(np.sum(grouping_variable==0))
    
    average_right = np.mean(aligned_signal[:,grouping_variable==1], axis=1)
    sem_right = np.std(aligned_signal[:,grouping_variable==1], axis=1)/np.sqrt(np.sum(grouping_variable==1))
    
    #Remove all the lines and plots on the axes before plotting new ones, only do this if they exist
    if not line_objects[0] is None: #Check first element in the list, if the plot is built from scratch this is none
        for q in line_objects:
            if type(q) == list:
                q[0].remove() #Stupidly, plt.plot generates a list of line objects
            else:
                q.remove()
                
    #Plot the data and store the line objects such that they can be removed easily.
    line_objects[0] = axes_object.fill_between(x_vect, average_right-sem_right, average_right+sem_right, color='#FC7659', alpha=0.2) #Left in blue
    line_objects[1] = axes_object.plot(x_vect, average_right, color='#FC7659')[0]
    
    line_objects[2] = axes_object.fill_between(x_vect, average_left-sem_left, average_left+sem_left, color='#ABBAFF', alpha=0.2) #Left in blue
    line_objects[3] = axes_object.plot(x_vect, average_left, color='#ABBAFF')
    
    line_objects[4] = axes_object.axvline(x=0, color='k',linestyle='--', linewidth=0.3)
    
    #return line_objects

#%%-------Define an updater function that refreshes the display of all the subplots 
#         when the current neuron changes

def figure_updater(fig_handle, data_axes, lines_on_plots, signal, trialdata, average_interval, window):
    '''The master function for initializing and updating the plots on the
       the figure. The states that are plotted are set inside this function 
       currently but may be passed in a later version.'''
       
    consider_states = ['DemonInitFixation', 'stimulus_event_timestamps', 'DemonWaitForResponse', 'DemonReward', 'DemonWrongChoice']
       
    for m in range(5):
        state_start_frame = state_time_stamps(consider_states[m], trialdata, average_interval)
        aligned_signal, x_vect =  get_state_start_signal(signal, state_start_frame, average_interval, window)
        plot_signal(data_axes[m], lines_on_plots[m], aligned_signal, x_vect, np.array(trialdata['response_side']))
        #lines_on_plots[m] = plot_signal(data_axes[m], [lines_on_plots[m]], aligned_signal, x_vect, np.array(trialdata['response_side']))
    
    
    fi.canvas.draw_idle()
#%%------Prepare callback functions

def move_forward(val):
    global current_neuron
    global num_input
    current_neuron = current_neuron + 1
    num_input.set_val(str(current_neuron))
    fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)

def move_backward(val):
    global current_neuron
    global num_input
    current_neuron = current_neuron - 1
    num_input.set_val(str(current_neuron))
    fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    
def jump_to(val):
    global current_neuron
    global num_input
    current_neuron = int(num_input.text)
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


