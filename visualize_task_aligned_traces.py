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

global traildata #The dataframe with the states, etc.
global signal #The array of calcium imaging data, columns are cells and rows are frames

current_neuron = 0


#%%------ Set the figure up

fi = plt.figure(figsize=(18,12))
fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
gs_data = GridSpec(nrows=4, ncols=5, top=0.9, bottom=0.2, left=0.05, right=0.95 )
gs_ui = GridSpec(nrows=2, ncols=3, top=0.1, bottom=0.05, left=0.75, right=0.95)
      
data_axes = []
for n in range(gs_data.nrows):
    for k in range(gs_data.ncols):
          data_axes.append(fi.add_subplot(gs_data[n,k]))  

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

#%%-----Define the functions to retrieve the relevant data and plot them


def find_state_start_frame_imaging:
    '''Find the timestamps of the onset of the respective task state or event'''
    
    
def get_state_start_signal:
    '''Retrieve calcium siganl for the respective event'''
    
    
def plot_signal:
    '''Plot the signal grouped by some binary variable'''
    
#%%-------Do the initial plotting with the current neuron



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


