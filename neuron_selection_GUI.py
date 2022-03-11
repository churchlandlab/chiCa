# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:11:34 2021

@author: Lukas Oesch
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider #For interactive control of frame visualization
from matplotlib.gridspec import GridSpec #To create a custom grid on the figure
from tkinter import Tk
from tkinter.filedialog import askopenfile  #This is to select a file from a dialog box

#Uncomment these two lines to test the function and make sure to insert 
#the path to the folder where your functions are located!
#import sys 
#sys.path.insert(0, 'C:/Users/Lukas Oesch/Documents/ChurchlandLab/TestDataChipmunk/TestMiniscopeAlignment/load_cnmfe_outputs')

import load_cnmfe_outputs

#%%
def run_neuron_selection(data_source = None):
    '''run_current_neuron(data_source = None)
    run_current_neuron(data_source = 'C:Users/Documents/analyzedData.hdf5')
    run_current_neuron(data_source = caiman_object)
    This function allows the user to go through individually identified
    putative neurons identified with CNMF-E and select good and discard bad
    ones. Accepted inputs are path string to saved outputs as HDF5 or caiman
    objects directly. When called without arguments the user is able to
    select a saved file. The neurons are presented according to their maximum
    fluorescence intensity value, such that bright components are shown first
    and dimmer "noise" or background components last. Unclassified components
    will be treated as discarded.'''
    
#%%-------Declare global variables and specify the display parameters 
    global neuron_contour
    global current_neuron #The neuron currently on display
    global accepted_components #List of booleans determining whether to keep a cell or not
    global keep_neuron #List of the indices of the cell that will be kept in the end
        
    current_neuron = 0 #Start with the first neuron
    display_window = 30 # in seconds 

#%%----Prepare by loading the data

    if data_source is None: # Let user select the file
        Tk().withdraw() #Don't show the tiny confirmation window
        data_source = askopenfile(title="Select a file", filetypes=[('HDF5 files','*.hdf5')]).name # The file comes as a weird TextIOWrapper
    
    # Load  retrieve the data and load the memory-mapped movie
    A, C, S, F, image_dims, frame_rate, neuron_num, recording_length, movie_file, spatial_spMat = load_cnmfe_outputs.load_data(data_source)
    
    # Load the motion corrected movie (memory mapped)
    Yr = np.memmap(movie_file, mode='r', shape=(image_dims[0] * image_dims[1], recording_length),
                   order='C', dtype=np.float32)
    # IMPORTANT: Pick the C-ordered version of the file and specify the dtype as np.float32 (!!!)
    movie = Yr.T.reshape(recording_length, image_dims[0], image_dims[1], order='F') # Reorder the same way as they do in caiman
    del Yr # No need to keep this one...
        
#%%------Initialize the list of booleans for good neurons and a list with the indices
    
    accepted_components = [None] * neuron_num # Create a list of neurons you want to keep or reject
    keep_neuron = [None] #Initialize the output of indices of the neurons to refit
    
#%%----Sort the cells according to maximum instensity
    
    intensity_maxima = np.max(C,1) #Get maximum along the second dimension, e.g. within each row
    
    idx_max_int = np.argsort(-intensity_maxima) #The negative sign make the sorting go in descending order
    
    #Sort the data accordingly
    C = C[idx_max_int,:]
    S = S[idx_max_int,:]
    A = A[:,:,idx_max_int]
    
#%%--Function to prepare display range for plotted traces and the frame number according to the given time

    def adjust_display_range(display_time, display_window, frame_rate, recording_length):
        display_frame = int(display_time * frame_rate)
        frame_window = display_window * frame_rate
        frame_range = np.array([np.int(display_frame-frame_window/2), np.int(display_frame+frame_window/2)])
        #Handle the exceptions where the display cannot be symmetric, start and end of the trace
        if frame_range[0] < 0:
            frame_range[0] = 0
            frame_range[1] = frame_window
        elif frame_range[1] > recording_length:
            frame_range[0] = recording_length - display_window * frame_rate
            frame_range[1] = recording_length
                
        return display_frame, frame_range
            
#%%--------Prepare the plot

    fi = plt.figure(figsize=(14,9))
    gs = GridSpec(nrows=4, ncols=4, height_ratios=[6,3.5,0.1,0.4], width_ratios=[5,3,0.5,1.5])
    fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
    
    movAx = fi.add_subplot(gs[0,0:1]) 
    movAx.xaxis.set_visible(False) #Remove the axes where not needed
    movAx.yaxis.set_visible(False)
    movAx.set_title("Raw movie", fontsize = 14)
    
    maskAx = fi.add_subplot(gs[0,1:4])
    maskAx.xaxis.set_visible(False) #Remove the axes where not needed
    maskAx.yaxis.set_visible(False)
    maskAx.set_title("Individual neuron denoised", fontsize = 14)
    
    traceAx = fi.add_subplot(gs[1,0:2])
    traceAx.set_xlabel('time (s)', fontsize=12)
    traceAx.set_ylabel('Fluorescence intensity (A.U.)', fontsize=12) 
    traceAx.tick_params(axis='x', labelsize=12)
    traceAx.tick_params(axis='y', labelsize=12)
    
    sliBarAx = fi.add_subplot(gs[3,0:2])
    sliBarAx.xaxis.set_visible(False) #Remove the axes where not needed
    sliBarAx.yaxis.set_visible(False)
    
    interactionAx = fi.add_subplot(gs[1,2:4])
    interactionAx.xaxis.set_visible(False) #Remove the axes where not needed
    interactionAx.yaxis.set_visible(False)
    
#%%----Start plotting

    #First find the time of peak activity
    display_frame = int(np.where(C[0] == C[0].max())[0])
    # Very annoying transformation of formats:
        #First find the occurence of the maximum as a tuple and access the first
        #element of this tuple, which is an array and needs to be turned into an int!
    display_time = display_frame/frame_rate
    
    #Call the function to prepare display range
    display_frame, frame_range = adjust_display_range(display_time, display_window, frame_rate, recording_length)
    
    #First the corrected movie with contour
    movie_frame = movAx.imshow(movie[display_frame,:,:], cmap='gray', vmin=0, vmax=np.max(movie)) #FIxate the display range here
    neuron_mask = A[:,:,current_neuron] > 0 #Theshold to get binary mask
    neuron_contour = movAx.contour(neuron_mask, linewidths=0.5) #Overlay binary mask on movie frame
    
    #Then plot the de-noised cell activity alone
    pixel_intensity_scaling = 1/np.max(A[:,:,current_neuron])
    max_acti = np.max(C[current_neuron]) #Find the maximum of the denoised trace
    mask_image = maskAx.imshow(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]),
                               cmap='gray', vmin=0, vmax=max_acti)
    #Also take the positive value for A to make sure it is bigge

    #Set up the plots for the traces
    time_vect = np.linspace(frame_range[0]/frame_rate, frame_range[1]/frame_rate, np.int(frame_range[1]-frame_range[0])) 
    C_line = traceAx.plot(time_vect, C[current_neuron, frame_range[0]:frame_range[1]], label='Denoised trace')
    S_line = traceAx.plot(time_vect, S[current_neuron, frame_range[0]:frame_range[1]], label='Estimated calcium transients')
    vert_line = traceAx.axvline(display_time, color='red')
    traceAx.grid()
    plt.setp(traceAx, xlim=(frame_range[0]/frame_rate, frame_range[1]/frame_rate))
    plt.setp(traceAx, ylim=(-5, round(np.max(C[current_neuron])+5))) #Scale y axis 
    # traceAx.tick_params(axis='x', labelsize=12)
    # traceAx.tick_params(axis='y', labelsize=12)
    # traceAx.xaxis.label.set_size(12)
    # traceAx.yaxis.label.set_size(12)
    traceAx.legend(prop={'size': 12})      
    
    # Now the text display
    # Static
    interactionAx.text(0.05,0.8,'Accept neuron:', fontsize = 12)
    interactionAx.text(0.05,0.7,'Discard neuron:', fontsize = 12)
    interactionAx.text(0.05,0.6,'Forward:', fontsize = 12)
    interactionAx.text(0.05,0.5,'Backward:', fontsize = 12)
    
    interactionAx.text(0.75,0.8,'c', fontsize = 12, fontweight = 'bold')
    interactionAx.text(0.75,0.7,'x', fontsize = 12, fontweight = 'bold')
    interactionAx.text(0.75,0.6,'>', fontsize = 12, fontweight = 'bold')
    interactionAx.text(0.75,0.5,'<', fontsize = 12, fontweight = 'bold')
    
    show_accepted = interactionAx.text(0.5, 0.2, 'Not decided', fontweight = 'bold', fontsize = 12,
        horizontalalignment = 'center', verticalalignment = 'center',
        bbox ={'facecolor':(1,1,1),'alpha':0.9, 'pad':20})
    
#%%--------Set up the slider 
    frame_slider = Slider(
        ax=sliBarAx,
        label='Time',
        valmin=0,
        valmax=recording_length/frame_rate, 
        valinit=display_time, 
        valstep=1/frame_rate) #Fix the steps to integers

    frame_slider.label.set_size(12)
    frame_slider.vline.set_visible(False)
    
#%%---The slider callback
    # The function to be called anytime a slider's value changes
    def frame_slider_update(val):
        
        display_frame, frame_range = adjust_display_range(val, display_window, frame_rate, recording_length)
        movie_frame.set_data(movie[display_frame,:,:])
        
        mask_image.set_data(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]))
        
        time_vect = np.linspace(frame_range[0]/frame_rate, frame_range[1]/frame_rate, np.int(frame_range[1]-frame_range[0])) 
        C_line[0].set_xdata(time_vect)
        C_line[0].set_ydata(C[current_neuron, frame_range[0]:frame_range[1]])
        #Stupidly the output of a call to plot is not directly a line object but a list of line
        #objects!
    
        S_line[0].set_xdata(time_vect)
        S_line[0].set_ydata(S[current_neuron, frame_range[0]:frame_range[1]])
        
        #Make the x-axis fit 
        plt.setp(traceAx, xlim=(frame_range[0]/frame_rate, frame_range[1]/frame_rate))
    
        vert_line.set_xdata(np.array([val, val]))
        fi.canvas.draw_idle()
    
#%%---Set up the key callback to switch between cells

    #Set the cell number as the parameter to be updated
    def cell_selection_update(event):
        global neuron_contour
        global current_neuron
        global keep_neuron

        # It's necessary here to set these to globals so that they can be redefined within the function
        
        if event.key == 'right' or event.key == 'c' or event.key == 'x':
            if event.key == 'c': #The case where we accept the putative neuron
                accepted_components[current_neuron] = True #This marks an accepted neuron and puts an entry
            elif event.key == 'x':  #The case where we reject the cell
                accepted_components[current_neuron] = False       
                
            if current_neuron < neuron_num:
                    current_neuron = current_neuron+1
                    
        elif event.key == 'left':
            if current_neuron > 0: 
                    current_neuron = current_neuron-1
                            
                            
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
        
        #Find the maximum activation of this neuron on the trace and jump to this
        # position
        display_frame = int(np.where(C[current_neuron] == C[current_neuron].max())[0])
        display_time = display_frame/frame_rate
        display_frame, frame_range = adjust_display_range(display_time, display_window, frame_rate, recording_length)
        
        #Adjust frame slider
        frame_slider.set_val(display_time)
        
        #Jump to the respective movie frame
        #movie_frame.set_data(movie[display_frame,:,:])
        
        #Update the contour on the already displayed frame
        #Need to remove the contours first, unfortunately
        for tp in neuron_contour.collections: 
            tp.remove()
                
        neuron_mask = A[:,:,current_neuron] > 0
        neuron_contour = movAx.contour(neuron_mask, linewidths=0.5)

        #Update the denoised plot
        pixel_intensity_scaling = 1/np.max(A[:,:,current_neuron])
        max_acti = np.max(C[current_neuron]) #Find the maximum of the denoised trace
        mask_image.set_data(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]))
        mask_image.set_clim(vmin=0, vmax=max_acti)
            
        #Set the plot with the traces accordingly
        C_line[0].set_ydata(C[current_neuron, frame_range[0]:frame_range[1]])
        S_line[0].set_ydata(S[current_neuron, frame_range[0]:frame_range[1]])
        plt.setp(traceAx, ylim=(-5, round(np.max(C[current_neuron])+5))) #Scale y axis
        
        #Finally also update the slider value
        
        
        #Display whether neuron is accepted or not
        if accepted_components[current_neuron] is None: #Not yet determined
            show_accepted.set_text('Not decided')
            show_accepted.set_color((0,0,0))
            show_accepted.set_bbox({'facecolor':(1,1,1),'alpha':0.9, 'pad':20})
        elif accepted_components[current_neuron] == True: #When accepted
            show_accepted.set_text('Accepted')
            show_accepted.set_color((1,1,1))
            show_accepted.set_bbox({'facecolor':(0.23, 0, 0.3),'alpha':0.9, 'pad':20})
        elif accepted_components[current_neuron] == False:
            show_accepted.set_text('Discarded')
            show_accepted.set_color((1,1,1))
            show_accepted.set_bbox({'facecolor':(0.15, 0.15, 0.15),'alpha':0.9, 'pad':20})
            
        fi.canvas.draw_idle()
        
#%%-----Action when fiugure is closed 
    def on_figure_closing(event):
         global keep_neuron
        
         index_selection = [i for i, val in enumerate(accepted_components) if val] # Only keep indices of accepted neurons and undecided
         
         original_indices = idx_max_int[index_selection] #Map the sorted data back to the original indices
         original_indices = np.sort(original_indices)
         keep_neuron[0:len(original_indices)] = list(original_indices) #Transform back to list
        
         print(f"Selection completed with {len(keep_neuron)} accepted neurons")
         print('-------------------------------------------------------------')
         
#%%-----Implement the callbacks

    # register the update function with each slider
    frame_slider.on_changed(frame_slider_update)

    # register the key presses
    fi.canvas.mpl_connect('key_press_event', cell_selection_update)
    
    # Detect the closing of the figure
    fi.canvas.mpl_connect('close_event', on_figure_closing)
    
    return keep_neuron, frame_slider
    #For mysterious reasons the slider has to be returned in order for it to work inside the function!
        
#%%-----Run from within to test
#keep_neuron, frame_slider = run_neuron_selection()