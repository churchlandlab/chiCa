# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:08:13 2024

@author: Lukas Oesch
"""
import numpy as np #For general use in all the functions
import matplotlib

def separate_axes(ax):
    '''Separate plot axes at bottom left corner and remove top and right spines.'''
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    yti = ax.get_yticks()
    yti = yti[(yti >= ax.get_ylim()[0]) & (yti <= ax.get_ylim()[1]+10**-3)] #Add a small value to cover for some very tiny added values
    ax.spines['left'].set_bounds([yti[0], yti[-1]])
    xti = ax.get_xticks()
    xti = xti[(xti >= ax.get_xlim()[0]) & (xti <= ax.get_xlim()[1]+10**-3)]
    ax.spines['bottom'].set_bounds([xti[0], xti[-1]])
    return

#----------------------------------------------------------------------------
#%%
def fancy_violin(axis, data, violin_colors=None, labels = None, x_label = None, y_label = None, widths = 0.5):
    '''Draw violin plot with a black boxplot inside and the mean marked with a scatter dot.'''
    
    gray = '#c1cdcd'
    
    if violin_colors is None:
        violin_colors = ['#f2e7c6'] * len(data)
        
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    
    parts = axis.violinplot(
            data, widths = widths, showmeans=False, showmedians=False,
            showextrema=False)
    
    for pc in range(len(parts['bodies'])):
        parts['bodies'][pc].set_facecolor(violin_colors[pc])
        parts['bodies'][pc].set_edgecolor('k')
        parts['bodies'][pc].set_alpha(1)
    
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    
    inds = np.arange(1, len(medians) + 1)
    axis.scatter(inds, medians, marker='o', color='white', edgecolors = gray, s=30, zorder=3)
    axis.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    axis.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if labels is not None:
        axis.set_xticks(np.arange(1, len(labels) + 1))
        axis.set_xticklabels(labels)
        axis.set_xlim(0.25, len(labels) + 0.75)
    return

#-----------------------------------------------------------------------------
#%%
def fancy_boxplot(axis, data, box_colors, labels = None, x_label = None, y_label = None, widths = 0.5, positions=None):
    '''Draw boxplots with mean as a scatter point'''
    gray = '#c1cdcd'
    if box_colors is None:
       box_colors = ['#f2e7c6'] * len(data)
    
    medianprops = dict({'color': 'k'})
    meanprops = dict(marker='o', markeredgecolor='black', markerfacecolor='white')
    bp = axis.boxplot(data, positions=positions, labels=labels, widths = widths, showmeans=True, showfliers = False, patch_artist=True, medianprops = medianprops, meanprops = meanprops)

    for k in range(len(box_colors)):
        bp['boxes'][k].set_facecolor(box_colors[k])
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if labels is not None:
        axis.set_xticks(np.arange(1,len(labels)+1))
        axis.set_xticklabels(labels)
        axis.set_xlim(0.25, len(labels) + 0.75)
    return
    
#---------------------------------------------------------
#%%
def box_scatter(axis, box_data, labels = None, bar_colors = None, x_label = None, y_label = None, scatter_size = 20):
    '''Boxplots with overlay of scattered data points'''
    gray = '#c1cdcd'
    box_error = np.std(box_data, axis=0) /  np.sqrt(box_data.shape[0])
    axis.axhline([0], color = 'k', linewidth = ax.spines['top'].get_linewidth())
    axis.bar(np.arange(1,box_data.shape[1]+1), np.mean(box_data, axis=0), color = bar_colors, edgecolor = 'k', linewidth = 1, yerr = box_error, ecolor = 'k', capsize = 3, error_kw = {'linewidth': 1})
    jitter = (np.random.rand(box_data.shape[0]) - 0.5) * 0.5
    for k in range(1, box_data.shape[1]+1):
        axis.scatter(k + jitter, box_data[:,k-1], scatter_size, c = 'w', linewidths = 1, edgecolors = gray, zorder = 2)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if labels is not None:
        axis.set_xticks(np.arange(1, len(labels) + 1))
        axis.set_xticklabels(labels)
        axis.set_xlim(0.25, len(labels) + 0.75)
    return

#------------------------------------------------------------------------
#%%
def fancy_scatter(axis, x_data, y_data, color = None, scatter_size = 20, x_label = None, y_label = None, scatter_labels = None, outline_color = None):
    '''Draw scatter plot with thin gray scatter outline.'''
    
    if outline_color is None:
        outline_color = '#c1cdcd' #Default is gray, this looks really nice on dense scatter plots!
    if isinstance(x_data, list):
        if color is None:
            co_map = matplotlib.cm.get_cmap('Spectral')
            color = [co_map(k/len(x_data)) for k in range(len(x_data))]
        if scatter_labels is None:
            scatter_labels = [None] * len(x_data)
        for k in range(len(x_data)):
            axis.scatter(x_data[k], y_data[k], s = scatter_size, c = color[k], edgecolor = outline_color, linewidth=0.3, label = scatter_labels[k])
    elif isinstance(x_data, np.ndarray):
            axis.scatter(x_data, y_data, s = scatter_size, c = color, edgecolor = outline_color, linewidth=0.3)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
#-----------------------------------------------------------------------------
#%%
def plot_timecourse(ax, data, frame_rate, index_list, spacer = 8, colors = None, linestyles = '-', vline_idx = None, line_labels = None,
                    x_label = None, y_label = None):
    '''Draw timecourse for data wit hmultiple alignment points and hence breaks (of size spacer).'''
    if not isinstance(data, list):
        data = list(data)
        
    if colors is None:
        cmap = matplotlib.cm.get_cmap('Spectral')
        colors = []
        for k in range(len(data)):
            colors.append(cmap(k/(len(data)-1))) #Sample the entire colormap space
    elif isinstance(colors, str):
        colors = [colors]*len(data)
        
    if isinstance(linestyles, str):
        linestyles = [linestyles] * len(data)
        
    if line_labels is None:
        line_labels = [None] * len(data)
        
    for k in range(len(data)):
       for q in range(len(index_list)):
           if q == 0:
               x_vect = (index_list[q] + q*spacer) / frame_rate
           else:
               x_vect = np.arange(x_vect[-1]*frame_rate + spacer, x_vect[-1]*frame_rate + spacer + len(index_list[q])) /frame_rate

           if len(data[k].shape) > 1:
               av = np.mean(data[k][index_list[q],:],axis=1)
               sem = np.std(data[k][index_list[q],:],axis=1)/np.sqrt(data[k].shape[1])
           else:
               av = np.squeeze(data[k][index_list[q]])
               sem = np.zeros([av.shape[0]])*np.nan
               
           ax.fill_between(x_vect, av - sem, av + sem, color = colors[k], alpha = 0.4, edgecolor = None)
           
           if q==0:
               ax.plot(x_vect, av, color = colors[k], label = line_labels[k], linestyle = linestyles[k])
           else:
               ax.plot(x_vect, av, color = colors[k],  linestyle = linestyles[k])
    
    if vline_idx is not None:
        for k in range(len(vline_idx)):
            ax.axvline((vline_idx[k] + k*spacer)/frame_rate, color = 'k', linestyle='--', linewidth = ax.spines['top'].get_linewidth())
    
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
