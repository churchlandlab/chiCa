# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:08:15 2022

@author: Lukas Oesch
"""

co = []
for k in decoder['models']:
    co.append(np.mean(k['model_coefficients'], axis=0))
    
coefficients = np.squeeze(np.array(co))    

r = np.corrcoef(coefficients, None, rowvar=False)
rr =  np.corrcoef(coefficients)

shuffles = 100
zero_lag_corr = np.zeros([coefficients.shape[0]])
zero_lag_corr_shuffled = np.zeros([coefficients.shape[0], shuffles])
for k in range(coefficients.shape[0]):
    zero_lag_corr[k] = np.corrcoef(coefficients[k,:], coefficients1[k,:])[0,1]
    for n in range(shuffles):
        zero_lag_corr_shuffled[k,n] = np.corrcoef(coefficients[k,:], coefficients1[k,np.random.permutation(coefficients1.shape[1])])[0,1]
 

C3 = []
for k in range(valid_trials.shape[0]):
    if np.isnan(state_start_frame[k])==0:
        C3.append(C_interpolated[3, int(state_start_frame[k])-20:int(state_start_frame[k])+21])


C3 = np.squeeze(np.array(C3))

resp = np.array(trialdata['response_side'][valid_trials])
idx = np.argsort(resp)


plt.figure()
plt.matshow(C3[idx,:], aspect='auto')

#%%
decoder = output_data

import matplotlib.pyplot as plt

av_model = []
st_model = []
av_shuffle = []
st_shuffle = []
frame_num = []

for k in range(decoder.shape[0]):
    av_model.append(np.mean(decoder['models'][k]['model_accuracy']))
    st_model.append(np.std(decoder['models'][k]['model_accuracy']))
    av_shuffle.append(np.mean(decoder['models'][k]['shuffle_accuracy']))
    st_shuffle.append(np.std(decoder['models'][k]['shuffle_accuracy']))
    frame_num.append(decoder['frame_from_alignment'][k])
    
av_model = np.squeeze(np.array(av_model))
st_model = np.squeeze(np.array(st_model))
av_shuffle = np.squeeze(np.array(av_shuffle))
st_shuffle = np.squeeze(np.array(st_shuffle[k]))
frame_num = np.squeeze(np.array(frame_num)) * 0.05

shuffle_color = '#bcbcbc'    
data_color = '#ee8314'


fi = plt.figure()
my_ax = fi.add_subplot(111)

my_ax.fill_between(frame_num, av_shuffle - st_shuffle, av_shuffle + st_shuffle, color=shuffle_color, alpha=0.4)
my_ax.fill_between(frame_num, av_model - st_model, av_model + st_model, color=data_color, alpha=0.4)

my_ax.plot(frame_num, av_shuffle, color=shuffle_color, label='Shuffled')
my_ax.plot(frame_num, av_model, color=data_color, label='Data')

my_ax.axvline(x=0, color= 'k', linestyle='--')

my_ax.set_ylim(0.4, 1)
my_ax.set_xlabel('Seconds stim train onset')
my_ax.set_ylabel('Decoding accuracy')
my_ax.set_title('Decoding prior choice balanced for prior stim category - stimulus period')
my_ax.legend(loc='best')




