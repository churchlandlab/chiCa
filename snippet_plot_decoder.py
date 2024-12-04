# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 23:54:49 2023

@author: Lukas Oesch
"""

av = []
st = []

for k in output_models:
    av.append(np.mean(k['model_accuracy']))
    st.append(np.std(k['model_accuracy']))
av = np.array(av)
st = np.array(st)   

shuf_av = []
shuf_st = []

for k in output_models:
    shuf_av.append(np.mean(k['shuffle_accuracy']))
    shuf_st.append(np.std(k['shuffle_accuracy']))
   
shuf_av = np.array(shuf_av)
shuf_st = np.array(shuf_st)  
   

plt.figure()
plt.fill_between(np.arange(shuf_av.shape[0]), shuf_av - shuf_st, shuf_av + shuf_st, color='#bcbcbc', alpha = 0.5)
plt.plot(shuf_av, color='#bcbcbc')

plt.fill_between(np.arange(av.shape[0]), av - st, av + st, color='#be3e3e', alpha = 0.5)
plt.plot(av, color='#be3e3e')

