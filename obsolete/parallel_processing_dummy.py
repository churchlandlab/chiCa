# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:32:32 2022

@author: Lukas Oesch
"""


def dummy_parallel(data, cols):
    import numpy as np
    col_sums = np.zeros([1,cols])
    for k in range(cols):
        col_sums[0,k] = np.sum(data[:,k])
    return col_sums