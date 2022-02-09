def binMatrix(mat, binSize):
    import numpy as np
    binnedMat=np.zeros((mat.shape[0],int(mat.shape[1]/binSize))) #create empty matrix to hold binned values
    binIDX=range(0,mat.shape[1],binSize) #create indexes of bins
    binIDX=np.append(binIDX,[mat.shape[1]]) 
        
    for n in range(0, mat.shape[0]):      #for each neuron
        binnedTrace=[] #create temp bin trace
        
        for i in range(1,len(binIDX)-1): #for each binSize
            binnedTrace.append(mat[n][binIDX[i]:binIDX[i+1]].mean())
                  
        binnedMat[n]=binnedTrace
    return binnedMat
            
