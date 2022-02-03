
    binnedMat=np.zeros((mat.shape[0],int(mat.shape[1]/window)))
    binIDX=np.arange(0,mat.shape[1],window)
    
    n=1        #for each neuron
    binnedTrace=[0]*int(mat.shape[1]/window) #create temp bin trace
    for i in range(1,len(binIDX)): #for each window
        binnedTrace.append(mat[n][binIDX[i-1]:binIDX[i]].mean())
    print(binnedTrace)
        # binnedMat[n][:]=binnedTrace
    

#%%
apple=binMat(zScoreC,100)

