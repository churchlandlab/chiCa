binnedTimeMat=binMatrix(zScoreC,50).transpose()
#%%
covarMatrix = np.cov(binnedTimeMat)
plt.figure()
plt.imshow(covarMatrix, vmin=0, vmax=1)
plt.colorbar()
