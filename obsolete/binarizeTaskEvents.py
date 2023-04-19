#binarizes event vectors

def binarizeTaskEvents():
    sideChoice=dataFrame['response_side']
    rightRewardIDX=[]
    leftRewardIDX=[]
    rewardIDX=[]
    for i in range(0,len(DemonRewardOnset)):
        rewardIDX.append(DemonRewardOnset[i][0])
    
    for i in range(0, len(rewardIDX)):
        if sideChoice[rewardIDX[i]] == 0.0:
            leftRewardIDX.append(i)
        if sideChoice[rewardIDX[i]] == 1.0:
            rightRewardIDX.append(i) 
    
    rightRewardVector=[]
    rightRewardVector=[0] * zScoreC.shape[1]
    leftRewardVector=[]
    leftRewardVector=[0] * zScoreC.shape[1]
    
    windowSize=int(5/(average_interval/1000))
      
    for i in rightRewardIDX:
        rightRewardVector[DemonRewardOnset[i][1]:DemonRewardOnset[i][1]+windowSize] = [1] * windowSize
    
    for i in leftRewardIDX:
        leftRewardVector[DemonRewardOnset[i][1]:DemonRewardOnset[i][1]+windowSize] = [1] * windowSize
    
    return rightRewardVector
