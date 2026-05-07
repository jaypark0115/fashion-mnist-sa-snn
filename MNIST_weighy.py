import scipy.ndimage as sp
import numpy as np
import pylab


def randomDelay(minDelay, maxDelay):
    return np.random.rand()*(maxDelay-minDelay) + minDelay
        
        
def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos

        
def sparsenMatrix(baseMatrix, pConn):
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList
        
    
def create_weights():
    
    nInput = 784
    nE = 625
    nI = nE 
    dataPath = './random/'
    weight = {}
    weight['ee_input'] = 0.3 
    weight['ei_input'] = 8.0 
    weight['ee'] = 0.1
    weight['ei'] = 4.0
    weight['ie'] = 4.0
    weight['ii'] = 0.4
    pConn = {}
    pConn['ee_input'] = 1.0 
    pConn['ei_input'] = 0.1 
    pConn['ee'] = 1.0
    pConn['ei'] = 0.0025
    pConn['ie'] = 0.9
    pConn['ii'] = 0.1
    
    
    print ('create random connection matrices')
    connNameList = ['X_to_Sen']
    for name in connNameList:
        weightMatrix = np.random.random((nInput, nE,1)) + 0.01
        weightMatrix *= weight['ee_input']
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            weightList = weightMatrix
        np.save(dataPath+name, weightList)
    
    
    
    print ('create connection matrices from E->I which are purely random')
    connNameList = ['Sen_in_E']
    for name in connNameList:
        weightMatrix = np.ones(nE)
        weightMatrix *= weight['ei_input']
        weightList = weightMatrix
        print ('save connection matrix', name)
        np.save(dataPath+name, weightList) 
        
    
    
    print ('create connection matrices from E->I which are purely random')
    connNameList = ['E_to_I']
    for name in connNameList:
        if nE == nI:
            weightList = np.ones(nE)*weight['ei']
        else:
            weightMatrix = np.random.random((nE, nI))
            weightMatrix *= weight['ei']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        print ('save connection matrix', name)
        np.save(dataPath+name, weightList)
        
        
        
    print ('create connection matrices from I->E which are purely random')
    connNameList = ['I_to_X']
    for name in connNameList:
        if nE == nI:
            weightMatrix = np.ones((nI, nE))
            weightMatrix *= weight['ie']
            for i in range(nI):
                weightMatrix[i,i] = 0
            weightList = weightMatrix
        else:
            weightMatrix = np.random.random((nI, nE))
            weightMatrix *= weight['ie']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        print ('save connection matrix', name)
        np.save(dataPath+name, weightList)
    
         
if __name__ == "__main__":
    create_weights()