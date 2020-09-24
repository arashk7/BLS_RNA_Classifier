import numpy as np
import scipy.io as scio
from RNA_1.BroadLearningSystem import BLS,BLS_AddEnhanceNodes,BLS_AddFeatureEnhanceNodes,bls_train_input,bls_train_inputenhance

dataFile = './mnist.mat'
data = scio.loadmat(dataFile)
traindata  = np.double(data['train_x']/255)
trainlabel = np.double(data['train_y'])
testdata   = np.double(data['test_x']/255)
testlabel  = np.double(data['test_y'])

N1 = 10  #  # of nodes belong to each window
N2 = 10  #  # of windows -------Feature mapping layer
N3 = 500 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps
M1 = 50  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------BLS_BASE---------------------------')
BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)