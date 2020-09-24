import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from RNA_1.BroadLearningSystem import BLS
import pandas as pd

features1 = pd.read_csv('../Dataset/Sample1.csv')
features1.head()
features2 = pd.read_csv('../Dataset/Sample2.csv')
features2.head()
features = pd.concat([features1, features2])
features.head()
# print(features)

features = features.replace('mod', 0)
features = features.replace('unm', 1)
features = features.replace(np.nan, 0, regex=True)

# print(features)
X = features[['q1', 'q2', 'q3', 'q4', 'q5', 'mis1', 'mis2', 'mis3', 'mis4', 'mis5']].astype(float)
Y = features['sample'].astype(int)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)

N1 = 10 # # of nodes belong to each window
N2 = 10  # # of windows -------Feature mapping layer
N3 = 500  # # of enhancement nodes -----Enhance layer
L = 5  # # of incremental steps
M1 = 50  # # of adding enhance nodes
s = 0.8  # shrink coefficient
C = 2 ** -30  # Regularization coefficient

y_train=y_train.values
y_test = y_test.values
y_train=np.array(y_train)
y_test =np.array(y_test)

X_train = np.array([[1,1,1],
                   [2,2,2],
                   [3,3,3]])
X_test  = np.array([[1],[1],[1]])
y_train=np.array([1,0,1])
y_test=np.array([1])
print('-------------------BLS_BASE---------------------------')
# BLS(np.transpose(X_train), np.transpose(y_train), np.transpose(X_test), np.transpose(y_test), s, C, N1, N2, N3)
BLS(np.transpose(X_train), y_train, np.transpose(X_test), y_test, s, C, N1, N2, N3)
