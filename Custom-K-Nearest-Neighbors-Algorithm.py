import numpy as np
from math import sqrt 
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import style
import warnings
style.use('fivethirtyeight')

dataset = {'k': [ [1,2],[2,3],[3,1] ], 'r' : [ [6,5],[7,7],[8,6] ]}
new_features = [5,7]
# [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset] #Show all the features in the dataset !
# plt.scatter(new_features[0],new_features[1])
# plt.show()

def k_nearest_neighbors(data,predict,k=3):
	if len(data) >= k :
		warnings.warn('K is set to value less than total voting groups!')
	return vote_result