import numpy as np
from math import sqrt 
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data,predict,k=3):
	if len(data) >= k :
		warnings.warn('K is set to value less than total voting groups!')
	distances = []
	for group in data :
		for features in data[group] :
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance,group])
	votes = [i[1] for i in sorted(distances)[:k]]
	#print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]


	return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace =True)
df.drop(['id'],1,inplace=True)
full_data =df.astype(float).values.tolist() #Trying to escape the quote values ! Making sure they are float , not String !
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}


train_data = full_data[:-int(test_size*len(full_data))] #Getting the first $test_size part of the full_data
test_data = full_data[-int(test_size*len(full_data)):] #Getting the last  $test_size part of the full_data

for i in train_data:
	train_set[i[-1]].append(i[:-1]) #Append all the features except the class (-because we already classified while creating this dictionary and we assign the data where it belongs 2 or 4 indexes in the dictionary !)
for i in test_data:
	test_set[i[-1]].append(i[:-1]) #Do the same thing to the test set
correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote = k_nearest_neighbors(train_set,data,k=5)
		if group == vote :
			correct +=1
		total +=1
print('Accuracy',float(correct)/float(total))






