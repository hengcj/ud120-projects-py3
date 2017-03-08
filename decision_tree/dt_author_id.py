#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

# print(len(features_train[0]))
t0 = time()
clf1 = tree.DecisionTreeClassifier(min_samples_split=40)
clf1.fit(features_train, labels_train)
print('fit time:', round(time() - t0, 3), 's')

t1 = time()
score1 = clf1.score(features_test, labels_test)
print(score1)
print('predict time:', round(time() - t1, 3), 's')

# t0 = time()
# clf2 = tree.DecisionTreeClassifier(min_samples_split=50)
# clf2.fit(features_train, labels_train)
# print('fit time:', round(time() - t0, 3), 's')
#
# t1 = time()
# score2 = clf2.score(features_test, labels_test)
# print(score2)
# print('predict time:', round(time() - t1, 3), 's')

#########################################################


