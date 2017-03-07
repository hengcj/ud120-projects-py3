#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

#########################################################
from sklearn import svm

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

# clf = svm.SVC(kernel='rbf', C=10000)
# t0 = time()
# clf.fit(features_train, labels_train)
# print('fit time:', round(time() - t0, 3), 's')
#
# t1 = time()
# pred = clf.predict(features_test)
# print('predict time:', round(time() - t1, 3), 's')
# # print('predict result:', pred)
#
# chrisPred = [pred[ii] for ii in range(0, len(pred)) if pred[ii] == 1]
# print('chris email num:', len(chrisPred))
#
# score = clf.score(features_test, labels_test)
# print('score:', score)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

parameters = {'kernel':['rbf'], 'C': [1, 100]}
svr = svm.SVC()
clf = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=-1)
t0 = time()
clf.fit(features_train, labels_train)
print('best_params_:',clf.best_params_)
print('fit time:', round(time() - t0, 3), 's')

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

t1 = time()
y_true, y_pred = labels_test, clf.predict(features_test)
print('predict time:', round(time() - t1, 3), 's')
print(classification_report(y_true, y_pred))


