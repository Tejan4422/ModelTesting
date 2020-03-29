# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:44:29 2020

@author: Tejan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)

dataset = pd.read_csv('titanic.csv')
dataset.head()
dataset.describe()
dataset.isnull().sum()

dataset['Age'].fillna(dataset['Age'].mean(), inplace = True)


for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x = col, y = 'Survived', data = dataset, kind = 'point', aspect = 2)
    
dataset['FamilyCnt'] = dataset['SibSp'] + dataset['Parch']

dataset.drop(['PassengerId', 'SibSp', 'Parch'], axis = 1, inplace = True)

dataset.groupby(dataset['Cabin'].isnull())['Survived'].mean()

dataset['CabinInd'] = np.where(dataset['Cabin'].isnull(), 0, 1)

gender_num = {'male' : 0, 'female' : 1}
dataset['Sex'] = dataset['Sex'].map(gender_num)

dataset.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)

features = dataset.drop('Survived', axis = 1)
labels = dataset['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)

for dataset1 in [y_train, y_val, y_test]:
    print(round(len(dataset1)/len(labels), 2))

tr_features = X_train
tr_labels = y_train

#LogisticRegression
from sklearn.linear_model import LogisticRegression

def print_results(results):
    print('BEST PARAMS : {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    std = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, std, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))
        
lr = LogisticRegression()
parameters = {
        'C' : [0.001, 0.01, 1, 10, 100, 1000]
        }
cv = GridSearchCV(lr, parameters, cv = 5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)

joblib.dump(cv.best_estimator_, 'C:/Users/Tejan/.spyder-py3/work/Ex_Files_Machine_Learning_Algorithms/Exercise Files/LR_model1.pkl')
    
#SVM
from sklearn.svm import SVC

def print_results_svm(results):
    print('BEST PARAMS : {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    std = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, std, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))

svc = SVC()
parameters_svc = {
        'kernel' : ['linear', 'rbf'],
        'C' : [0.1, 1, 10]
        
        }
cv_svc = GridSearchCV(svc, parameters_svc, cv = 5)
cv_svc.fit(tr_features, tr_labels.values.ravel())
print_results_svm(cv_svc)  
joblib.dump(cv_svc.best_estimator_, 'C:/Users/Tejan/.spyder-py3/work/Ex_Files_Machine_Learning_Algorithms/Exercise Files/SVC_model1.pkl')
  
#Neural NEtwork
from sklearn.neural_network import MLPClassifier

def print_results_NN(results):
    print('BEST PARAMS : {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    std = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, std, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))

mlp = MLPClassifier()
parameters_nn = {
        'hidden_layer_sizes' : [(10,), (50,), (100,)],
        'activation' : ['relu', 'tanh', 'logistic'],
        'learning_rate' : ['constant', 'invscaling', 'adaptive']
        }
cv_nn = GridSearchCV(mlp, parameters_nn, cv = 5)
cv_nn.fit(tr_features, tr_labels.values.ravel())
print_results_NN(cv_nn)

cv_nn.best_estimator_
joblib.dump(cv_nn.best_estimator_, 'C:/Users/Tejan/.spyder-py3/work/Ex_Files_Machine_Learning_Algorithms/Exercise Files/NN_model1.pkl')

#RandomForest
from sklearn.ensemble import RandomForestClassifier
def print_results_rf(results):
    print('BEST PARAMS : {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    std = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, std, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))

rf = RandomForestClassifier()
parameters_rf = {
        'n_estimators' : [5, 50, 250],
        'max_depth' : [2,4,8,16,32, None]
        }
cv_rf = GridSearchCV(rf, parameters_rf, cv = 5)
cv_rf.fit(tr_features, tr_labels.values.ravel())
print_results_rf(cv_rf)
joblib.dump(cv_rf.best_estimator_, 'C:/Users/Tejan/.spyder-py3/work/Ex_Files_Machine_Learning_Algorithms/Exercise Files/RF_model1.pkl')


#Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
def print_results_gb(results):
    print('BEST PARAMS : {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    std = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, std, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))

parameters_gb = {
        'n_estimators' : [5,50,250,500],
        'max_depth' : [1,3,5,7,9],
        'learning_rate' : [0.01, 0.1, 1, 10, 100]
        }
cv_gb = GridSearchCV(gb, parameters_gb, cv = 5)
cv_gb.fit(tr_features, tr_labels.values.ravel())
print_results_gb(cv_gb)
joblib.dump(cv_gb.best_estimator_, 'C:/Users/Tejan/.spyder-py3/work/Ex_Files_Machine_Learning_Algorithms/Exercise Files/GB_model1.pkl')


