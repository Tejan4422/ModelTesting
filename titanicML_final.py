# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:02:39 2020

@author: Tejan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

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


#Testing the best model
val_features = X_val
val_labels = y_val

te_features = X_test
te_labels = y_test

models = {}
for mdl in ['LR', 'SVC', 'NN', 'RF', 'GB']:
    models[mdl] = joblib.load('C:/Users/Tejan/.spyder-py3/work/Ex_Files_Machine_Learning_Algorithms/Exercise Files/{}_model1.pkl'.format(mdl))
   
def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{}-- Acuuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
          accuracy,
          precision,
          recall,
          round(end - start)*1000, 1))

for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)
    
for name, mdl in models.items():
    evaluate_model(name, mdl, te_features, te_labels)
    
evaluate_model('Random FOrest', models['RF'], te_features, te_labels)    
        
    
    
    