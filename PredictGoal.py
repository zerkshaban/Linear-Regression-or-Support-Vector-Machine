# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:30:51 2017
@author: Zerk Shaban
"""

import pandas as pd
import math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression


df= pd.read_csv('data_v1.csv')

#insepection of dataset
print(df.head())
print(df.tail())

#normalizing the column names inconsistency
df.rename(columns={'Date':'date','Season':'season','home':'home','visitor':'visitor','hgoal':'hgoal','vgoal':'vgoal'},inplace=True)

print(df.columns) #data inconsistency
print(df.shape)   #rows and columns of data set
print(df.info())  #to check for missing values in column of large dataset

df =df[['season','home','visitor','hgoal','vgoal']]

df['precentage_change']=(df['hgoal']-df['vgoal'])/df['vgoal']*100

df = df[['season','vgoal','hgoal','precentage_change']]

forcast_col = 'vgoal'
df.fillna(method='ffill',inplace=True)

forcast_out = int(math.floor(0.0001*len(df)))

df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X)

#X =X[:-forcast_out+1]
df.dropna(inplace=True)

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#uncomment any of the following classifier
#clf = LinearRegression(n_jobs=10) 
clf = svm.SVR()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)


print('Accuracy =',accuracy*100)