#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:46:13 2018

@author: ryan_g
"""


import pandas as pd


train_data=pd.read_csv('train.csv',header=0)
test_data=pd.read_csv('test.csv',header=0)

#print(train_data.penalty.unique())

train_data['penalty_num'] = train_data.penalty.map({'none':0,  'l2':1,  'l1':2,  'elasticnet':3})
test_data['penalty_num'] = test_data.penalty.map({'none':0,  'l2':1,  'l1':2,  'elasticnet':3})

#print(train_data.columns)

features=['l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs',
       'n_samples', 'n_features', 'n_classes', 'n_clusters_per_class',
       'n_informative', 'flip_y', 'scale', 'penalty_num']

X=train_data[features]
X_Test=test_data[features]
Y=train_data['time']


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.175)


"""
#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(X_train)
X_train = minmaxscaler.transform(X_train)
X_test = minmaxscaler.transform(X_test)
X_Test =minmaxscaler.transform(X_Test)

LR = LinearRegression()
model = LR.fit(X_train, Y_train)

"""
"""
#ridge
from sklearn.linear_model import Ridge

from sklearn.preprocessing import MaxAbsScaler

maxabsscaler =MaxAbsScaler()
maxabsscaler.fit(X_train)
X_train = maxabsscaler.transform(X_train)
X_test = maxabsscaler.transform(X_test)
X_Test =maxabsscaler.transform(X_Test)

clf = Ridge()
model=clf.fit(X_train, Y_train)
 
Y_test_pred=model.predict(X_test)


from sklearn import metrics
print(metrics.mean_squared_error(Y_test, Y_test_pred))


Y_pred=model.predict(X_Test)
"""
"""
from sklearn.linear_model import BayesianRidge

from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(X_train)
X_train = minmaxscaler.transform(X_train)
X_test = minmaxscaler.transform(X_test)
X_Test =minmaxscaler.transform(X_Test)

br = BayesianRidge()
model=br.fit(X_train, Y_train)

Y_test_pred=model.predict(X_test)
from sklearn import metrics
print(metrics.mean_squared_error(Y_test, Y_test_pred))

Y_pred=model.predict(X_Test)

"""


"""
#lasso
from sklearn.linear_model import Lasso


from sklearn.preprocessing import MaxAbsScaler

maxabsscaler =MaxAbsScaler()
maxabsscaler.fit(X_train)
X_train = maxabsscaler.transform(X_train)
X_test = maxabsscaler.transform(X_test)
X_Test =maxabsscaler.transform(X_Test)

reg = Lasso(alpha = 0.1)
model=reg.fit(X_train, Y_train)

Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=10000,
   normalize=False, positive=False, precompute=False, random_state=None,
  selection='cyclic', tol=0.0001, warm_start=False)

Y_test_pred=model.predict(X_test)
from sklearn import metrics
print(metrics.mean_squared_error(Y_test, Y_test_pred))

Y_pred=model.predict(X_Test)

"""

#neural network
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_Test =scaler.transform(X_Test)

from sklearn.neural_network import MLPRegressor
#model = MLPRegressor(solver='adam', alpha=1e-4, hidden_layer_sizes=(65,57,48,40,33),max_iter=10000)
model = MLPRegressor(activation='relu', alpha=1e-03, batch_size='auto',
       early_stopping=False,
       epsilon=1e-04, hidden_layer_sizes=(), learning_rate='constant',
       learning_rate_init=0.005, max_iter=10000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.9, random_state=0, shuffle=True,
       solver='adam', tol=0.002, validation_fraction=0.002, verbose=False,
       warm_start=False)

model.fit(X_train, Y_train)
Y_test_pred=model.predict(X_test)
Y_train_pred=model.predict(X_train)

from sklearn import metrics
print(metrics.mean_squared_error(Y_test, Y_test_pred))
print(metrics.mean_squared_error(Y_train, Y_train_pred))

Y_pred=model.predict(X_Test)








"""
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(X_train)
X_train = minmaxscaler.transform(X_train)
X_test = minmaxscaler.transform(X_test)
X_Test =minmaxscaler.transform(X_Test)

from sklearn import svm
clf = svm.SVR()  
clf.fit(X_train, Y_train) 
Y_pred=clf.predict(X_Test)


from sklearn import metrics
print(metrics.mean_squared_error(Y_test, Y_test_pred))


Y_pred=model.predict(X_Test)
"""



"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_Test =scaler.transform(X_Test)

from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(X_train)
X_train = minmaxscaler.transform(X_train)
X_test = minmaxscaler.transform(X_test)
X_Test =minmaxscaler.transform(X_Test)
"""







