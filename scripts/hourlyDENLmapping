#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:42:10 2021

@author: menglu
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from scipy import stats
from pandas import read_csv  
from sklearn.impute import SimpleImputer 
from merf import MERF
import math
import inspect
from sklearn.ensemble import RandomForestRegressor
os.getcwd()
#ap = pd.read_csv('/Users/menglu/Documents/Github/mobiair/DENL17_hr.csv')
ap = pd.read_csv('/Users/dmenglu/Documents/Github/mobiair/DENL17_hr_spread.csv')
train_size = int(0.8* len(ap)) 
# sampling only over space. because prediction over time is presumably better than over space (i.e.lower temporal variance). if we have value at t1, it will predict t2 very well
def wide2long (wide):
    pm = pd.melt(wide, var_name = "hours", value_vars= [str(i) for i in range(24)])  # default value_name "value"
    print(pm.index)
    x=wide.filter (regex="pop|nig|trop|ele|wind|temp|ind|GH|road")
    x1 = pd.concat([x]*24, ignore_index=True)

    pmlong = pm.join(x1, how = "outer") # use outer to check mistakes, if correct it should be all the same all kinds of joins
    return pmlong

def preprocessing (ap):
    ap_pred = ap.sample(frac=1) #shuffel
    XY_train = ap_pred [:train_size]
    XY_test  = ap_pred [train_size:]
    # Define a size for your train set 
  
    # wide2long only for hourly and selected variables to avoid parameterization (only for this project). The "melt" in R should save lots of hassels. But we want to try to do everyth
    
    ap = ap.filter (regex="pop|nig|trop|ele|wind|temp|ind|GH|road|hour|value")
    
    XYtrain = wide2long(XY_train)
    XYtest = wide2long(XY_test)
    
    X_train = XYtrain.drop(columns = "value")
    Y_train = XYtrain[ "value"]
    
    X_test = XYtest.drop(columns = "value")
    Y_test = XYtest[ "value"]
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = preprocessing(ap)
X_train, Y_train, X_test, Y_test

xg_reg = xgb.XGBRegressor(objective = "reg:squarederror",booster = "dart", learning_rate = 0.007, max_depth =6 , n_estimators = 300,gamma =5, alpha =2) 
xg_reg.fit(X_train ,Y_train) # predictor at the station and station measurements
y_hatxgb = xg_reg.predict(X_test) # 1 degree tile
math.sqrt(metrics.mean_squared_error(Y_test, y_hatxgb))
metrics.r2_score(Y_test,y_hatxgb)
 
 
inspect.signature(MERF)
xgb_reg = xgb.XGBRegressor(objective = "reg:squarederror",booster = "dart", learning_rate = 0.007, max_depth =6 , n_estimators = 300,gamma =5, alpha =2) 

merf = MERF(xgb_reg, max_iterations = 20)
Z_train = np.ones((len(X_train), 1))

clusters_train = X_train['hours']
clusters_test= X_test['hours']
my_imputer = SimpleImputer()

X_train = my_imputer .fit_transform(X_train)   # fit missing
X_test  = my_imputer .fit_transform(X_test)  
merf.fit(X_train,  Z_train, clusters_train, Y_train)


    

# %% [code]
Z_test = np.ones((len(X_test), 1))
y_hat = merf.predict(X_test, Z_test, clusters_test)
y_hat

# %% [code]
metrics.explained_variance_score(y_hat, Y_test)


# %% [code]
metrics.r2_score(y_hat, Y_test)

# %% [code]
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, Y_train)
y_hatrf = rf.predict(X_test)

# %% [code]
metrics.r2_score(y_hatrf, Y_test)

# %% [code]
xg_reg = xgb.XGBRegressor(objective = "reg:squarederror",booster = "dart", learning_rate = 0.007, max_depth =6 , n_estimators = 3000,gamma =5, alpha =2) 
xg_reg.fit(X_train ,Y_train) # predictor at the station and station measurements
y_hatxgb = xg_reg.predict(X_test) # 1 degree tile
 

# %% [code]
metrics.r2_score(y_hatxgb, Y_test)

