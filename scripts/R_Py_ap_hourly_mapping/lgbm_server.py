#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:24:49 2021

@author: menglu
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from scipy import stats
from pandas import read_csv  
from sklearn.impute import SimpleImputer 
import gdal
import math
import inspect
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import rasterio
import rasterio.mask 
import geopandas as gpd
 
filedir = '/data/projects/mobiair'
ras_rootdir = '/data/gghdc/gap/2021/output/areas/'


import osmnx as ox
wuhan = ox.geocode_to_gdf('武汉, China')
utrecht = ox.geocode_to_gdf('Utrecht') 
 
#os.getcwd()

# if we just do random sampling in space-time. We have a quite high R2. 
# But it is because if the Location 1 t1 is in the training, then the L1 t2 is going to be small. So it is not a reliable accuracy assessment. 
# Most importantly, for the location we want to predict, we dont know the entire time series.
spreadurl = 'https://raw.githubusercontent.com/mengluchu/mobiair/master/mapping_data/DENL17_hr_spread.csv'
res = 25
ap = pd.read_csv(spreadurl)

#if for only 100m 
if res == 100:
    ap = ap.drop(ap.filter(regex='_25$|_50$').columns, axis = 1)
ap.shape
train_size = int(0.8* len(ap)) 

# make sure your github is open! 
 
# let's do it properly 

# sampling only over space. because prediction over time is better than over space (i.e.lower temporal variance). But for the location we want to predict, we dont know the entire time series. if we have value at t1, it will predict t2 very well
def wide2long (wide):
    pm = pd.melt(wide, var_name = "hours", value_vars= [str(i) for i in range(24)])  # default value_name "value"
    print(pm.index)
    x=wide.filter (regex="pop|nig|trop|ele|wind|temp|ind|GH|road")
    x1 = pd.concat([x]*24, ignore_index=True)

    pmlong = pm.join(x1, how = "outer") # use outer to check mistakes, if correct it should be all the same all kinds of joins
    return pmlong

def preprocessing (ap, rand_stat=1, hour2int = False, onehotencode = True, select_hr = None):
    ap_pred = ap.sample(frac=1,   random_state=rand_stat ) #shuffel
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
    
    if hour2int:
        X_train["hours"] = X_train["hours"].astype(int)
        X_test["hours"] = X_test["hours"].astype(int)
    
    if onehotencode:
        encoded_x = X_train.drop(columns = "hours")

        feature_train=pd.get_dummies(X_train["hours"])  
        feature_test=pd.get_dummies(X_test["hours"])  
        if encoded_x is None:
            X_train = feature_train
            X_test = feature_test
        else:
             X_test= X_test.join(feature_test).drop(columns = "hours")
             X_train= X_train.join(feature_train).drop(columns = "hours")
             
    if select_hr is not None:
        XY_train2 = XYtrain[XYtrain["hours"] == str(select_hr)].drop(columns = "hours")
        XY_test2 = XYtest[XYtest["hours"] == str(select_hr)].drop(columns = "hours")
        X_train = XY_train2.drop(columns = "value")
        Y_train = XY_train2[ "value"]   
        X_test = XY_test2.drop(columns = "value")
        Y_test = XY_test2[ "value"]
       
        print("X shape: : ", X_train.shape, X_test.shape)

    return X_train, Y_train, X_test, Y_test

#X_train, Y_train, X_test, Y_test = preprocessing(ap, rand_stat = 1, hour2int=False, onehotencode=True, select_hr=1)
#X_train, Y_train, X_test, Y_test
 

def onlyfit_lightgbm(X_train, Y_train, X_test, Y_test):
    hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1', 'rmse'],
    'learning_rate': 0.007,
    'feature_fraction': 0.8,
     'verbose': 0,
    "max_depth": 6,
    "max_bin": 512,
    "num_leaves": 40,
    "num_iterations": 100000,
    "n_estimators": 300,
    "verbose":-1
    }
  #   'bagging_fraction': 0.7,
  #  'bagging_freq': 10, "num_leaves": 12, 
  
    gbm = lgb.LGBMRegressor(**hyper_params)    
    gbm.fit(X_train ,Y_train, eval_set=[(X_test, Y_test)],
       eval_metric='l1',
        early_stopping_rounds=1000, verbose = 2000) # predictor at the station and station measurements
    return (gbm)

def rasterlgm(X_test, gbm):
    # predictor at the station and station measurements
    y_pred = gbm.predict(pd.DataFrame(X_test).T, num_iteration=gbm.best_iteration_)
    return (y_pred)


# read tif as array
# ras_dir for projecting the polygon
def get_province (filedir, ras_dir = None, provname = "Utrecht"):   
    rasterfile=f'{ras_dir}/wind_speed_10m_9.tiff' 
    file_path = filedir+"/NLD_adm_shp/NL_poly.shp"

    gdf_prov = gpd.read_file(file_path)
    #rd new  projection
    Utrecht = gdf_prov[gdf_prov.PROV_NAAM == provname]
    if rasterfile is not None:      # if None then project later
        src = rasterio.open(rasterfile)
        rcrs = src.crs
        Utrecht = Utrecht.to_crs (rcrs)
    return (Utrecht)

def crop_2array(ras_dir, X_train,maskpoly=None, savename = None):
    le = len(X_train)
    utrecht_ras = []
    for i in X_train.columns:
            rasterdir = f'{ras_dir}/{i}.tiff'
            #le = len(os.listdir(rasterdir ) )
            arr = np.array(gdal.Open(rasterdir).ReadAsArray()   )
            
            print(i)
        
            with rasterio.open(rasterdir) as src:
                rcrs = src.crs
                out_meta = src.meta
                if maskpoly is not None:
                   # maskpoly=maskpoly.set_crs("EPSG:4326")
                    maskpoly = maskpoly.to_crs (rcrs) 
                    
                    out_image, out_transform = rasterio.mask.mask(src, maskpoly['geometry'], crop=True)
                    out_image = out_image.squeeze()
                    
                    out_meta.update({"driver": "GTiff",
                                     "height": out_image.shape[0],
                                     "width": out_image.shape[1],
                                     "transform": out_transform})
                else:
                    out_image =src.read()   
                out_image = out_image.squeeze()
                
                out_image[out_image <0]=np.nan
                utrecht_ras.append(out_image)
    
    Ut_ras = np.array(utrecht_ras) 
    Ut_ras = np.moveaxis (Ut_ras, 0, 2)
    if savename is not None:
        np.save(savename, Ut_ras)  
    return (Ut_ras, out_meta)

 
def predicthourly(hr, out_meta, ap, Ut_ras, rastername):
    X_train, Y_train, X_test, Y_test = preprocessing(ap,1, False, True , hr ) # do it for all the hours
    gbm = onlyfit_lightgbm(X_train, Y_train, X_test, Y_test) 
    # only evaluate using the gbm.fit but not doing it manually (i.e. not returning the metrics)
    
    #test1 = np.random.rand(3,3,X_test.shape[1])
    t1 = np.apply_along_axis(rasterlgm, 2, Ut_ras, gbm)
    t1 = t1.astype("float32")
    #    plt.imshow(t1)      
    t1 = t1.squeeze()
    t1 = np.expand_dims(t1, axis=0)
    with rasterio.open(f'{filedir}/prediction/{rastername}_{hr}.tif', 'w', **out_meta) as dst:
        dst.write(t1)

#Utrecht = get_province(filedir, ras_dir, "Utrecht") # project later

citypoly = [utrecht, wuhan]
 
for j in ['106682', '111152']:
  #  rastername = str(os.listdir(ras_rootdir)[j])
    
    if j=="106682":
        cityp = citypoly[0]
        rastername = "Utrecht"
    elif j=="111152":
        cityp = citypoly[1]
        rastername= "Wuhan"
    ras_dir = ras_rootdir+ j
    
    X_train, Y_train, X_test, Y_test = preprocessing(ap, 1, False, True , 1 ) # only for getting the names
    Ut_ras, out_meta = crop_2array(ras_dir, X_train, maskpoly = cityp) # get all the rasters
    
    
     
    for i in range (0,24):
        predicthourly(i, out_meta,ap, Ut_ras, rastername)

 


