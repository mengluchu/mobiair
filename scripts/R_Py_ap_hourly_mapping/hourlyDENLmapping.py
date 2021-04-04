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
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask 
import geopandas as gpd
 
#os.getcwd()

# if we just do random sampling in space-time. We have a quite high R2. 
# But it is because if the Location 1 t1 is in the training, then the L1 t2 is going to be small. So it is not a reliable accuracy assessment. 
# Most importantly, for the location we want to predict, we dont know the entire time series.
aplong_ulr = 'https://raw.githubusercontent.com/mengluchu/mobiair/master/mapping_data/DENL17_hr.csv'
spreadurl = 'https://raw.githubusercontent.com/mengluchu/mobiair/master/mapping_data/DENL17_hr_spread.csv'
res = 100 
ap = pd.read_csv(spreadurl)

#if for only 100m 
if res is 100:
    ap = ap.drop(ap.filter(regex='_25$|_50$').columns, axis = 1)
ap.shape
train_size = int(0.8* len(ap)) 

# make sure your github is open! 
def random_testtrain():
    ap = pd.read_csv(aplong_ulr)
    ap_pred = ap.filter (regex="pop|nig|trop|ele|wind|temp|ind|GH|road|value|hour")
    X_train, X_test, Y_train, Y_test = train_test_split(ap_pred, ap['wkd_hr_value'], test_size=0.2, random_state=42)
    X_train["hours"] = X_train["hours"].astype(int)
    
    X_train =X_train.drop(columns = "wkd_hr_value")
    
    X_test["hours"] = X_test["hours"].astype(int)
    X_test =X_test.drop(columns = "wkd_hr_value")
    #print(X_test.shape, X_train.shape, Y_train.shape, Y_test.shape)
    
    xg_reg = xgb.XGBRegressor(objective = "reg:squarederror",booster = "dart", learning_rate = 0.007, max_depth =6 , n_estimators = 300,gamma =5, alpha =2) 
    xg_reg.fit(X_train ,Y_train) # predictor at the station and station measurements
    y_hatxgb = xg_reg.predict(X_test) # 1 degree tile
    print("mae", abs(y_hatxgb - Y_test).mean(),  "rmse",math.sqrt(((y_hatxgb - Y_test)*(y_hatxgb - Y_test)).mean()), "r2",metrics.r2_score(Y_test,y_hatxgb))
    
    xgb.plot_importance(xg_reg, grid=False, max_num_features= 50, importance_type='gain', title='Feature importance')
      
random_testtrain()

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
 

# way 2 easier 
def get_r2_numpy_corrcoef(x, y):
    return np.corrcoef(x, y)[0, 1]**2
def fitxgb (X_train, Y_train, X_test, Y_test, plot =False, r2=True):
    xg_reg = xgb.XGBRegressor(objective = "reg:squarederror",booster = "dart", learning_rate = 0.007, max_depth =6 , n_estimators = 300,gamma =5, alpha =2) 
    xg_reg.fit(X_train ,Y_train) # predictor at the station and station measurements
    yhatxgb = xg_reg.predict(X_test) # 1 degree tile
    mae = abs(yhatxgb - Y_test).mean()
    rmse =  math.sqrt(((yhatxgb - Y_test)*(yhatxgb - Y_test)).mean())
    rrmse = rmse / Y_test.median()
    if r2:
        r2 =metrics.r2_score(Y_test, yhatxgb)     
    else:
        r2 = get_r2_numpy_corrcoef(Y_test, yhatxgb)
    print("mae", mae ,  "rrmse", rrmse, "r2",r2)
    #math.sqrt(metrics.mean_squared_error(Y_test, y_hatxgb))
    #metrics.r2_score(Y_test,y_hatxgb)
    #print("mae", "rmse", abs(yhatxgb - Y_test).mean(), math.sqrt(((yhatxgb - Y_test)*(yhatxgb - Y_test)).mean()))
    if plot:
        xgb.plot_importance(xg_reg, grid=False, max_num_features= 30, importance_type='gain', title='Feature importance')
        plt.savefig('/Users/menglu/Documents/Github/xgb.png', dpi=1200)
    return(mae, rmse, rrmse, r2)


def fit_lightgbm(X_train, Y_train, X_test, Y_test,  r2=True):
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
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    mae = abs(y_pred - Y_test).mean()
    rmse =  math.sqrt(((y_pred - Y_test)*(y_pred - Y_test)).mean())
    rrmse = rmse / Y_test.median()
    if r2:
        r2 =metrics.r2_score(Y_test, y_pred)     
    else:
        r2 = get_r2_numpy_corrcoef(Y_test, y_pred)
    print("mae", mae ,  "rrmse", rrmse, "r2",r2)
    #math.sqrt(metrics.mean_squared_error(Y_test, y_hatxgb))
    #metrics.r2_score(Y_test,y_hatxgb)
    #print("mae", "rmse", abs(yhatxgb - Y_test).mean(), math.sqrt(((yhatxgb - Y_test)*(yhatxgb - Y_test)).mean()))
    return (mae, rmse, rrmse, r2)

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

#not used in this study 
def gdal_2array(ras_dir, X_train, savename = None) :  #X_train: dataframe used for training
    le = len(X_train)
    result = []
    for i in X_train.columns: # dataframe names
        rasterdir = f'{ras_dir}/{i}.tif'
        #le = len(os.listdir(rasterdir ) )
        arr = np.array(gdal.Open(rasterdir).ReadAsArray())
        result.append(arr)
        print(i)
    
    result = np.array(result) 
    result = np.moveaxis (result, 0, 2)
    if savename is not None:
        np.save(savename, result)
    return (result)

def get_province (filedir, rasterfile=f'{ras_dir}/wind_speed_10m_9.tif', provname = "Utrecht"):   
     
    file_path = filedir+"/NLD_adm_shp/NL_poly.shp"

    gdf_prov = gpd.read_file(file_path)
    #rd new  projection
    Utrecht = gdf_prov[gdf_prov.PROV_NAAM == provname]
    if rasterfile is not None:      # if None then project later
        src = rasterio.open(rasterfile)
        rcrs = src.crs
        Utrecht = Utrecht.to_crs (rcrs)
    return (Utrecht)

def crop_2array(ras_dir, X_train, polygon=Utrecht, savename = None):
    le = len(X_train)
    utrecht_ras = []
    for i in X_train.columns:
            rasterdir = f'{ras_dir}/{i}.tif'
            #le = len(os.listdir(rasterdir ) )
            arr = np.array(gdal.Open(rasterdir).ReadAsArray()   )
            
            print(i)
        
            with rasterio.open(rasterdir) as src:
                rcrs = src.crs
                polygon = polygon.to_crs (rcrs) 
                
                out_image, out_transform = rasterio.mask.mask(src, polygon['geometry'], crop=True)
                out_image = out_image.squeeze()
                out_meta = src.meta
                out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[0],
                 "width": out_image.shape[1],
                 "transform": out_transform})

                
                out_image[out_image <0]=np.nan
                utrecht_ras.append(out_image)
    
    Ut_ras = np.array(utrecht_ras) 
    Ut_ras = np.moveaxis (Ut_ras, 0, 2)
    if savename is not None:
        np.save(savename, Ut_ras)  
    return (Ut_ras, out_meta)

def plot_predictor (Ut_ras)
    fig, axs = plt.subplots(nrows=8, ncols=7, figsize=(55,15),
                            subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axs.flat):
        if i < Ut_ras.shape[2]:
            
            a = ax.imshow(Ut_ras[:,:,i])
            ax.set_title(X_train.columns[i])
            fig.colorbar(a,  ax=ax)
        
    #    ax.colorbar()
    plt.savefig(filedir+"prediUt.png")

 
def predicthourly(hr, out_meta):
    X_train, Y_train, X_test, Y_test = preprocessing(ap,1, False, True , hr ) # do it for all the hours
    gbm = onlyfit_lightgbm(X_train, Y_train, X_test, Y_test) 
    # only evaluate using the gbm.fit but not doing it manually (i.e. not returning the metrics)
    
    #test1 = np.random.rand(3,3,X_test.shape[1])
    t1 = np.apply_along_axis(rasterlgm, 2, Ut_ras, gbm)
    t1 = t1.astype("float32")
    #    plt.imshow(t1)      
    t1 = t1.squeeze()
    t1 = np.expand_dims(t1, axis=0)
    with rasterio.open(f'{filedir}/prediction/NL100_t{hr}.tif', 'w', **out_meta) as dst:
        dst.write(t1)

filedir = '/Users/menglu/Documents/Github/mobiair'
ras_dir =  os.path.join("/Volumes","Meng_Mac", "NL_100m") 
   
Utrecht = get_province(filedir, None, "Utrecht") # project later

X_train, Y_train, X_test, Y_test = preprocessing(ap, 1, False, True , 1 ) # only for getting the names
Ut_ras, out_meta = crop_2array(ras_dir, X_train) # get all the rasters

for i in range (0,2):
    predicthourly(i, out_meta)
    

  #inspect.signature(MERF)
def merf(normalise = False):
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'rmse'],
        'learning_rate': 0.001,
        'feature_fraction': 0.8,
         
        "max_depth": 6,
        "max_bin": 512,
        "num_leaves": 40,
        "num_iterations": 100000,
        "n_estimators": 300,
        "verbose": -1
        }
      #   'bagging_fraction': 0.7,
      #  'bagging_freq': 10, "num_leaves": 12, 
      
    gbm = lgb.LGBMRegressor(**hyper_params)    
    
    ap2 = ap.fillna(method = "pad") 
    ap2.isna().sum().sum()
    X_train, Y_train, X_test, Y_test = preprocessing(ap2, hour2int = True, onehotencode = False)
      
    Z_train = np.ones((len(X_train), 1))
    
    clusters_train = X_train['hours']
    clusters_test= X_test['hours']
    
    X_train1 = X_train.drop(["hours"],axis = 1)
    
    X_test1 = X_test.drop(["hours"],axis = 1)
    
    if normalise:
        X_train1 =(X_train1-X_train1.mean())/X_train1.std()
        X_test1 =(X_test1-X_test1.mean())/X_test1.std()
    # we should not nornalise the Y (response)    
    #   Y_train1 =(Y_train-Y_train.mean())/Y_train.std()
         
    #my_imputer = SimpleImputer()
    #X_train1 = my_imputer .fit_transform(X_train1)   # fit missing
    #X_test1  = my_imputer .fit_transform(X_test1)  
    
    # normalising for boosting is commonly not necessary, but for the mixed effect models 
    # we actually may want to normalise. But we only normalise X (predictors)!
       # check if missing 
    print( Y_train1.isnull().any().any(),X_train1.isnull().any().any(),X_test.isnull().any().any())
    
    merf = MERF(gbm, max_iterations = 4)
    merf.fit(X_train1,  Z_train, clusters_train, Y_train1)
    
    Z_test = np.ones((len(X_test1), 1))
    y_pred_ = merf.predict(X_test1, Z_test, clusters_test)
    # also normalise the response and prediction wont work
    #if normalise:
    #    y_pred = y_pred_*Y_train.std()+Y_train.mean() 
        
    mae = abs(y_pred - Y_test).mean()
    rmse =  math.sqrt(((y_pred - Y_test)*(y_pred - Y_test)).mean())
    rrmse = rmse / Y_test.median()
    r2 = get_r2_numpy_corrcoef(Y_test, y_pred)
    return(mae, rmse, rrmse, r2)

# merf result: it is much slower because it fits gbm n times. 
#mae, rmse, rrmse, r2 = merf()

# hour to int 
X_train, Y_train, X_test, Y_test = preprocessing(ap, hour2int = True, onehotencode = False)
mae1, rmse1, rrmse1, r21 = fitxgb (X_train, Y_train, X_test, Y_test,r2 = False)
lgbmae1, lgbrmse1, lgbrrmse1, lgbr21 = fit_lightgbm(X_train, Y_train, X_test, Y_test, r2 = False) 
# onehot encoding
X_train, Y_train, X_test, Y_test = preprocessing(ap, hour2int =False, onehotencode = True)
mae2, rmse2, rrmse2, r22 = fitxgb (X_train, Y_train, X_test, Y_test, r2 = False, plot=False)
lgbmae2, lgbrmse2, lgbrrmse2, lgbr22= fit_lightgbm(X_train, Y_train, X_test, Y_test,r2 = False) 
# select one hour (time step)
#as_int_xgb = pd.DataFrame([mae1, rmse1, rrmse1, r21], columns = ["as_int_xgb"])





result = pd.DataFrame({"as_int_xgb": [mae1, rmse1, rrmse1, r21],
                           "as_int_lgb":[lgbmae1, lgbrmse1, lgbrrmse1, lgbr21],
                           "onehot_xgb":[mae2, rmse2, rrmse2, r22 ],
                           "onehot_lgb":[lgbmae2, lgbrmse2, lgbrrmse2, lgbr22]})
                          # ,"merf": [mae, rmse, rrmse, r2]}                      

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2  
            _y = p.get_y() + p.get_height() +0.1
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 
            
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
        print(2)    
    else:
        _show_on_single_plot(axs)

def plotmetrics(result):
    result.index=["mae","rmse","rrmse", "r2"]

    result['metrics'] = result.index
 
    meltresult = pd.melt(result,id_vars=['metrics'])
    ax = sns.barplot(data = meltresult, x= "metrics", y = "value", hue = "variable")
    show_values_on_bars(ax)
    plt.plot()
    
plotmetrics(result)  

 #hourly
def getmatrix ():
    r1 = {}
    rrmset = {}
    rmset = {}
    for i in range(24):
        X_train, Y_train, X_test, Y_test = preprocessing(ap, False, True , i )
        _, rmse, rrmse , r2 = fitxgb (X_train, Y_train, X_test, Y_test, plot = False, r2 = False)
        r1 = np.append(r1,r2)
        rrmset = np.append(rrmset, rrmse)
        rmset= np.append(rmset, rmse)
         
        print(i)
    r1 = np.delete(r1, np.where(np.nan))
    rrmset= np.delete(rrmset, np.where(np.nan))
    rmset = np.delete(rmset, np.where(np.nan))    
    return rmset, rrmset, r1

rmse, rrmse, r2 =  getmatrix () 

#r1 = np.delete(r1, np.where(np.nan)) 
#r1
 
r1.mean() # 0.66
rrmse.mean()  # 0 .4
rmse.mean() # 9.1 
def plot1(obj, labl):
    
    plt.bar(range(len(obj)), obj )
    plt.ylabel(labl)
    plt.title("mean " + labl+":  "+ str(round(obj.mean(),1)))
    plt.show()
plt.figure(0)
plot1(rrmse, "RRMSE")
plt.figure(1)
plot1(rmse, "RMSE")
plt.figure(2)
plot1(r1 , "Rsquared")

#ligtht
X_train, Y_train, X_test, Y_test = preprocessing(ap, False, False, select_hr=None)
#catboost
X_train, Y_train, X_test, Y_test = preprocessing(ap, False, False, select_hr=None)
train_pool = Pool(X_train, 
                  Y_train, 
                  cat_features=["hours"])
test_pool = Pool(X_test, 
                 cat_features=["hours"]) 

model = CatBoostRegressor(iterations=300, 
                          depth=6, 
                          learning_rate=0.007, 
                          loss_function='RMSE')
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
preds = model.predict(test_pool)
print(preds)

#1 = np.delete(r1, np.where(r1 ==16)) 
#r1 == np.nan_to_num
#r1[16] ="NaN"
#type(np.nan)
#r1.mean()
#r1  /16  # 0.56 0-16
#r1 /5: # 0.625 19-23
#catboost
 
 
# encode string class values as integers
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
X = X_train


 
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
# way 1
def fitxgb (X_train, Y_train, X_test, Y_test):
    dtrain = xgb.DMatrix(X_train, label=Y_train )
    dtest = xgb.DMatrix(X_test, label=Y_test)
    params = {'max_depth': 6, 'eta': 0.007, 'silent': 1, 'n_estimators' : 300 }
    model = xgb.train(params, dtrain,  100)
    yhatxgb = model.predict(dtest)# Fit
    abs(yhatxgb - Y_test).mean()
    math.sqrt(((yhatxgb - Y_test)*(yhatxgb - Y_test)).mean())
    xgb.plot_importance(model, grid=False, max_num_features= 30, importance_type='gain', title='Feature importance')
    plt.savefig('/Users/menglu/Documents/Github/xgb.png', dpi=1200)

        #encoded_xtest = X_test.drop(columns = "hours")
        #label_encoder = LabelEncoder()
        #feature = label_encoder.fit_transform(X_train["hours"])
        #feature = feature.reshape(X_train.shape[0], 1)
        #onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        #feature = onehot_encoder.fit_transform(feature)