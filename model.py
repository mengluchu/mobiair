import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
#from xgboost import plot_importance
from matplotlib import pyplot

from osgeo import gdal

# from sklearn.model_selection import train_test_split

from datetime import datetime
import pcraster


random_seed = 1
cpus = 24



def train(varname):
  ap = pd.read_csv('glo4variables.csv')
  ap_pred = ap.filter (regex='road|nightlight|population|temp|wind|trop|indu|elev|radi')

  s = datetime.now()
#  Xtrain_rf, Xtest_rf, Ytrain2, Ytest2 = train_test_split(ap_pred, ap[varname + '_value'], test_size=0.0, random_state=random_seed)
  print('split    {}'.format(datetime.now() - s ))

  Xtrain_rf = ap_pred

  var = varname + '_value'
  Ytrain2 = ap.filter (regex=var)
  columns = list(ap_pred.columns)
  print(columns)

  columns2 = list(Ytrain2.columns)
  print(columns2)
  return Xtrain_rf, Ytrain2, columns



def get_rasters(src_dir, columns):
  s = datetime.now()
  dir_path = os.path.join(str(src_dir), 'laea')
  result = {}
  for col in columns:
      fname = '{}.map'.format(col)
      path = os.path.join(dir_path, fname)
      raster = pcraster.readmap(path)
      arr = pcraster.pcr2numpy(raster, np.nan)
      arr2 = np.reshape(arr, (-1,))

      result[col] = arr2

  df = pd.DataFrame.from_dict(result)

  print('raster   {}'.format(datetime.now() - s ))

  return df


def fit(Xtrain, Ytrain):
  s = datetime.now()
  #xg_reg = xgb.XGBRegressor(n_jobs = cpus, objective = "reg:squarederror", booster = "dart", learning_rate = 0.007, max_depth = 6, n_estimators = 3000, gamma = 5, reg_alpha = 0, reg_lambda = 2, random_state=random_seed)
  xg_reg = xgb.XGBRegressor(n_jobs = cpus, objective = "reg:squarederror", booster = "gbtree", learning_rate = 0.007, max_depth = 6, n_estimators = 3000, gamma = 5, reg_alpha = 0, reg_lambda = 2, random_state=random_seed)
  print(xg_reg)
  xg_reg.fit(Xtrain, Ytrain)
  pyplot.rcParams.update({'font.size': 4})
  xgb.plot_importance(xg_reg, grid=False, importance_type='gain', title='Feature importance {}'.format(Ytrain.columns[0]))

  fname = '{}.png'.format(Ytrain.columns[0])
  pyplot.savefig(fname, dpi=600)
  print('fit      {}'.format(datetime.now() - s ))
  return xg_reg

def predict(reg, data, src_dir, var_name):
  s = datetime.now()
  res = reg.predict(data)
  print('predict  {}'.format(datetime.now() - s ))
  path = os.path.join(str(src_dir), 'laea', 'clone_25m.map')
  pcraster.setclone(path)
  rows = pcraster.clone().nrRows()
  cols = pcraster.clone().nrCols()
  cl = pcraster.clone().cellSize()
  raster_np = res.reshape(rows,cols)
  raster = pcraster.numpy2pcr(pcraster.Scalar, raster_np, -999)
  clone = pcraster.readmap(path)
  raster = pcraster.ifthen(clone, raster)

  ofname = os.path.join(str(src_dir), 'laea', '{}.map'.format(var_name))
  pcraster.report(raster, ofname)




def calc(csv_filename):
  varnames = ['wkd_day', 'wnd_day', 'wkd_night', 'wnd_night']

  centres = np.loadtxt(csv_filename, delimiter=',')

  directories = [int(v) for v in centres[:,2]]
  directories = [2]


  for varname in varnames:
    Xtrain_rf, Ytrain2, columns = train(varname)


    reg = fit(Xtrain_rf, Ytrain2)


    for directory in directories:

      data = get_rasters(directory, columns)


      res = predict(reg, data, directory, varname)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Name of the projection centre file")
    args = parser.parse_args()

    lookup = calc(args.input)
