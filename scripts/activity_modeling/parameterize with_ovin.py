#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:48:36 2021

@author: menglu
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import stats
from sklearn.impute import SimpleImputer 
import math
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw
import imp
import geopandas as gpd
from shapely.geometry import Point
import random
import scipy 
from scipy import stats as st 
import shapely.speedups
from shapely.ops import nearest_points
from shapely.geometry import  MultiPoint 

shapely.speedups.enable()
imp.reload(np)
#from scipy.stats import powerlaw as sci_powerlaw
Ovin = pd.read_csv('/Users/menglu/Downloads/Ovin.csv')
sports = gpd.read_file('/Users/menglu/Documents/GitHub/mobiair/locationdata/Ut_indoorsport.gpkg')
Uni= gpd.read_file('/Users/menglu/Documents/GitHub/mobiair/locationdata/Ut_Uni_coll.gpkg')
# travel distance get mean and standard deviation of the log distribution if the distributions are log normal.  

# check is lognormal
#st.shapiro(np.log(student_outdoor[['KAf_mean']].apply(lambda x: x+1)))[-1]<0.01

def log_mean_sd (df):
 mu =  df.apply(lambda x: x+1). apply(lambda x: np.log(x)).apply(lambda x: np.mean(x))
 sd =  df.apply(lambda  x: x+1). apply(lambda x: np.log(x)).apply(lambda x: np.std(x))
  
 return (*mu, *sd)

def mean_sd (df):
 mu =  df.apply(lambda x: np.mean(x))
 sd =  df.apply(lambda x: np.std(x))
  
 return (*mu, *sd)

def lognormal_mean_sd_scipy(df):
    shape, loc, scale = scipy.stats.lognorm.fit(df.apply(lambda x: x+1), floc=0)
    sd = shape
    mu = np.log(scale)
    return (mu, sd)
 
def gen_lnorm(df, method="manual"):
  if method == "manual":
        mu, sd = log_mean_sd(df)
      
  else:      
        mu, sd = lognormal_mean_sd_scipy(df)
      
        #mu1, sd1 = log_mean_sd(df)
        #print(mu-mu1,sd-sd1) for checking purpose. 
  gen = np.random.lognormal (mu, sd, 1)-1
  if gen < 0:
      gen = 0.1 
  return(gen)


# powerlaw is not so successful
#results = powerlaw.Fit(xmin = 0.1, xmax = 50, data = Ovin[['KAf_mean']])
#sim_dist = results.power_law.generate_random(1)
#results.plot_pdf()
 
 
#based on social occupation and travel goal, output work, outdoor, shopping distances 
def distance(socialpartition="Scholier/student"):
    work_dist = Ovin.query('Doel == "Werken" & MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
    
    outdoor_dist = Ovin.query('Doel == "Sport/hobby" &  MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
    shopping_dist = Ovin.query('Doel =="Winkelen/boodschappen doen"& MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
    return(work_dist,outdoor_dist,shopping_dist)

def disto2d(workloc, homeloc): # input geopoints. 
    xcoord_work = workloc.centroid.x
    ycoord_work = workloc.centroid.y
    xcoord_home = homeloc.centroid.x
    ycoord_home =homeloc.centroid.y
    
    dis, dur, route= r.distance(ycoord_home, xcoord_home, ycoord_work, xcoord_work, cls)
               
    return(dis, dur, route)

 
# get potential destination points and a union of it. 
def pot_dest(goal):
    if goal == "work":
            w_gdf = gpd.GeoDataFrame(crs={'init': 'epsg:4326'},
                                     geometry=[Point(xy) for xy in zip(workdf.lon, workdf.lat)])
    elif goal == "uni": 
            w_gdf = Uni
    elif goal == "shops":
            w_gdf = shops
    u = w_gdf.unary_union
    return (w_gdf, u)
 #p: home point, w_gdf: destination (e.g. work) geopandas, u, union geopandas, calculate once so move out of function. goal: activity, sopa: social participation
def getdestloc (p, w_gdf, u, goal = "work", sopa = "Scholier/student"):
    
    nearestpoint = nearest_points(p,u)[1]
    mindist = p.distance(nearestpoint)
  
    work_dist, outdoor_dist, shopping_dist = distance(sopa) 
    # calculate distance (radius)
    if  goal  == "work": 
        sim_dist = gen_lnorm(work_dist, "")
        sim_distdeg = sim_dist/110.8
    elif goal == "sports":
        sim_dist = gen_lnorm(outdoor_dist, "")
        sim_distdeg = sim_dist/110.8     
    elif goal == "shopping":
        sim_dist = gen_lnorm(shopping_dist, "")
        sim_distdeg = sim_dist/110.8
    
    if sim_distdeg < mindist :
        sim_distdeg = mindist
        des_p= nearestpoint
        num_points = 0
        print(f'use nearest point as simulated distance is too short.')
    
    # calculate a buffer and select points. 
    else:
        pbuf=p.buffer(sim_distdeg) # distance to degree`maybe better to project. 
        pbuf.crs={'init': 'epsg:4326', 'no_defs': True}
        worklocset =w_gdf[w_gdf.within(pbuf)]
        num_points = len(worklocset)
        print(f'sample from {num_points} points')
        workloc =  worklocset.sample(n = 1, replace=True, random_state=1)
        des_p = workloc.iloc[0]["geometry"]  
    return (p, des_p,num_points )



filedir = "~/Documents/GitHub/mobiair/locationdata/"

home_csv = filedir+"Uhomelatlon.csv"
homedf = pd.read_csv( home_csv) 
nr_locations = homedf.shape[0]
work_csv = filedir+"Uworklatlon.csv"  #working locations of each homeID. Will later group by homeID for sampling
workdf = pd.read_csv( work_csv)  #for randomly sample working locations
       
w_gdf, u  = pot_dest("uni")            

for id in range (10): 
    h =homedf.loc[id]
    p=Point(h.lon, h.lat) # distance to degree`maybe better to project. 
    
    op, dp, num_p = getdestloc(p, w_gdf, u)
    print(op, dp, num_p )
    MultiPoint([op,dp])

 

#disto2d(op, dp)
dp["geometry"]
.xy
op.plot()
#work_dist, outdoor_dist, shopping_dist = distance( "Werkzaam >= 30 uur per week") 

#work_dist, outdoor_dist, shopping_dist = distance( "Werkzaam 12-30 uur per week" ) 
#work_dist, outdoor_dist, shopping_dist = distance(  "Werkloos"  ) 

#log_mean_sd(work_dist)  # 1.31, 1.025 # slightly larger mean and lorger variance
#log_mean_sd(outdoor_dist)  # 1.27, 1.018
#log_mean_sd(shopping_dist)  # 1.00, 1.009

#mean_sd(work_dist)  # 5.7 9.3      fulltime 16.4, 16.9 halftime 10.7, 12. 7 jobless 9 13
#mean_sd(outdoor_dist)  # 5.6, 9             9.3, 12.44         6.6, 9.3             8.9   12
#mean_sd(shopping_dist)  # 4.3, 8.8          5.4, 9.6           4.4, 7.8             3.6  7
 # same result as log_mean_sd


#gen_lnorm(student_work_dist, "")
#gen_lnorm(student_outdoor_dist, "")
#plt.hist(np.random.lognormal (mu, sd, 20 ),     bins = 20)
 

#mu, sd = mean_sd(student_outdoor)

#plt.hist(np.random.normal(mu, sd, 20 ),     bins = 20)
