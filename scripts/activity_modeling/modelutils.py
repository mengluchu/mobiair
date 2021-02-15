#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:12:24 2021

@author: menglu
"""

# Log transform and then get mean and standard deviation (SD) given a dataframe column. 

import pandas as pd
import numpy as np  
import os
import math
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.geometry import LineString

import scipy.stats
import scipy
import shapely.speedups
from shapely.ops import nearest_points 
import networkx as nx
import osmnx as ox
import pyproj
from shapely.ops import transform



def log_mean_sd (df):
 mu =  df.apply(lambda x: x+1). apply(lambda x: np.log(x)).apply(lambda x: np.mean(x))
 sd =  df.apply(lambda  x: x+1). apply(lambda x: np.log(x)).apply(lambda x: np.std(x))
  
 return (*mu, *sd)

# Get the mean and standard deviation (SD) given a dataframe column. 
def mean_sd (df):
 mu =  df.apply(lambda x: np.mean(x))
 sd =  df.apply(lambda x: np.std(x))
  
 return (*mu, *sd)

# Log transform and then get mean and standard deviation (SD) given a dataframe column, using scipy, same as log_mean_sd
def lognormal_mean_sd_scipy(df):
    shape, loc, scale = scipy.stats.lognorm.fit(df.apply(lambda x: x+1), floc=0)
    sd = shape
    mu = np.log(scale)
    return (mu, sd)
 
# generate lognorm distribution, "manual" use log_mean_sd, else use scipy to get mean and sd.     
def gen_lnorm(df, method="manual"):
  if method == "manual":
        mu, sd = log_mean_sd(df)   
  else:      
        mu, sd = lognormal_mean_sd_scipy(df)      
        #mu1, sd1 = log_mean_sd(df)
        #print(mu-mu1,sd-sd1) for checking purpose. same results
  gen = np.random.lognormal (mu, sd, 1)-1
  if gen < 0:
      gen = 0.1 
  return(gen)
 

# given two geopoints, home and work (origin and destination), output an array with [xcoord_home,ycoord_home,xcoord_work,ycoord_work]
def disto2d(homeloc,workloc): # input geopoints. 
    xcoord_work = workloc.centroid.x
    ycoord_work = workloc.centroid.y
    xcoord_home = homeloc.centroid.x
    ycoord_home =homeloc.centroid.y
    p = [xcoord_home,ycoord_home,xcoord_work,ycoord_work]
    return(p)
'''
# input:  a dataframe of lon, lat
# output: geopandas dataframe and a union of it. 
# calculate once so make it a function. 
'''
def pot_dest(goal):
    if type(goal) is pd.core.frame.DataFrame: 
            w_gdf = gpd.GeoDataFrame(crs={'init': 'epsg:4326'},
                                     geometry=[Point(xy) for xy in zip(goal.lon, goal.lat)])
    else: # geopandas
        w_gdf = goal
    u = w_gdf.unary_union
    return (w_gdf, u)


#based on social occupation and travel goal, output work, outdoor, shopping distances, based on the Ovin dataset paprameter name
def distanceOvin(Ovin, socialpartition="Scholier/student", age_from =18, age_to = 50):
    if socialpartition == "Scholier/student":# only consider eductation for students
        work_dist = Ovin.query('Doel == "Onderwijs/cursus volgen" & MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
    else:
        work_dist = Ovin.query('Doel == "Werken" & MaatsPart=="{0}" & {1} <= age_lb <={2}'.format(socialpartition, age_from, age_to))[['KAf_mean']]
    
    outdoor_dist = Ovin.query('Doel == "Sport/hobby" &  MaatsPart=="{0}" & {1} <= age_lb <={2} '.format(socialpartition, age_from, age_to))[['KAf_mean']]
    shopping_dist = Ovin.query('Doel =="Winkelen/boodschappen doen"& MaatsPart=="{0}" & {1} <= age_lb <={2}'.format(socialpartition, age_from, age_to))[['KAf_mean']]
    return(work_dist,outdoor_dist,shopping_dist)

'''
Get destination location point, this one needs the nongeneral distanceOvin function above. If you have the distance (lognormal distributed) as input, 
use the function getdestloc_simple. Note for the destination point selection, only a variable of potential distances is needed (for 
characterising the distribution for simulation). 
 
input: p: home point, w_gdf: destination (e.g. work) geopandas, u, union geopandas, 
       goal: activity for getting the distance, sopa: social participation for getting the distance
output: home point, destinaton point, number of candidate points (to sample a destination point from).
''' 

def getdestloc (p, w_gdf, u, Ovin, goal = "work", sopa = "Scholier/student",  age_from =18, age_to = 50): #for students, work -> eductation
    
    nearestpoint = nearest_points(p,u)[1] #get nearest point
    mindist = p.distance(nearestpoint) #find the distance to the nearest point
  
    work_dist, outdoor_dist, shopping_dist = distanceOvin(Ovin=Ovin, socialpartition = sopa, age_from=age_from, age_to= age_to) #get distance and generate the distribution
    # calculate distance (radius)
    if  goal  == "work": 
        sim_dist = gen_lnorm(work_dist, "")
        sim_distdeg = sim_dist/110.8 #convert to degree
    elif goal == "sport":
        sim_dist = gen_lnorm(outdoor_dist, "")
        sim_distdeg = sim_dist/110.8     
    elif goal == "shopping":
        sim_dist = gen_lnorm(shopping_dist, "")
        sim_distdeg = sim_dist/110.8 
    
    if sim_distdeg < mindist : # if the distance is shorter than the distance to the nearest points. take the nearest point
        sim_distdeg = mindist
        des_p= nearestpoint
        num_points = 0
        print(f'use nearest point as simulated distance is too short.')
    # else calculate a buffer and select points. 
    else:
        pbuf=p.buffer(sim_distdeg) # distance to degree`maybe better to project. 
        pbuf.crs={'init': 'epsg:4326', 'no_defs': True}
        worklocset =w_gdf[w_gdf.within(pbuf)]
        num_points = len(worklocset)
        print(f'sample from {num_points} points')
        workloc =  worklocset.sample(n = 1, replace=True, random_state=1)
        des_p = workloc.iloc[0]["geometry"] #get point out of the geopandas dataframe 
    return (p, des_p,num_points)



'''
input a distance dataframe (single-columned or select a column) instead of using the Ovin for it. otherwise same as getdestloc
'''
def getdestloc_simple (p, w_gdf, u, distvar): #for students, work -> eductation
    
    nearestpoint = nearest_points(p,u)[1] #get nearest point
    mindist = p.distance(nearestpoint) #find the distance to the nearest point
    #print(mindist) degree
    sim_dist = gen_lnorm(distvar, "")
    sim_distdeg = sim_dist/110.8 
    
    if sim_distdeg < mindist : # if the distance is shorter than the distance to the nearest points. take the nearest point
        sim_distdeg = mindist
        des_p= nearestpoint
        num_points = 0
        print(f'use nearest point as simulated distance is too short.')
    # else calculate a buffer and select points. 
    else:
        pbuf=p.buffer(sim_distdeg) # distance to degree`maybe better to project. 
        pbuf.crs={'init': 'epsg:4326', 'no_defs': True}
        worklocset =w_gdf[w_gdf.within(pbuf)]
        num_points = len(worklocset)
        print(f'sample from {num_points} points')
        workloc =  worklocset.sample(n = 1, replace=True, random_state=1)
        des_p = workloc.iloc[0]["geometry"] #get point out of the geopandas dataframe 
    return (p, des_p,num_points)

'''
# main function calculating all the destination points to dataframe. 
input: homedf: dataframe of home (original) locations, names have to be "lat, lon". 
       goal:   dataframe or geopandas dataframe of work (destination) locations. names have to be "lat lon"    
       n: number of fisrt "n" points to calculate, e.g. n = 4, only calculate for the first 4 points. 
       csvname: name/dir for saving the csv file. if the csvname is None, not saving anything. 
       
output: return the dataframe of lat lon of the original and destination locations. home_lon, home_lat, work_lon, work_lat, number of candicate destinations. 
'''

def storedf(homedf, goal, dist_var, des_type = "work", sopa = "Scholier/student", age_from =18, age_to = 50, n=50, csvname = None):    
       
    totalarray = [0,0,0,0,0]
    w_gdf, u  = pot_dest(goal) #get work(destination location)
    for id in range (n): 
        h =homedf.loc[id]
        p=Point(h.lon, h.lat) # distance to degree`maybe better to project. 
        if dist_var.shape[1] is 1:
            op, dp, num_p  = getdestloc_simple(p, w_gdf, u, dist_var)
        else: 
            op, dp, num_p  = getdestloc(p, w_gdf, u, dist_var, des_type, sopa,  age_from = age_from, age_to = age_to)
        parray = disto2d(op,dp)
            
        parray.insert(4,num_p)     
            
            #totalarray = totalarray.append(parray)
            #print(totalarray)
        totalarray = np.concatenate((totalarray, parray), axis=0)
    
    total = np.array (totalarray )
    
    totalre = total.reshape([-1, 5])[1:,:]
    totalre = pd.DataFrame(totalre)
    totalre = totalre.rename (columns = {0:"home_lon", 1: "home_lat", 2:"work_lon", 3:"work_lat", 4: "num_candi"})
    if not csvname is None:
        totalre.to_csv(f'{csvname}.csv')
    return totalre

''' 
example:
filedir = "/Users/menglu/Documents/GitHub/mobiair/"
Ovin = pd.read_csv(filedir+'human_data/dutch_activity/Ovin.csv')
home_csv = filedir+"locationdata/Uhomelatlon.csv"
homedf = pd.read_csv( home_csv) 
Uni= gpd.read_file(filedir+'locationdata/Ut_Uni_coll.gpkg')

simp = Ovin.query('Doel == "Onderwijs/cursus volgen" & MaatsPart=="Scholier/student" & 10<=age_lb<=20')[['KAf_mean']]

storedf(homedf, Uni, n= 5, dist_var=simp, csvname= None) # general function

storedf(homedf, Uni, n= 5, dist_var=Ovin, des_type = "work",sopa = "Scholier/student", csvname= None) # for Ovin
'''

def travelmean_from_distance2work (dis, param = None):
    tra_mode = ['car','bicycle','foot',"train"]
    if param is None:
        if dis <1000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.001, 0.1, 0.899 ,0] )[0]
        elif dis <6000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.05, 0.9, 0.05,0] )[0]
        elif dis <10000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.8, 0.2, 0.000,0] )[0]
        else:
          cls_ = np.random.choice(tra_mode, 1, p = [1, 0, 0,0] )[0]
          
    elif param == "NL" : #parameterise from the Ovin2014 dataset
        if dis <1000:
         cls_ = np.random.choice(tra_mode, 1, p = [0.14, 0.53, 0.33,0] )[0]
        elif dis <2500:
          cls_ = np.random.choice(tra_mode, 1, p = [0.2, 0.6, 0.2,0] )[0]
        elif dis <3700:
          cls_ = np.random.choice(tra_mode, 1, p = [0.3, 0.65, 0.05,0] )[0]
        elif dis < 5000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.4, 0.55, 0.05,0] )[0]
        elif dis <7500:
          cls_ = np.random.choice(tra_mode, 1, p = [0.6, 0.4, 0.00,0] )[0]
        elif dis <10000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.7, 0.3, 0.00,0] )[0]
        elif dis <15000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.9, 0.1, 0,0] )[0]
        else:
          cls_ = np.random.choice(tra_mode, 1, p = [0.9, 0, 0,0.1] )[0] #the travel by by train needed at 0.1
 
    elif param == "NL_student": #parameterise from the Ovin2014 dataset for school child and student, they cycle more and take more public transport        
        if dis <1000:
         cls_ = np.random.choice(tra_mode, 1, p = [0, 0.5, 0.5 ,0] )[0]
        elif dis <2500:
          cls_ = np.random.choice(tra_mode, 1, p = [0.1, 0.7, 0.2 ,0] )[0]
        elif dis <3700:
          cls_ = np.random.choice(tra_mode, 1, p = [0.3, 0.7, 0.0 ,0] )[0]
        elif dis < 5000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.35, 0.65, 0.0 ,0] )[0] # start bus 0,1 bestuurder auto0.1
        elif dis <7500:
          cls_ = np.random.choice(tra_mode, 1, p = [0.45, 0.55, 0.0 ,0] )[0]
        elif dis <10000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.6, 0.4, 0.0 ,0] )[0] # auto include metro 0.1 , bus 0,1
        elif dis <15000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.5, 0.4, 0 ,0.1] )[0] # also include train 
        elif dis < 30000:
          cls_ = np.random.choice(tra_mode, 1, p = [0.7, 0.1, 0 ,0.2] )[0] #electronic or brombike
        else:
          cls_ = np.random.choice(tra_mode, 1, p = [0.3, 0, 0 ,0.7] )[0] #the travel by by train needed at 0.7
 
def travelmean_from_distance2work_df (f_d, dis):
    tra_mode = ['car','bicycle','foot',"train"]
    prob = f_d.iloc[-sum(f_d.iloc[:,0].values > dis),1:].values
    cls_ = np.random.choice(tra_mode, 1, p =prob )[0]
    return cls_   

#for i in range(1, nr_locations):
# cls =['car','bicycle','foot']
#for cls in ['car','bicycle','foot']:
def queryroute(id, cls ='bicycle', writegpkg = True, r_act="h2w" ):          
           xcoord_home = float(homedf.loc[id,"lon"])
           ycoord_home = float(homedf.loc[id,"lat"])
           xcoord_work = float(workdf.loc[id,"lon"])
           ycoord_work = float(workdf.loc[id,"lat"]) 
          # print(xcoord_home,ycoord_home,xcoord_work,ycoord_work)# home and work locations from dataframe
           
           dis, dur = r.distance(ycoord_home,xcoord_home,ycoord_work,xcoord_work, cls)
           if writegpkg:
               r.gpkg(ycoord_home,xcoord_home,ycoord_work,xcoord_work, cls, f'/data/projects/mobiair/routes/{r_act}_{id}_{cls}.gpkg')

           return(dis, dur)

#home and work locations in one df, generated locations
def queryroute_df(id, df, routedir = "routes", cls ='bicycle', writegpkg = True, r_act="h2w" ):          
           xcoord_home = float(df.loc[id,"home_lon"])
           ycoord_home = float(df.loc[id,"home_lat"])
           xcoord_work = float(df.loc[id,"work_lon"])
           ycoord_work = float(df.loc[id,"work_lat"]) 
          # print(xcoord_home,ycoord_home,xcoord_work,ycoord_work)# home and work locations from dataframe
           
           dis, dur = r.distance(ycoord_home,xcoord_home,ycoord_work,xcoord_work, cls)
           if writegpkg:
               if not os.path.exists("/data/projects/mobiair/"+routedir):
                   os.mkdir("/data/projects/mobiair/"+routedir)
               r.gpkg(ycoord_home,xcoord_home,ycoord_work,xcoord_work, cls, f'/data/projects/mobiair/{routedir}/{r_act}_{id}_{cls}.gpkg')

           return(dis, dur)
 


#separating route model and scheduel model 
#wo means work and sports, for one person. input travel duration (time in seconds)
def schedule_general_wo (duration, filedir, name = "work_sport", save_csv = True , time_interval=0.01):

  h2w_mean= 9 # mean time left to work
  w2h_mean = 17
  
  h2w_sd = 1
  w2h_sd = 1

  home2work_start = np.random.normal(h2w_mean,  h2w_sd,  1)[0] 
  work2home_start = np.random.normal(w2h_mean,  w2h_sd, 1 )[0]
  home2work_end = home2work_start+ duration/3600
  work2home_end = work2home_start + duration/3600 # meaning at home again
    
  work_start = home2work_end+time_interval
  work_end=work2home_start-time_interval
  outdoor_evening = work2home_end+1.5
  outdoor_morning = home2work_start-1.5

  start_time = np.round(np.array([0.0, home2work_start, work_start, work2home_start, work2home_end, outdoor_evening, outdoor_evening+1]),2)
     
  end_time = np.round(np.array([*start_time[  1: len(start_time)]-time_interval, 23.9]),2)
  
  activity = ["home", "h2w", "work", "w2h","home", "free_time", "home"]
   
  freetime = np.random.choice([1,4,5,6]) #free time has 4 modes, at home, to sports, 2000m buffer around home, random walk around home

  activity_code = [1 , 2 , 3 , 2, 1, freetime, 1]
  # activity_code: 1:home, 2: work, 3, h2w or w2h, 
    #4, h2sport
  #5, 2000m buffer around home (the person can be anywhere around home), 
  #6, random walk around home  
  
#  activity_type_code = [1, 2, 1, 2, 1, freetime_type, 1]
  # activity_type_code: 1: point, 2: route, 3: buffer, 4: randomwalk 
  data = [start_time,end_time,activity, activity_code]               
   
  schedule = pd.DataFrame(data=data).T
  schedule = schedule.rename (columns = {0:"start_time", 1: "end_time", 2:"activity", 3:"activity_code"})
  if save_csv:
      schedule.to_csv(f'{filedir}/{name}.csv')
  return schedule

def create_graph(loc, dist, transport_mode, loc_type="address"):
   
#    Transport mode = ‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’
     
    if loc_type == "address":
            G = ox.graph_from_address(loc, dist=dist, network_type=transport_mode)
    elif loc_type == "points":
            G = ox.graph_from_point(loc, dist=dist, network_type=transport_mode )
    return G

def getstartend(id, df ):          
           xcoord_home = float(df.loc[id,"home_lon"])
           ycoord_home = float(df.loc[id,"home_lat"])
           xcoord_work = float(df.loc[id,"work_lon"])
           ycoord_work = float(df.loc[id,"work_lat"]) 
           start = (  ycoord_home, xcoord_home)
           end = (ycoord_work, xcoord_work )
           return (start, end)

def qget(id, df ):          
           x1= float(df.loc[id,"home_lon"])
           y1 = float(df.loc[id,"home_lat"])
           x2 = float(df.loc[id,"work_lon"])
           y2 = float(df.loc[id,"work_lat"]) 
           return y1,x1,y2,x2


def nodes_to_linestring(route, G):
    coords_list = [(G.nodes[i]['x'], G.nodes[i]['y']) for i in route ]
    line = LineString(coords_list)
    return(line)

#geom = ogr.CreateGeometryFromWkt(route_.wkt)
#I tried many ways but the easiest is to save to gpd. 
#not allowing one shapely but more is ok...

# note for car we use routes of min travel time, but for others we use shortest distance routes.     
# for walking and biking we have a speed. this is good because we can do it differently for children and adults. 
# OSM default speed: https://www.targomo.com/developers/resources/concepts/assumptions_and_defaults/
# OSM default footway: 5km/h , speed_bike 15km/h    
 
def getroute(id, df, f_d, Gw, Gb, Gd, speed_walk = 5, speed_bike = 15):
    start, end = getstartend(id, df=df) 
    
    apprEucl = Point(start).distance(Point(end))*110    #approximate Euclidean distance km     
    if apprEucl <0.001: #less than 1 m, jitter 100 m 
        end = (end[0], end[1]+0.001) 
    cls_ = travelmean_from_distance2work_df( f_d,apprEucl*1000)
    print(apprEucl,  cls_)
    if cls_ == "train":     
        cls_ = "car"
    
    if cls_ == "foot":
        G=Gw
        speed = speed_walk
    elif cls_ =="bicycle":
        G=Gb
        speed = speed_bike
    else:
        G=Gd
        speed =100 # for train
    start_node = ox.get_nearest_node(G, start,"haversine")
    end_node = ox.get_nearest_node(G, end , "haversine")
    
    if start_node ==end_node:
       travel_distance = 0 
       travel_time = 0
       route_ = Point(start)
    # Calculate the shortest path
    # nx and ox are the same: route1 = nx.shortest_path(G, start_node, end_node, weight='travel_time')
   
    #traveldistance is calculated using the route of min length. so not exactly the same as travel time.
    else:
        travel_distance = nx.shortest_path_length(G, start_node,end_node, weight='length')
        if cls_ == "car":
            travel_time = nx.shortest_path_length(G, start_node,end_node, weight='travel_time')
            route = ox.distance.shortest_path(G, start_node, end_node, weight='travel_time')
        else:
            travel_time = travel_distance /1000/ speed*3600 
            route = ox.distance.shortest_path(G, start_node, end_node, weight='length')
        route_ = nodes_to_linestring(route,G) #Method defined above
    
    return (route_, travel_time, travel_distance,cls_)
   
    #now complete book keeping, later maybe better to have an id
def roundlist (list_, n=2):
    return list(np.round(np.array(list_), n)) 

def makefolder (dir_):
    if not os.path.exists(dir_):
                   os.mkdir(dir_)
                   
def wgs2laea (p):
    wgs84 = pyproj.CRS('EPSG:4326')
    rd= pyproj.CRS('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    project = pyproj.Transformer.from_crs(wgs84, rd, always_xy=True)
    p=transform(project.transform, p)
    return (p)                 