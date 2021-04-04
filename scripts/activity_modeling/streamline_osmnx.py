#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:02:06 2021

@author: menglu
"""
import os
scriptdir = "/Users/menglu/Documents/GitHub/mobiair/scripts/activity_modeling"
os.chdir(scriptdir)
import modelutils as m
import pandas as pd 
import geopandas as gpd
import numpy as np
import scipy.stats

import plotly_express as px
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.geometry import LineString

from osgeo import ogr, osr

#G = ox.project_graph(G)

#apprEucl =ox.distance.euclidean_dist_vec(*qget(1, df)) # if not using Euclidean distance 
filedir = "/Users/menglu/Documents/GitHub/mobiair/"
savedir = "/Volumes/Meng_Mac/mobi_result/Uni/" # each profile a savedir. 
savedir2 = "/Volumes/Meng_Mac/mobi_result/Uni_real/" # esavedir2 for comparison. 


Gw= ox.io.load_graphml(filepath=f'{filedir}osmnx_graph/ut10kwalk.graphml')
Gb= ox.io.load_graphml(filepath=f'{filedir}osmnx_graph/ut10bike.graphml') # note: add speed only works for freeflow travel speeds, osm highway speed
Gd= ox.io.load_graphml(filepath=f'{filedir}osmnx_graph/ut10drive.graphml')

home_csv = f'{filedir}locationdata/Uhomelatlon.csv"
homedf =pd.read_csv(home_csv)
Ovin = pd.read_csv(f'{filedir}human_data/dutch_activity/Ovin.csv')

#University students eductation locations 
Uni= gpd.read_file(filedir+'locationdata/Ut_Uni_coll.gpkg') 

# travel probability for university students
f_d = pd.read_csv( f"{filedir}/distprob/example_uni_stu.csv")  

#generate destination locations

Uni_ut_home = pd.read_csv(filedir +"locationdata/Uni_Ut_home.csv") # home locations for generating activities. (with known Uni) 
Uni_ut_homework = pd.read_csv(filedir +"locationdata/Uni_Ut_homework.csv") # for comparison 

nr_locations = Uni_ut_home.shape[0]
nr_locations
#start

ite = 1 # first iteration 
#homedf.shape[0]
#Uni_ut_home, Uni, n= nr_locations, dist_var=Ovin, des_type = "work",sopa = "Scholier/student", age_from = 18, age_to=99 ,csvname= f'{savedir}genloc/h2w_{ite}'
#real_od: no simulation, origin and destination provided, as a dataframe, home_lon, home_lat, work_lon, work_lat  

def generate_activity(Gw, Gb, Gd, f_d, savedir, real_od =None, ori=None, des=None, n=10, dist_var=None, des_type=None, sopa=None, age_from=None, age_to=None):

    m.makefolder(f"{savedir}genloc") # make folder only if the dir doesnt exist. 
    m.makefolder(f"{savedir}genroute")
    m.makefolder(f"{savedir}genroute_act2")
    m.makefolder(f"{savedir}genloc_act2")
    m.makefolder(f"{savedir}gensche")
        
    csvname= f'{savedir}genloc/h2w_{ite}'
    
    #1. generate OD locations
    if real_od is None:     
        df = m.genODdf(homedf = ori, goal = des,  dist_var=dist_var, des_type= des_type, sopa =sopa, 
                   age_from = age_from, age_to=age_to, n = n,  csvname= csvname) 
    else:
        df = real_od
    if savedir is not None: 
        df.to_csv(f'{savedir2}genloc/h2w_real')
        
    allroutes= []
    alltravel_time= []
    alltravel_distance= []
    alltravel_mean = []
    
    #2. generate activity schedules  
    for id in range(n):
     
        route, travel_time, travel_distance, travel_mean = m.getroute(id, Gw=Gw, Gb=Gb, Gd=Gd, df=df, f_d=f_d)       
        m.schedule_general_wo(travel_time, savedir+"gensche", name = f"ws_iter_{ite}_id_{id}") #ws: work and sport
        # generated activities and save the tables 
        allroutes.append(route)
        alltravel_time.append(travel_time)
        alltravel_mean.append(travel_mean)
        alltravel_distance.append(travel_distance)    
     
    #3. generate OD routes.  
    d = {'duration_s': m.roundlist(alltravel_time), 'distance_m': m.roundlist(alltravel_distance), 'travel_mean': alltravel_mean, 'geometry': allroutes}     
    gpd1= gpd.GeoDataFrame(d, crs={'init': 'epsg:4326'}) 
    gpd1.to_file(f'{savedir}genroute/route_{ite}.gpkg')

# generated activities and get the routes
generate_activity(ori = Uni_ut_home, des = Uni, n= nr_locations, 
                  dist_var=Ovin, des_type = "work",sopa = "Scholier/student", 
                  age_from = 18, age_to=99, Gw= Gw, Gb= Gb, Gd= Gd,  f_d = f_d, 
                  savedir = savedir)  

 # generated activiites using real work locaitons 
generate_activity(real_od = Uni_ut_homework,n= nr_locations, Gw= Gw, Gb= Gb, Gd= Gd, f_d = f_d,savedir = savedir2)  
imp.reload(m) 
#Point(start).distance(Point(end))*110 

 
''' alternative ways of saving routes
import fiona
from shapely.geometry import mapping
schema = {
    'geometry': 'LineString',
    'properties': {'id': 'int'},
}
with fiona.open(filedir+f'route{id}.shp', 'w','ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(route_),
        'properties': {'id': 123},
    })


Json = geom.ExportToJson()

import json

with open(filedir+'route.json', 'w') as json_file:
    json.dump(Json, json_file)
 '''
#nodes ,edge = ox.graph_to_gdfs(G) 

#get routes as lines

# from Gboeing but doesnt work
#from shapely.geometry import MultiLineString
#route_pairwise = zip(route[:-1], route[1:])
#edges = ox.graph_to_gdfs(G, nodes=False).set_index(['u', 'v']).sort_index()
#lines = [edges.loc[uv, 'geometry'].iloc[0] for uv in route_pairwise]
#MultiLineString(lines)


#G = create_graph('Utrecht', 5000, cls_)# walk
#ox.plot_graph(G)
#G = ox.add_edge_speeds(G) #Impute but only for "car"
#G = ox.add_edge_travel_times(G) #Travel time
#ox.io.save_graphml(G, filepath=filedir+"utgraph.graphml")
#edge doesnt work?
#ox.io.save_graph_geopackage(G, filepath=filedir+"utgraph.gpkg", encoding='utf-8', directed=False)
#nodes ,edge = ox.utils_graph.graph_to_gdfs(G) 
#nodes.to_file ("nodes.gpkg")
#edge.to_file("edge.gpkg")

#ox.io.save_graph_geopackage(G, filepath=filedir+"utgraph.gpkg", encoding='utf-8', directed=False)
#G1 = gpd.read_file(filedir+"utgraph.gpkg")
#not easy to load because of the need of edge and nodes 


#same: np.array(route1)-np.array(route)
#Plot the route and street networks

#fig, ax = ox.plot_graph_route(Gb, route, route_linewidth=6, node_size=0, bgcolor='k' )
#fig.savefig("route.png")


#ox.config(use_cache=True, log_console=True)

#import random
#filedir = "/data/projects/mobiair"
#home_csv = filedir+"/locationdata/Uhomelatlon.csv"
#work_csv = filedir+"/locationdata/Uworklatlon.csv"  #working locations of each homeID. Will later group by homeID for sampling
#homedf = pd.read_csv( home_csv) 
#workdf = pd.read_csv( work_csv)  #for randomly sample working locations
#nr_locations = homedf.shape[0]

#f_d = pd.read_csv( "/Users/menglu/Documents/GitHub/mobiair/distprob/example_fulltime.csv")        
