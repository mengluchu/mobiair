#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:21:18 2021

@author: menglu
"""
from mobiair import Routing
import pandas as pd 
import numpy as np
import os
#import random
filedir = "/data/projects/mobiair"
#home_csv = filedir+"/locationdata/Uhomelatlon.csv"
#work_csv = filedir+"/locationdata/Uworklatlon.csv"  #working locations of each homeID. Will later group by homeID for sampling
#homedf = pd.read_csv( home_csv) 
#workdf = pd.read_csv( work_csv)  #for randomly sample working locations
#nr_locations = homedf.shape[0]

#f_d = pd.read_csv( "/Users/menglu/Documents/GitHub/mobiair/distprob/example_fulltime.csv")        

f_d = pd.read_csv( "/data/projects/mobiair/distprob/example_uni_stu.csv")        
r = Routing(server='127.0.0.1', port_car=55001, port_bicycle=55002, port_foot=55003, port_train=55004)


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
    elif param == "fulltime": # reading from the table is the most convenient way, a more cubersome way see "activity_model"
          prob = f_d.iloc[-sum(f_d.iloc[:,0].values > dis),1:].values
          cls_ = np.random.choice(tra_mode, 1, p =prob )[0]
    return cls_

#get prob. from a table
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

#home and work locations in one df, get distance, duration. 
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
 
#id =2  #student id 
#1 from this i will get work and home location, then calculate route_duration. 
#route_duration =np.random.normal(0.3,0.1,1) #18 min #query from OSM 
#cls = random.sample(['car','bicycle','foot'],1) better based on distance
 

# df: home and work locations routedir: gpkg routes, mode4dis, which mode, with the travelmean_..._df only need to input dataframe

def generate_stu_eve (id, df, routedir="routes_real", mode4dis=f_d,   save_csv = True , time_interval=0.01):
  
  dis, duration= queryroute_df(id, df, routedir, cls = "bicycle", writegpkg = False) 
  if type(mode4dis) is str: 
  #netherlands, default -- the parameters will be calculated from survey data. 
      cls_ = travelmean_from_distance2work(dis, param=mode4dis)
  else: 
      cls_ = travelmean_from_distance2work_df(mode4dis, dis)
 
  dis, duration  = queryroute_df(id, df, routedir,  cls = cls_)  
  
  name= "Stu_Eve"+str(id)
  h2w_mean= 9 # mean time left to work
  w2h_mean = 17
  
  h2w_sd = 1
  w2h_sd = 1
  print("distrance:",dis,"duration(min):",int(duration/60)) 
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
  
  activity = ["home", "h2w_"+cls_, "work", "w2h_"+cls_,"home", "outdoor", "home"]
  data = [start_time,end_time,activity]               
   
  schedule = pd.DataFrame(data=data).T
  schedule = schedule.rename (columns = {0:"start_time", 1: "end_time", 2:"activity"})
  if save_csv:
      schedule.to_csv(filedir+"/act_schedule/"+ name+"p.csv")
  return schedule


 
#name = "Stu_Eve_o_tm"+str(id)  # o_tm means ovin travelmode
#df = pd.read_csv(filedir+"/genloc/ut_h2Uni.csv")
for j in range(2,5):
    df = pd.read_csv(f'{filedir}/genloc/ut_Uni{j}.csv')
    #df = pd.read_csv(filedir+"/locationdata/Uni_Ut_homework.csv")
    for id in range(df.shape[0]):   
        schedule = generate_stu_eve(id, df, f'route_{j}', f_d ,save_csv=True)
        print(id)
        print(schedule)   

#with open('/data/lu01/aamyfile.geojson', 'w') as f:
#    drump(route,f)