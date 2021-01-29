#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:21:18 2021

@author: menglu
"""
from mobiair import Routing
import pandas as pd 
import numpy as np
#from geojson import dump
#import random
filedir = "/data/lu01/"
home_csv = filedir+"Uhomelatlon.csv"
work_csv = filedir+"Uworklatlon.csv"  #working locations of each homeID. Will later group by homeID for sampling
homedf = pd.read_csv( home_csv) 
workdf = pd.read_csv( work_csv)  #for randomly sample working locations
nr_locations = homedf.shape[0]

f_d = pd.read_csv( "/Users/menglu/Documents/GitHub/mobiair/distprob/example_fulltime.csv")        

r = Routing(server='127.0.0.1', port_car=55001, port_bicycle=55002, port_foot=55003, port_train=55004)



# Distance
dist = 7700
# Query values from the index
prob = f_d.iloc[-sum(f_d.iloc[:,0].values > dist),1:].values
# Print values
print(prob)


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
          cls_ = np.random.choice(tra_mode, 1, p = [0.7, 0.1, 0 ,02] )[0] #electronic or brombike
        else:
          cls_ = np.random.choice(tra_mode, 1, p = [0.3, 0, 0 ,0.7] )[0] #the travel by by train needed at 0.7
    elif param == "fulltime":
        a = fulltime.iloc[:,1:5].values.astype(float)
        if dis <1000:
         cls_ = np.random.choice(tra_mode, 1, p = a[0])[0]
        elif dis <2500:
          cls_ = np.random.choice(tra_mode, 1, p = a[1])[0]
        elif dis <3700:
          cls_ = np.random.choice(tra_mode, 1, p = a[2])[0]
        elif dis < 5000:
          cls_ = np.random.choice(tra_mode, 1, p =a[3])[0]
        elif dis <7500:
          cls_ = np.random.choice(tra_mode, 1, p = a[4])[0]
        elif dis <10000:
          cls_ = np.random.choice(tra_mode, 1, p = a[5])[0]
        elif dis <15000:
          cls_ = np.random.choice(tra_mode, 1, p = a[6] )[0]
        else:
          cls_ = np.random.choice(tra_mode, 1, p = a[7])[0] #the travel by by train needed at 0.1
  
    return cls_
        

#for i in range(1, nr_locations):
# cls =['car','bicycle','foot']
#for cls in ['car','bicycle','foot']:
def queryroute(id, cls ='bicycle'):          
           xcoord_home = float(homedf.loc[id,"lon"])
           ycoord_home = float(homedf.loc[id,"lat"])
           xcoord_work = float(workdf.loc[id,"lon"])
           ycoord_work = float(workdf.loc[id,"lat"]) 
          # print(xcoord_home,ycoord_home,xcoord_work,ycoord_work)# home and work locations from dataframe
           
           dis, dur = r.distance(ycoord_home,xcoord_home,ycoord_work,xcoord_work, cls)
           return(dis, dur)


id =2  #student id 
#1 from this i will get work and home location, then calculate route_duration. 
#route_duration =np.random.normal(0.3,0.1,1) #18 min #query from OSM 
#cls = random.sample(['car','bicycle','foot'],1) better based on distance
 
time_interval = 0.01
#n = 1 #seed
#set.seed(n)


name = "Stu_Eve_o_tm"+str(id)  # o_tm means ovin travelmode

def generate_stu_eve (id,  mode4dis=None,  save_csv = True):
  
  dis, duration, route= queryroute(id, cls = "bicycle") 
  
  #netherlands, default -- the parameters will be calculated from survey data. 
  cls_ = travelmean_from_distance2work(dis, param=mode4dis)
  
  dis, duration = queryroute(id, cls = cls_)  
  r.gpkg(47.5596321,7.5883672,46.9479288,7.4481001, cls_, f'route{id}_{cls}.gpkg')

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
      schedule.to_csv("/data/lu01/mobiair_model/act_schedule/"+ name+"p.csv")
  return schedule, route

for id in range(10,20):   
    schedule, route = generate_stu_eve(id, "NL_student",save_csv=True)
    print(id)
    print(schedule)   

#with open('/data/lu01/aamyfile.geojson', 'w') as f:
#    drump(route,f)