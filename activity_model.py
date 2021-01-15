#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:21:18 2021

@author: menglu
"""
from mobiair import Routing
import pandas as pd 
import numpy as np
#import random
filedir = "/data/lu01/"
home_csv = filedir+"Uhomelatlon.csv"
work_csv = filedir+"Uworklatlon.csv"  #working locations of each homeID. Will later group by homeID for sampling
homedf = pd.read_csv( home_csv) 
workdf = pd.read_csv( work_csv)  #for randomly sample working locations
nr_locations = homedf.shape[0]
        
r = Routing(server='127.0.0.1', port_car=5001, port_bicycle=5002, port_foot=5003)

        
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
name = "Stu_Eve_"+str(id)

def generate_stu_eve (id,    save_csv = True ):
  tra_mode = ['car','bicycle','foot']
  dis, duration = queryroute(id, cls = "bicycle") 
  
  #netherlands, default -- the parameters will be calculated from survey data. 
  
  if dis <1000:
      cls_ = np.random.choice(tra_mode, 1, p = [0.001, 0.1, 0.899] )[0]
  elif dis <6000:
      cls_ = np.random.choice(tra_mode, 1, p = [0.05, 0.9, 0.05] )[0]
  elif dis <10000:
      cls_ = np.random.choice(tra_mode, 1, p = [0.8, 0.2, 0.000] )[0]
  else:
      cls_ = np.random.choice(tra_mode, 1, p = [1, 0, 0] )[0]

  dis, duration = queryroute(id, cls = cls_)  
  
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
  return schedule

for id in range(10,20):   
    schedule = generate_stu_eve(id, save_csv=True)
    print(id)
    print(schedule)   
    