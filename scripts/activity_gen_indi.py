#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:44:36 2021

@author: menglu
"""
import numpy as np
import pandas as pd
id =2  #student id 
#1 from this i will get work and home location, then calculate route_duration. 
route_duration =np.random.normal(0.3,0.1,1) #18 min #query from OSM 
time_interval = 0.01
#n = 1 #seed
#set.seed(n)
name = "Stu_Eve_"+str(id)
duration = route_duration
def generate_stu_eve (id,   duration = route_duration, save_csv = True ):

  name= "Stu_Eve"+str(id)
  h2w_mean= 9 # mean time left to work
  w2h_mean = 17
  
  h2w_sd = 1
  w2h_sd = 1
   
  home2work_start = np.random.normal(h2w_mean,  h2w_sd,  1)[0] 
  work2home_start = np.random.normal(w2h_mean,  w2h_sd, 1 )[0]
  home2work_end = home2work_start+ duration[0]
  work2home_end = work2home_start + duration[0]
    
  work_start = home2work_end+time_interval
  work_end=work2home_start-time_interval
  outdoor_evening = work2home_start+2
  outdoor_morning = home2work_start-2

  start_time = np.round(np.array([0.0, home2work_start, work_start, work2home_start, outdoor_evening, outdoor_evening+1]),2)
     
  end_time = np.round(np.array([*start_time[  1: len(start_time)]-time_interval, 23.9]),2)
  
  activity = ["home", "h2w", "work", "w2h", "outdoor", "home"]
  data = [start_time,end_time,activity]               
   
  schedule = pd.DataFrame(data=data).T
  schedule = schedule.rename (columns = {0:"start_time", 1: "end_time", 2:"activity"})
  if save_csv:
      schedule.to_csv("~/Documents/GitHub/mobiair/activity/indi/"+ name+"p.csv")
  return schedule

for id in range(10,20):   
    schedule = generate_stu_eve(id, save_csv=False)
    print(schedule)