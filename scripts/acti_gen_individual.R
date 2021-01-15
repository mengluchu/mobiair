

#students, do sports in the evening

id =2  #student id 
#1 from this i will get work and home location, then calculate route_duration. 
route_duration = rnorm(1,0.3,0.1) #18 min #query from OSM 
time_interval = 0.01
#n = 1 #seed
#set.seed(n)
name = paste("Stu_Eve",id, sep = "_")
generate_stu_eve = function(id,   duration = route_duration )
{
  name=paste("Stu_Eve",id, sep = "_")
  h2w_mean= 9 # mean time left to work
  w2h_mean = 17
  
  h2w_sd = 1
  w2h_sd = 1
  
  home2work_start = rnorm(n = 1, mean = h2w_mean, sd = h2w_sd) 
  work2home_start = rnorm(n= 1, mean = w2h_mean, sd = w2h_sd)
  home2work_end = home2work_start+duration
  work2home_end = work2home_start + duration
  
  work_start = home2work_end+time_interval
  work_end=work2home_start-time_interval
  outdoor_evening = work2home_start+2
  outdoor_morning = home2work_start-2
  
   start_time = round(c(0.0, home2work_start, work_start, work2home_start, outdoor_evening, outdoor_evening+1), digits = 2)
   end_time = round(c(start_time[2: length(start_time)]-time_interval, 23.9) ,digits = 2)               
   activity = c("home", "h2w", "work", "w2h", "outdoor", "home")
                      
   schedule = data.frame (start_time, end_time, activity)
   write.csv(schedule, paste0("~/Documents/GitHub/mobiair/activity/indi/", name,".csv"))
   }
read.csv(paste0("~/Documents/GitHub/mobiair/activity/indi/", name,".csv"))
# student, do sports in the morning
names_ = paste("Stu_Eve",1:10, sep = "_")
lapply(1:10, generate_stu_eve)