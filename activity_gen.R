

#students, do sports in the evening
generate_stu_eve = function(name="Stu_Eve", num_sim = 15)
{
  h2w_mean= 9 # mean time left to work
  w2h_mean = 17
 
  h2w_sd = 1
  w2h_sd = 1
 
  home2work =  round(rnorm(n = num_sim, mean = h2w_mean, sd = h2w_sd ))
  work2home = round(rnorm(n=num_sim, mean = w2h_mean, sd = w2h_sd  ))
  outdoor_evening = work2home+2
  outdoor_morning = home2work-2
  schedule = array(dim =c(24,num_sim))
  for (N in c(1:num_sim))
  {schedule [home2work[N],N] = "h2w"
  schedule [work2home[N],N] = "w2h"
  schedule [outdoor_evening[N],N] = "outdoor"
 
  athome= c(outdoor_evening[N]-1, (outdoor_evening[N]+1):24, 1:(home2work[N]-1)  )
  atwork= c((home2work[N]+1 ):(work2home[N]-1))
  schedule[atwork,N]="work"
  schedule[athome,N]="home"}
 
  df = data.frame(schedule)
  names(df)= paste(name, 1:num_sim, sep = "_")
  write.csv(df, paste0("~/Documents/GitHub/mobiair/activity/", name,".csv"))}
  
# student, do sports in the morning

generate_stu_mor = function(name="Stu_Mor", num_sim = 15)
{
  h2w_mean= 9 # mean time left to work
  w2h_mean = 17
  
  h2w_sd = 1
  w2h_sd = 1
  
  home2work =  round(rnorm(n = num_sim, mean = h2w_mean, sd = h2w_sd ))
  work2home = round(rnorm(n=num_sim, mean = w2h_mean, sd = w2h_sd  ))
  outdoor_evening = work2home+2
  outdoor_morning = home2work-2
  schedule = array(dim =c(24,num_sim))
  for (N in c(1:num_sim))
  {schedule [home2work[N],N] = "h2w"
  schedule [work2home[N],N] = "w2h"
  schedule [outdoor_morning[N],N] = "outdoor"
  
  athome= c((outdoor_morning[N]+1):(home2work[N]-1), 1:(outdoor_morning[N]-1), (work2home[N]+1):24 )
  atwork= c((home2work[N]+1 ):(work2home[N]-1))
  schedule[atwork,N]="work"
  schedule[athome,N]="home"}
  
  df = data.frame(schedule)
  names(df)= paste(name, 1:num_sim, sep = "_")
  write.csv(df, paste0("~/Documents/GitHub/mobiair/activity/", name,".csv"))}

generate_stu_mor()

  read.csv("~/Documents/GitHub/mobiair/activity/Stu_Eve.csv")

#full-time workers: only differences are: (1) go to work earlier, at 8, (2) the more fixed time schedule, i.e. smaller sd. (3) do sports in the evening 
generate_fulltime_workers = function(name="worker_fulltime", num_sim = 6)
  {
    h2w_mean= 8 # mean time left to work
    w2h_mean = 17
    
    h2w_sd = 0.4
    w2h_sd = 0.4
    
    home2work =  round(rnorm(n = num_sim, mean = h2w_mean, sd = h2w_sd ))
    work2home = round(rnorm(n=num_sim, mean = w2h_mean, sd = w2h_sd  ))
    outdoor_evening = work2home+2
    schedule = array(dim =c(24,num_sim))
    for (N in c(1:num_sim))
    {schedule [home2work[N],N] = "h2w"
    schedule [work2home[N],N] = "w2h"
    schedule [outdoor_evening[N],N] = "outdoor"
    
    athome= c(outdoor_evening[N]-1, (outdoor_evening[N]+1):24, 1:(home2work[N]-1)  )
    atwork= c((home2work[N]+1 ):(work2home[N]-1))
    schedule[atwork,N]="work"
    schedule[athome,N]="home"}
    
    df = data.frame(schedule)
    names(df)= paste(name, 1:num_sim, sep = "_")
    write.csv(df, paste0("~/Documents/GitHub/mobiair/activity/", name,".csv"))}

    generate_fulltime_workers()
  
  read.csv("~/Documents/GitHub/mobiair/activity/worker_fulltime.csv")
  
  
  generate_stu_mor = function(name="Stu_Mor", num_sim = 15)
  {
    h2w_mean= 9 # mean time left to work
    w2h_mean = 17
    
    h2w_sd = 1
    w2h_sd = 1
    
    home2work =  round(rnorm(n = num_sim, mean = h2w_mean, sd = h2w_sd ))
    work2home = round(rnorm(n=num_sim, mean = w2h_mean, sd = w2h_sd  ))
    outdoor_evening = work2home+2
    outdoor_morning = home2work-2
    schedule = array(dim =c(24,num_sim))
    for (N in c(1:num_sim))
    {schedule [home2work[N],N] = "h2w"
    schedule [work2home[N],N] = "w2h"
    schedule [outdoor_morning[N],N] = "outdoor"
    
    athome= c((outdoor_morning[N]+1):(home2work[N]-1), 1:(outdoor_morning[N]-1), (work2home[N]+1):24 )
    atwork= c((home2work[N]+1 ):(work2home[N]-1))
    schedule[atwork,N]="work"
    schedule[athome,N]="home"}
    
    df = data.frame(schedule)
    names(df)= paste(name, 1:num_sim, sep = "_")
    write.csv(df, paste0("~/Documents/GitHub/mobiair/activity/", name,".csv"))}
  
  generate_stu_mor()
  
  read.csv("~/Documents/GitHub/mobiair/activity/Stu_Eve.csv")
  
  #full-time workers: only differences are: (1) go to work earlier, at 8, (2) the more fixed time schedule, i.e. smaller sd. (3) do sports in the evening 
  generate_homemakers = function(name="homemakers", num_sim = 15)
  {
     
    schedule = array("home", dim =c(24,num_sim))
    for (N in c(1:num_sim))
      
    {
      schedule [sample(size =1, x =7:22), N] = "outdoor"
    schedule [sample(8:21,size =1),N] = "shopping"
    
    schedule [sample(8:21,size =2),N] = "home"  #(2/13 not shopping)
    schedule [sample(7:22,size =2),N] = "home"#(2/15 not outdoor)
   }
    
    df = data.frame(schedule)
    names(df)= paste(name, 1:num_sim, sep = "_")
    write.csv(df, paste0("~/Documents/GitHub/mobiair/activity/", name,".csv"))}
  generate_homemakers()
  read.csv("~/Documents/GitHub/mobiair/activity/homemakers.csv")
  
  