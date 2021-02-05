library(data.table)
library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)
a = fread("~/Documents/GitHub/mobiair/human_data/dutch_activity/Ovin.csv")

Uni_work = a%>%filter(  MaatsPart=="Scholier/student"&age_lb>=18 &VertProv == "Utrecht"&Doel == "Onderwijs/cursus volgen")%>%select(OPID, Doel, arr_lon, arr_lat, dep_lon, dep_lat)
Uni_home = a%>%filter(  MaatsPart=="Scholier/student"&age_lb>=18 &VertProv == "Utrecht"& Doel == "Naar huis" )%>%select(OPID, Doel, arr_lon, arr_lat, dep_lon, dep_lat)
workhome = merge(Uni_home, Uni_work, by = "OPID", all = T)%>%na.omit()%>%unique()# 49 unique home and work location

# they are not the same people. They DO NOT depart from home to do sport. When using our model we do have to assume they do depart from home.
# it is acutally interesting that we can also make our model according to where they depart.

# work is for going to school, education
Unihome = workhome%>%select(OPID,arr_lon.x, arr_lat.x)%>%rename("lon" = arr_lon.x, "lat"= arr_lat.x) %>%apply(2,nafill,type = "locf" )
Uniwork = workhome%>%select(OPID,arr_lon.y, arr_lat.y)%>%rename("lon" = arr_lon.y, "lat"= arr_lat.y) %>%apply(2,nafill,type = "locf" )

Unihw = workhome%>%select(OPID,arr_lon.x, arr_lat.x,arr_lon.y, arr_lat.y)%>%rename("home_lon" = arr_lon.x, "home_lat"= arr_lat.x,"work_lon" = arr_lon.y, "work_lat"= arr_lat.y) %>%apply(2,nafill,type = "locf" )


write.csv(Unihome, "~/Documents/GitHub/mobiair/locationdata/Uni_Ut_home.csv")
write.csv(Uniwork, "~/Documents/GitHub/mobiair/locationdata/Uni_Ut_work.csv")
write.csv(Unihw, "~/Documents/GitHub/mobiair/locationdata/Uni_Ut_homework.csv")

# these two datasets are for testing real vs. simulated. For univeristy students.

# some people go to different work locaitons. so 36 unique id, but we use all of them for study, assuming these people are different.
#Uni2work = workhome%>%select(OPID,dep_lon.y, dep_lat.y)%>%rename("lon" = dep_lon.y, "lat"= dep_lat.y)%>%apply(2,nafill,type = "locf" )
#(Uni2work-Unihome)%>%table #most people go from home to work...
#Uni_homework= Uni_homework%>%complete(OPID, nesting(Doel))%>%data.frame()%>%unique()
#Uni_homework[-c(76,79, 228, 229),]%>%spread( Doel, OPID )
#Uni_sport = a%>%filter(  MaatsPart=="Scholier/student"&age_lb>=18 &VertProv == "Utrecht" & Doel =="Sport/hobby")



Uni_home$OPID%>%table() # some people go home 7 times...
Uni_work$OPID%>%length() # 93/47, somepeople go to cursus 4 times.
Uni_work$OPID%>%unique()%>%length() # 47

Uni_work$OPID[Uni_sport$OPID] # completely different people doing work and spaorts, and 17 unique out of 25 people going to work


Uni_w_home = Uni_work%>%select(dep_lon, dep_lat)%>%rename("lon" = dep_lon, "lat"= dep_lat) %>%apply(2,nafill,type = "locf" )
Uni_s_home = Uni_sport%>%select(dep_lon, dep_lat)%>%rename("lon" = dep_lon, "lat"= dep_lat)%>%apply(2,nafill,type = "locf" )
# fill na with the last observation carried forward
Uni_w_work = Uni_work%>%select(arr_lon, arr_lat)%>%rename("lon" = arr_lon, "lat"= arr_lat)%>%apply(2,nafill,type = "locf" )
sum(Uni_w_work == Uni_s_home) # but also not depart from school... besides 2

Uni_w_work = Uni_work%>%select(dep_lon, dep_lat)%>%rename("lon" = dep_lon, "lat"= dep_lat)



U17student = a%>%filter(  MaatsPart=="Scholier/student"&age_lb<18)
halftime = a%>%filter( MaatsPart=="Werkzaam 12-30 uur per week")
nrow(Unistudent)
nrow(U17student)
Unistudent$VertPC

 U17student$VertPC%>%length()
a$MaatsPart)
