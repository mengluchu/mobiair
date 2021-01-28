library(data.table)
library(dplyr)
library(ggplot)
library(purrr)
library(tidyr)
a = fread("~/Downloads/Ovin.csv")

#summary(as.numeric(a$VertPC))
#summary(lm(KAf_high~age_lb+Sted, data =a ))
work = a%>%filter(Doel == "Werken")
shopping = a%>%filter(Doel == "Winkelen/boodschappen doen")

#profile 
schoolstu = a%>%filter(MaatsPart=="Scholier/student"&age_lb<18) #22,485/30,218
uni = a%>%filter(MaatsPart=="Scholier/student"&age_lb>18)#22,485/30,218
fulltime = a%>%filter(MaatsPart=="Werkzaam >= 30 uur per week")#22,485/30,218
parttime = a%>%filter(MaatsPart== "Werkzaam 12-30 uur per week" )#22,485/30,218

#purpose
studentshopping = a%>%filter(Doel == "Winkelen/boodschappen doen"& MaatsPart=="Scholier/student")
single = a%>%filter(Doel == "Werken" & MaatsPart=="Eigen huishouding")
student = a%>%filter(Doel == "Werken" & MaatsPart=="Scholier/student")
halftime = a%>%filter(Doel == "Werken" & MaatsPart=="Werkzaam 12-30 uur per week")

# write distance table  
write_dist_prob = function(prof = "Uni_stu", df_prof = uni){
  dist = c(1, 2.5, 3.7, 5, 7.5, 10, 15, 100 )
  
  distprob = function(i , df ,dist){
    
    ss =df%>%
      filter(KAf_mean >0.1&KAf_mean<=dist[i] )%>%
      select(Rvm, KAf_mean)%>%
      group_by(Rvm )%>%
      summarise(n=n())%>%
      mutate(freq = n / sum(n))
    
    train = ss%>%filter(Rvm =="Trein")%>%select(freq)%>%round(digits = 2)
    walk = ss%>%filter(Rvm =="Te voet")%>%select(freq)%>%round(digits = 2)
    bike = ss%>%filter(Rvm =="Fiets (elektrisch en/of niet-elektrisch)")%>%select(freq)%>%round(digits = 2)
    auto = 1-train - walk - bike
    
    abwt = data.frame(auto, bike, walk, train )
    names(abwt) = c("auto", "bike", "walk", "train") 
    
    abwt
  }
  
  dist_prob = data.frame(t(sapply(1:length(dist), distprob, df_prof, dist)))
  #rownames(dist_prob) = paste0("km:", dist)
  aaa=do.call("cbind",dist_prob)
  #rownames(aaa)= NULL
  write.csv(aaa, paste0("~/Documents/GitHub/mobiair/distprob/", prof, ".csv"))
}

write_dist_prob(prof ="Uni_stu", uni )
write_dist_prob(prof ="school_U17", schoolstu )
write_dist_prob(prof ="fulltime", fulltime)
write_dist_prob(prof ="parttime", parttime )

aa1=read.csv("~/Documents/GitHub/mobiair/distprob/Uni_stu.csv")
aa2=read.csv("~/Documents/GitHub/mobiair/distprob/school_U17.csv")
aa3 = read.csv("~/Documents/GitHub/mobiair/distprob/parttime.csv")
aa4=read.csv("~/Documents/GitHub/mobiair/distprob/fulltime.csv" )

aa1-aa2
aa1-aa3
aa3-aa4

dist = c(1, 2.5, 3.7, 5, 7.5, 10, 15, 100 )

distnames = paste0("<", dist, "km")

distnames2 = dist*1000

proce = function(aa1){
  rownames (aa1) = distnames2
  aa1%>%dplyr::select(-X) 
}

aa1 = proce(aa1)
aa2 = proce(aa2)
aa3 =proce(aa3)
aa4 = proce(aa4)

#write.csv(aa1, "~/Documents/GitHub/mobiair/distprob/example_uni_stu.csv" )
#write.csv(aa2, "~/Documents/GitHub/mobiair/distprob/example_school_U17.csv" )
#write.csv(aa3, "~/Documents/GitHub/mobiair/distprob/example_parttime.csv" )
#write.csv(aa4, "~/Documents/GitHub/mobiair/distprob/example_fulltime.csv" )
#write.csv(aa1, "~/Documents/GitHub/mobiair/distprob/illu_uni_stu.csv" )
#write.csv(aa2, "~/Documents/GitHub/mobiair/distprob/illu_school_U17.csv" )
#write.csv(aa3, "~/Documents/GitHub/mobiair/distprob/illu_parttime.csv" )
#write.csv(aa4, "~/Documents/GitHub/mobiair/distprob/illu_fulltime.csv" )

