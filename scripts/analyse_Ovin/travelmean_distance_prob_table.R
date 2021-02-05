library(data.table)
library(dplyr)
library(ggplot)
library(purrr)
library(tidyr)
a = fread("~/Documents/GitHub/mobiair/human_data/dutch_activity/Ovin.csv")

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
student = a%>%filter(Doel == "Onderwijs/cursus volgen" & MaatsPart=="Scholier/student"&age_lb>18)
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
    train = ifelse( nrow(train) == 0, 0, train[[1]]) 
    walk = ss%>%filter(Rvm =="Te voet")%>%select(freq)%>%round(digits = 2)
    walk = ifelse( nrow(walk) == 0, 0, walk[[1]]) 
    bike = ss%>%filter(Rvm =="Fiets (elektrisch en/of niet-elektrisch)")%>%select(freq)%>%round(digits = 2)
    bike = ifelse( nrow(bike) == 0, 0, bike[[1]]) 
    auto = 1-train - walk  - bike 

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
# didnt separate purpose.
write_dist_prob(prof ="Uni_stu", uni )
write_dist_prob(prof ="Uni_stu_uni", student)

write_dist_prob(prof ="hal_work", halftime)
write_dist_prob(prof ="school_U17", schoolstu )
write_dist_prob(prof ="fulltime", fulltime)
write_dist_prob(prof ="parttime", parttime )
aa0=read.csv("~/Documents/GitHub/mobiair/distprob/Uni_stu.csv")
aa1=read.csv("~/Documents/GitHub/mobiair/distprob/hal_work.csv")
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

write.csv(aa1, "~/Documents/GitHub/mobiair/distprob/example_uni_stu.csv" )
write.csv(aa2, "~/Documents/GitHub/mobiair/distprob/example_school_U17.csv" )
write.csv(aa3, "~/Documents/GitHub/mobiair/distprob/example_parttime.csv" )
write.csv(aa4, "~/Documents/GitHub/mobiair/distprob/example_fulltime.csv" )
#write.csv(aa1, "~/Documents/GitHub/mobiair/distprob/illu_uni_stu.csv" )
#write.csv(aa2, "~/Documents/GitHub/mobiair/distprob/illu_school_U17.csv" )
#write.csv(aa3, "~/Documents/GitHub/mobiair/distprob/illu_parttime.csv" )
#write.csv(aa4, "~/Documents/GitHub/mobiair/distprob/illu_fulltime.csv" )

 

maketheplot = function(aa2, plotname) {
  aa2["distance"] = as.numeric(rownames(aa2))
  aa2 = aa2%>%arrange( distance)
  dfaa = gather(aa2, "transportation_mean", "value", -distance)
  
  
  ggplot(dfaa, aes(x=as.factor(distance/1000), y =value, fill = transportation_mean)) +
    geom_bar(stat = "identity")+scale_fill_brewer(palette="Paired")+ylab("probability")+xlab("distance to school (km)")+theme_bw()+
    theme(
      panel.border = element_blank(), # frame or not
      strip.background = element_rect(
        color="#FFFFFF", fill="#FFFFFF", size=0.5, linetype = NULL),
      axis.text.x = element_text(angle = 45),
      strip.text.y = element_text(
        size = 12, color = "black", face = "bold"
      ),
      panel.grid.major = element_blank(), panel.grid.minor = element_blank()
    )+ scale_x_discrete(labels=c("1" = "less than 1", "2.5" = "1 - 2.5",
                                 "3.7" = "2.5 - 3.7","5" = "3.7 - 5","7.5" = "5 - 7.5","10" = " 7.5- 10", "15" = "10 - 15", "100" = "Longer than 15"))
  
  ggsave(plotname)
}
maketheplot(aa2= aa1, "~/Documents/GitHub/mobiair/ditance_vs_transmean_hal_work.png")

