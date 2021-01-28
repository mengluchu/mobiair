# from a persons social occupation, e.g. full time working or not, and the purpose, e.g. go to work, and going to do hobby, we can get a distribution of the distance. We sample from this distribution, calculate a buffer of possible locations to go. 
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


#plot
aa2 = read.csv("~/Documents/GitHub/mobiair/distprob/illu_fulltime.csv" )
aa2["distance"] = rownames(aa2)
library(tidyr)
dfaa = gather(aa2, "transportation_mean", "value", -distance)

 
ggplot(dfaa, aes(x=distance, y =value, fill = transportation_mean)) + 
  geom_bar(stat = "identity")+scale_fill_brewer(palette="Paired")+ylab("probability")+xlab("distance to school")+theme_bw()+
  theme(
    panel.border = element_blank(), # frame or not 
    strip.background = element_rect(
      color="#FFFFFF", fill="#FFFFFF", size=0.5, linetype = NULL),
    axis.text.x = element_text(angle = 45),
    strip.text.y = element_text(
      size = 12, color = "black", face = "bold"
    ),
    panel.grid.major = element_blank(), panel.grid.minor = element_blank()
  )


 
summary(lm(age_lb~Rvm+Sted, data =a1 ))
summary(lm(KAf_mean~Rvm+Maat, data =a ))

summary(lm(KAf_mean~Rvm, data =a1 ))

summary(lm(KAf_mean~age_lb, data =a1 ))

summary(lm(KAf_mean~age_lb, data =a1 ))

names(a)
summary( aov(KAf_mean~MaatsPart, data = work)) # occupation is associated with travel distance
summary( aov(age_lb~Rvm, data = work)) # age is associated with travel mean
summary( aov(age_lb~Rvm+KVertTijd, data = work)) # age is associated with travel mean and time to depart
install.packages("ggpubr")
library("ggpubr")
ggboxplot(work, x = "Rvm", y = "age_lb",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))
 
ggboxplot(uni, x = "Rvm", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))


ggboxplot(schoolstu, x = "Rvm", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))

ggbarplot(ss,x= "Rvm", y = "freq")+theme(axis.text.x = element_text(angle = 90))

train = ss%>%filter(Rvm =="Trein")%>%select(freq)
walk = ss%>%filter(Rvm =="Te voet")%>%select(freq)

bike = ss%>%filter(Rvm =="Fiets (elektrisch en/of niet-elektrisch)")%>%select(freq)
auto = 1-train - walk - bike



ggboxplot(halftime, x = "Rvm", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))

ggboxplot(work, x = "MaatsPart", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))
 
#Generate distribution of travel distance to work according to social occupationalStatus, then based on this we can choose work locations
socialpartition = unique(a$MaatsPart)
i = 1
mu1 = c()
sd1 = c()
for ( i in 1: 10){
socialpartition[i]
mmax = a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[i])%>%select(KAf_mean)%>%max
mu =  a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[i])%>%select(KAf_mean)%>% apply(2, function(x) x+0.1) %>% apply(2, log)%>%apply(2, mean)
sd =  a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[i])%>%select(KAf_mean)%>% apply(2, function(x) x+0.1)  %>% apply(2, log)%>%apply(2, sd)
mu1 [i] = mu
sd1[i] = sd
}
df = shopping

getmean_sd = function (df)
{
mu =  df%>%select(KAf_mean) %>%apply(2, function(x) x+1) %>% apply(2, log)%>% apply(2, mean)
sd =  df%>%select(KAf_mean) %>%apply(2, function(x) x+1) %>% apply(2, log)%>% apply(2, sd)
return (list(mu, sd))
}
ss= student%>%select(KAf_mean)
ss2= student%>%select(KAf_mean)%>%apply(2, function(x) x+1)
dfss = data.frame(ss)
dfss["log(distance)"] = (log(ss2)-1)
names(dfss)[1]= "distance"
dfss = melt(dfss)

ggplot(dfss, aes(x=value)) + 
  geom_histogram(color="black",aes(y = stat(count) / sum(count)), fill="white", bins = 30)+geom_density(alpha=.2, fill="lightblue") +facet_grid(variable~.)+
  ylab("frequency")+xlab("distance to school")+theme_bw()+
  theme(
    panel.border = element_blank(), # frame or not 
    strip.background = element_rect(
      color="#FFFFFF", fill="#FFFFFF", size=0.5, linetype = NULL),
    
    strip.text.y = element_text(
      size = 12, color = "black", face = "bold"
    ),
    panel.grid.major = element_blank(),panel.grid.minor = element_blank()
  )

ggsave("~/Documents/GitHub/mobiair/ditance_to_school.png")

hist(log(ss))
getmean_sd (student)

getmean_sd (shopping)
getmean_sd (work)
getmean_sd (studentshopping)
getmean_sd (student)

mu1 [i] = mu
sd1[i] = sd
# 

t1 = rnorm(1000, mean = 20, sd = 1)
tlog = log (t1)
t2 = rlnorm(1000, meanlog = mean(tlog), sdlog= sd(tlog)) 

hist(t1)
hist(t2)
mean(t1)
mean(t2)
sd(t1)
sd(t2)
              #exp(mean(t1) + 0.5*(sd(t1)^2)) 
hist(t1)
hist(exp(t2))

plot(sd1)
plot(mu1, col = "red")
expn = exp(rnorm (20, mu, sd))
expn[expn<mmax]

par(mfrow =c(1,2))
hist(expn[expn<50])
hist(a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[1])%>%select(KAf_mean)%>%unlist)
   #apply(2, function(x){x+1})
exp(log(10))
 dlnorm(x, meanlog = 0, sdlog = 1, log = FALSE)
 