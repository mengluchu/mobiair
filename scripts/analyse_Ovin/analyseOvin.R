# from a persons social occupation, e.g. full time working or not, and the purpose, e.g. go to work, and going to do hobby, we can get a distribution of the distance. We sample from this distribution, calculate a buffer of possible locations to go. 
library(data.table)
library(dplyr)
a = fread("~/Downloads/Ovin.csv")
head(a)
summary(as.numeric(a$VertPC))
nrow(a)
summary(lm(KAf_high~age_lb+Sted, data =a ))
work = a%>%filter(Doel == "Werken")
shopping = a%>%filter(Doel == "Winkelen/boodschappen doen")
a$MaatsPart%>%unique()


studentshopping = a%>%filter(Doel == "Winkelen/boodschappen doen"& MaatsPart=="Scholier/student")
single = a%>%filter(Doel == "Werken" & MaatsPart=="Eigen huishouding")
student = a%>%filter(Doel == "Werken" & MaatsPart=="Scholier/student")
halftime = a%>%filter(Doel == "Werken" & MaatsPart=="Werkzaam 12-30 uur per week")
"Werkloos"
a$MaatsPart%>%unique()
a$Doel%>%unique()

names(a)
 
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

ggboxplot(single, x = "Rvm", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))


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
ss= student%>%select(KAf_mean)%>%apply(2, function(x) x+1)
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
 