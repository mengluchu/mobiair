sav = read_spss("~/Downloads/dutch_activity/OViN2014_Databestand.sav")
names(sav)
sav[1,1:4]
library(data.table)
a = fread("~/Documents/GitHub/mobiair/human_data/trip_survey.csv", encoding="unknown", stringsAsFactors=T)
names(a)
a[2]
