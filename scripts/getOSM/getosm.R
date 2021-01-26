install.packages("osmdata")
library(osmdata)
library(raster)
 

indoorsport = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "leisure", value =c("sports_hall", "sports_centre", "fitness_centre")) %>% osmdata_sf() 
Univ_col = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "amenity", value =c("university", "college")) %>% osmdata_sf() 
school = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "amenity", value =c("school")) %>% osmdata_sf() 

a = data.frame(Univ_col$osm_points$geometry)
plot(school$osm_points$geometry)
length(school$osm_points$geometry)