install.packages("osmdata")
library(osmdata)
library(raster)
library(stars)

indoorsport = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "leisure", value =c("sports_hall", "sports_centre", "fitness_centre")) %>% osmdata_sf() 
Univ_col = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "amenity", value =c("university", "college")) %>% osmdata_sf() 
school = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "amenity", value =c("school")) %>% osmdata_sf() 


#road1 = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "highway", value = c("primary", "primary_link")) %>% 
#  osmdata_sf()
type = c("motorway","motorway_link","trunk,trunk_link",
"primary","primary_link",
"secondary","secondary_link",
"tertiary","tertiary_link",
"residential","residential_link")


osm_query2raster = function(bboxname, key, value, rastername, resx, resy)
{ road = opq(bbox=bboxname )%>%add_osm_feature(key = key, value =value) %>% osmdata_sf()
road = st_rasterize(road$osm_lines,dx = resx, dy = resy )
write_stars(road, rastername)
}

osm_query2raster(bboxname = "phoenix arizona", value = "primary", "ph1.tif", 0.005, 0.005)

plot(raster("try.tif"))
plot(a)
getwd()
r2 = raster("~/Documents/R_default//utrecht/road2.tif")
r3 = raster("~/Documents/R_default//utrecht/road3.tif")
r4 = raster("~/Documents/R_default//utrecht/road4.tif")
r5 = raster("~/Documents/R_default//utrecht/road5.tif")
r3 =resample(r3, r2)
r4 = resample(r4, r2)
r5 = resample (r5, r2)

writeRaster(r5,filename = "~/Documents/R_default/utrecht/road5.tif", overwrite =T)
 
?resample
road2 = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "highway", value =c("primary", "primary_link")) %>% 
  osmdata_sf()
road3 = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "highway", value = c("secondary","secondary_link")) %>% 
  osmdata_sf()
road4 = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "highway", value = c("tertiary","tertiary_link")) %>% 
  osmdata_sf()
road5 = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "highway", value = c("residential","residential_link")) %>% 
  osmdata_sf()
road2 = st_rasterize(road2$osm_lines,dx = 0.0005, dy = 0.0005 )
road3 = st_rasterize(road3$osm_lines,dx = 0.0005, dy = 0.0005 )

road4 = st_rasterize(road4$osm_lines,dx = 0.0005, dy = 0.0005 )
road5 = st_rasterize(road5$osm_lines,dx = 0.0005, dy = 0.0005 )

write_stars(road5, "road5.tif")

plot(raster("road5.tif"))
plot(a)

osm_query2raster(bboxname = "utrecht netherlands", "highway", value = type[1:4], "try.tif", 0.005, 0.005)
plot(raster("ttry.tif"))
library(sf)

nc = st_transform(st_read("/Volumes/TOSHIBA EXT/global_mapping_dataglll/gap_class_1.gpkg", layer = "lines",package="sf"), aoi = c(51.992876, 5.004480, 52.192876, 5.204480 ))

a = st_layers("/Volumes/TOSHIBA EXT/global_mapping_dataglll/gap_class_4.gpkg")
b = st_sfc(st_point(c(52.092876, 5.104480))) # create 2 points
b = st_buffer(b, dist = 1) # convert points to circles, 1 degree
plot(b)

ogr2ogr -f 'GPKG' -clipsrc 51.992876, 5.004480, 52.192876, 5.204480 "utrecht.gpkg" "/data/gghdc/gap/2020/data" Towns
st_crop(a, b)

library(dplyr)
cities_sqlite <- 
  tbl(src_sqlite("/Volumes/TOSHIBA EXT/global_mapping_dataglll/gap_class_4.gpkg"), "lines")%>%
  filter(name=="Utrecht")

print(cities_sqlite, n = 5) 