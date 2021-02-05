library(sf)
library(purrr)
library(geosphere)
library(tibble)
library(raster)
library(readr)
library(tidyr)
library(stringr)
#param sched schedule
#param loc route, buffer or location to extract values
#param rasterpath path for predictions.
expo=function(sched, loc = sf1, rasterpath= "~/Documents/GitHub/mobiair/prediction/"  )
{
  start =floor(sched$start_time)
  end = ceiling(sched$end_time)
  lf = list.files(rasterpath, pattern = "*tif$")
  
  ras = paste0 (rasterpath, "NL100_t",start[act_num]:end[act_num], ".tif")
  st= stack(ras) # this takes quite sometime
  #meanap = calc(st, mean)
  exp = unlist( raster::extract(st, loc))
  mean(exp)* (sched$end_time - sched$start_time)[act_num]
}
#sort generated route based on the  pattern of "_sth_num_" 
sortlist = function(sflist1){
  l = lapply(sflist1, str_split,"_" )%>%unlist
  ind = c(seq(3, length(l), by =4))
  sflist1 [ order(as.numeric(l[ind]))]
}
#library(geojsonsf)
#geojson_sf("~/Documents/GitHub/mobiair/route.json")

rasterVis::levelplot(a)
a = raster("~/Documents/GitHub/mobiair/prediction/NL100_t6.tif")
sflist1 =list.files("~/Downloads/route_real/", full.names = T)
sflist2 =list.files("~/Downloads/route_1/", full.names = T)
sflist3 =list.files("~/Downloads/route_2/", full.names = T)
sflist4 =list.files("~/Downloads/route_3/", full.names = T)


schflist =list.files("~/Downloads/act_schedule/", full.names = T)

sflist1 = sortlist(sflist1)
sflist2 = sortlist(sflist2)
sflist3 = sortlist(sflist3)
sflist4 = sortlist(sflist4)
schflist = schflist[order(lapply(schflist, parse_number)%>%unlist)]
 
expos = function(i, sflist = sflist1){
  
  sf1 =st_read(sflist[i])
  #sf1 =st_read("~/Documents/GitHub/mobiair/route2.shp")
  #sf1 = sf1%>%st_set_crs( "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
  
  sf1 = st_transform(sf1 , crs = "+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")
  act_num = 2 # w2h
  sched = read.csv(schflist[i])
  expo(sched=sched, loc = sf1)
}
h2w_real = sapply(1:30,expos, sflist = sflist1)
h2w_sim = sapply(1:30,expos, sflist = sflist2)
h2w_sim2 = sapply(1:30,expos, sflist = sflist3)
h2w_sim3 = sapply(1:30,expos, sflist = sflist4)
plot(h2w_real, typ = "b") 
points( h2w_sim, col = "red")
points( h2w_sim2, col = "green")
points( h2w_sim3, col = "blue")
expos(i,sflist = sflist2)
system.time(expo(sched))# 1.2seconds
df = data.frame(id  = 1:length(h2w_real),h2w_real,h2w_sim,h2w_sim2,h2w_sim3)
dfg = df%>%gather(key, value, -id)
dfg
ggplot(dfg, aes(id, value))+geom_point(aes(color = factor(key), size = factor(key), shape =factor( key)))+ 
  scale_color_manual(values=c("#FF3333","#FFC0CB", "#D2B48C", "#87CEFA")) +scale_size_manual(values=c(2.8,1.8,1.8,1.8))+
  theme_classic() 
ggsave("~/Downloads/sims.png")
#library(ggplot2)
#ggplot(data.frame(exp))+geom_line()
#as.numeric(substr(lf[i], start = 8, stop  = nchar(lf[i])-4))

get rasters > = start and <=end
getmean
h2wexp = mean*(sched$end = shed$starttime)
hexp = same way
workexp same way
h2sportsameway

```
ext = extract(a, sf1, along= TRUE)
ext = ext[[1]]
mean(ext)

str(sf1$geom)
transect_df = purrr::map_dfr(ext, as_tibble ,.id = "ID")
transect_df
plot(ext, typ = 'b')
plot(a)
plot(sf1, add = T )

sf1[1]$geom[[1]]
#> Warning: `as_data_frame()` is deprecated as of tibble 2.0.0.
#> Please use `as_tibble()` instead.
#> The signature and semantics have changed, see `?as_tibble`.
#> This warning is displayed once every 8 hours.
#> Call `lifecycle::last_warnings()` to see where this warning was generated.
transect_coords = xyFromCell(a, transect_df$cell)

pair_dist = geosphere::distGeo(transect_coords, f = 1/298.257222101)[-nrow(transect_coords)]
transect_df$dist = c(0, cumsum(pair_dist))
