library(raster)
library(rgdal)
cextent = readOGR("~/Downloads/NLD_adm/NLD_adm0.shp")
allmap = list.files("/Volumes/Meng_Mac/100m/laea", pattern = ".map$", full.names = T)


s = stack(allmap)

ext <- sp::spTransform(cextent, crs(s))

cs = crop(s,ext)
# not good practice, super slow, takes maybe 8 hours
cs = projectRaster(cs, crs = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")) 
 
myRaster <- writeRaster(cs,"/Volumes/Meng_Mac/NL100m.tif", format="GTiff")
a=raster("/Volumes/Meng_Mac/NL100m.tif")
plot(a)