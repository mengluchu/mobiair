library(raster)
library(rgdal)
cextent = readOGR("~/Documents/GitHub/mobiair/NLD_adm/NLD_adm0.shp")
allmap = list.files("/Volumes/Meng_Mac/100m/laea", pattern = ".map$", full.names = T)


s = stack(allmap)
#names(stack("/Volumes/Meng_Mac/NL100.gri"))
ext <- sp::spTransform(cextent, crs(s))
cs = crop(s,ext)
myRaster <- writeRaster(cs,filename =paste0("/Volumes/Meng_Mac/NL_100m,",names(cs)), bylayer =TRUE, format="GTiff") 
myRaster <- writeRaster(cs,"/Volumes/Meng_Mac/NL100.gri") 

nlayers(cs)
# not good practice, super slow, takes maybe 8 hours
cs = projectRaster(cs, crs = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")) 

plot(cs[[1]])
myRaster <- writeRaster(cs,"/Volumes/Meng_Mac/NL100m.tif", format="GTiff")
a=raster("/Volumes/Meng_Mac/NL100m.tif")
#writeRaster(stk, filename=names(stk), bylayer=TRUE,format="GTiff")
plot(a)