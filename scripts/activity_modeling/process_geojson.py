#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:10:43 2021

@author: menglu
"""
from rasterio.crs import CRS
import rasterio
import pyproj
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point



a = gpd.tools.geocode("Utrecht")
a.buffer(1)
# pyproj.CRS("+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs ").to_epsg())

#gdf.to_crs("+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")

filedir = "~/Documents/GitHub/mobiair/locationdata/

home_csv = filedir+"Uhomelatlon.csv"
df = pd.read_csv(home_csv)
df.columns
gdf = gpd.GeoDataFrame( 
    crs={'init': 'epsg:4326'},
    geometry=[Point(xy) for xy in zip(df.lon, df.lat)])

type(gdf.iloc[1]['geometry'])
#gdf.iloc[1]['geometry'].buffer(1)

rasterfile = "/Volumes/Meng_Mac/NL_100m/road_class_3_300.tif"

new_image = rasterfile
output_image = os.path.join("/Volumes/Meng_Mac/", "try1.tif")
src = rasterio.open(new_image)
dataset
with rasterio.open(new_image, 'r') as src:
  profile = src.profile
  profile.update(
      dtype=rasterio.uint8,
      count=1,
  )
data = src.read(window=window)


with rasterio.open(output_image,'w', **profile) as dst:
    patch_size = 500

    for i in range((src.shape[0] // patch_size) + 1):
        for j in range((src.shape[1] // patch_size) + 1):
            # define the pixels to read (and write)
            window = rasterio.windows.Window(
                j * patch_size,
                i * patch_size,
                # don't read past the image bounds
                min(patch_size, src.shape[1] - j * patch_size),
                min(patch_size, src.shape[0] - i * patch_size)
            )
            print(window)
            data = src.read(window=window)
            
            img_swp = np.moveaxis(data, 0, 2)
            img_flat = img_swp.reshape(-1, img_swp.shape[-1])

            img_ndvi = band_index(img_flat, 3, 2)
            img_ndwi = band_index(img_flat, 1, 3)

            img_w_ind = np.concatenate([img_flat, img_ndvi, img_ndwi], axis=1)

            # remove no data values, store the indices for later use
            # a later cell makes the assumption that all bands have identical no-data value arrangements
            m = np.ma.masked_invalid(img_w_ind)
            to_predict = img_w_ind[~m.mask].reshape(-1, img_w_ind.shape[-1])
            
            # predict
            if not len(to_predict):
                continue
            
            img_preds = lgbm.predict(to_predict)
            
            # add the prediction back to the valid pixels (using only the first band of the mask to decide on validity)
            # resize to the original image dimensions
            output = np.zeros(img_flat.shape[0])
            output[~m.mask[:,0]] = img_preds.flatten()
            output = output.reshape(*img_swp.shape[:-1])
            
            # create our final mask
            mask = (~m.mask[:,0]).reshape(*img_swp.shape[:-1])

            # write to the final file
            dst.write(output.astype(rasterio.uint8), 1, window=window)
            dst.write_mask(mask, window=window)
            
            
            
            
            
gdf.buffer(1)
import json
from osgeo import ogr, osr


a = gpd.read_file('/Users/menglu/Downloads/routes/h2w_10_bicycle.gpkg')
a.plot()
gpd.read_file('/Users/menglu/Documents/GitHub/mobiair/locationdata/Ut_indoorsport.gpkg')

gpd.read_file('/Users/menglu/Documents/GitHub/mobiair/aamyfile.geojson')
f = open('/Users/menglu/Documents/GitHub/mobiair/aamyfile.geojson')
data = json.load(f)

import gdal
ds = gdal.OpenEx(data)
layer = ds.GetLayer()
featureCount = layer.GetFeatureCount()
 
feature = layer.GetNextFeature()
#feature.GetGeometryRef().ExportToWkb()
 
print(ds.GetDriver().ShortName)
print(feature.GetGeometryRef().ExportToWkt())

####

mem_ds = ogr.GetDriverByName('MEMORY').CreateDataSource('')
dst_layer = mem_ds.CreateLayer('route', geom_type=ogr.wkbLineString)

dst_layer


wgs_layer.ResetReading()
wgs_feature = wgs_layer.GetNextFeature()

wgs_feature
wgs_geometry = wgs_feature.GetGeometryRef()
wgs_geometry   
dst_geometry = ogr.CreateGeometryFromWkb(wgs_geometry.ExportToWkb())
feature = ogr.Feature(dst_layer.GetLayerDefn())
feature.SetGeometry(dst_geometry)
dst_layer.CreateFeature(feature)
feature = None
dst_layer = mem_ds.GetLayer()

source_feature = dst_layer.GetNextFeature()
source_feature

env = source_feature.GetGeometryRef().GetEnvelope()
env[1]

target_ds = gdal.GetDriverByName('MEM').Create('', xsize=route_nr_cols, ysize=route_nr_rows, bands=1, eType=gdal.GDT_Byte)
    target_ds.SetGeoTransform((route_min_x, self.cellsize, 0, route_max_y, 0, -self.cellsize))



