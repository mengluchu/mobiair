#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:21:40 2021

@author: menglu
"""
import pandas as pd 
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from rasterstats import zonal_stats, point_query
import pyproj
from shapely.ops import transform
import modelutils as m
import rasterio
from matplotlib import pyplot as plt
from rasterio.plot import show_hist
from math import modf
import osmnx as ox
from scipy import signal 
wuhan = ox.geocode_to_gdf('武汉, China')
utrecht = ox.geocode_to_gdf('Utrecht province') 
utrecht.plot() 
filedir = "/Users/menglu/Documents/GitHub/mobiair/"
preddir =f"{filedir}prediction/"

savedir = "/Volumes/Meng_Mac/mobi_result/Uni/" # each profile a savedir. 
                   
def wgs2laea (p):
    wgs84 = pyproj.CRS('EPSG:4326')
    rd= pyproj.CRS('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    project = pyproj.Transformer.from_crs(wgs84, rd, always_xy=True)
    p=transform(project.transform, p)
    return (p)  

def plot_raster ():
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(55,15)) 
                            
    for i, ax in enumerate(axs.flat):
        src = rasterio.open(f'{preddir}NL100_t{i}.tif')

        ax.set_axis_off()
        a=ax.imshow(src.read(1), cmap='pink')
        ax.set_title(f' {i:02d}:00')
         
    cbar = fig.colorbar(a, ax=axs.ravel().tolist())
    cbar.set_label(r'$NO_2$', rotation = 270)

    plt.show()

#plt.savefig(savedir+"prediUt.png")

#show_hist(src, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)

#plt.show()

#ls = os.listdir(preddir)

def gethw (df):
    ph=Point(float(df.home_lon) , float(df.home_lat) )
    ph = wgs2laea(ph)
    pw=Point(float(df.work_lon) , float(df.work_lat) )
    pw = wgs2laea(pw)
    return(ph, pw)

def buffermean(p, ext , rasterfile):
   
    pbuf=p.buffer(ext)
    z= zonal_stats(pbuf, rasterfile, stats = 'mean')[0]['mean']
    
    return z  


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


#plt.imshow(gkern(2000,500))
    
#extract pollution map in the buffer and convolve with gaussian kernel.  
def gaussKernconv(p, ext , rasterfile, sd = 1):
   
    pbuf=p.buffer(ext)
    getras= zonal_stats(pbuf,rasterfile ,stats="count", raster_out=True)[0]['mini_raster_array']
 
    ag = gkern (getras.data.shape[0], sd)
    
    z = signal.convolve2d(getras.data,ag, mode = 'valid') # valid does not pad. 
      
    return z  


''' for testing
j =1
homework.loc[j]
ph, pw = gethw(homework.loc[j])
rasterfile =f'{preddir}NL100_t{1}.tif'
buffermean(ph, 300, rasterfile)
p = ph
''' 



def getconcen(act_num, rasterfile, df, routegeom, ext = 100, extgaus = 2000, sd=300, indoor_ratio = 0.7 ):
    ph, pw = gethw(df)
    return {
       1: indoor_ratio * point_query(ph, rasterfile ).pop(), #home
       2: np.nan_to_num(np.mean(point_query(routegeom, rasterfile)),0),# route #can easily make buffer out of it as well, here the ap already 100m so not needed.
       3: indoor_ratio *point_query(pw, rasterfile ).pop(), # work_indoor
       4: buffermean(p, ext, rasterfile), # freetime 1 will change into to sport (second route)
       5: gaussKernconv(p, extgaus, rasterfile, sd = 1), # freetime 2, distance decay, outdoor. 
       6: buffermean(p, ext, rasterfile)  # freetime 3, in garden or terras  
       } [act_num]                                              

#schefile = os.listdir(schedir)
 

 
# rasterfile = f'{preddir}NL100_t{i}.tif'

def remove_none(nparray): 
    arr = np.array(nparray)
    return(arr[arr!= np.array( None)])
'''  
#test
for i in range(1,7):
    getconcen(act_num= i,
              rasterfile=f'{preddir}NL100_t{i}.tif', 
              df = Uni_ut_homework.loc[j], 
              routegeom=route.loc[j]['geometry'])
'''
#still doing hourly
ext = 300 # 300 m 
iteration = 1

ODdir = savedir +"genloc/"
ODfile =f'h2w_{iteration}.csv' 
homework =gpd.read_file(ODdir+ODfile) # for comparison #gpd can read csv as well, just geom as None.

def cal_exp(filedir, savedir, iteration, ext = 100, extgaus=2000, gaussd = 300,  save_csv = True):
    ODdir = savedir +"genloc/"
    ODfile =f'h2w_{iteration}.csv' 
    homework =gpd.read_file(ODdir+ODfile) # for comparison #gpd can read csv as well, just geom as None.
    routedir = savedir+'genroute/'
    routefile = f'route_{iteration}.gpkg' # get route file for all people, only one route file,geodataframe
    route= gpd.read_file(routedir+routefile)
    route = route.to_crs('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    
    schedir = savedir+'gensche/'
    # exp is concentration weighted by time duration
    exp_each_act = []
    exp_each_person= []
    n = len(homework)
    for j in range(n): #iterate over each person
    
        sched = pd.read_csv(f'{schedir}ws_iter_1_id_{j}.csv') #each person has a schedule, only schedule is file per person.
        start = sched['start_time']
        end =sched['end_time']
        start_int=np.floor(start).astype(int)
        end_int = np.ceil(end).astype(int) # for using range, actually np.floor ()
        act_num = sched['activity_code']
        
        for k in range(sched.shape[0]): # iterate over schedule
            conh = 0 # hourly concentration for each activity
            missingtimeh = 0 
            missingtime = 0
            if start_int[k] == end_int[k]-1: # less than one hour
                    con = getcon(act_num[k], f'{preddir}NL100_t{start_int[k]}.tif',  homework.loc[j], route.loc[j]['geometry'], ext = ext, extgaus = extgaus, sd = gaussd)
                    if con is not None or con is not np.nan:
                        conh = con * (end[k]-start[k]) # start percentage multiply by concentration of the hour, next hour will get the rest of the percentage
                        missingtimeh=0
                    else: 
                        conh = 0
                        missingtimeh = end[k]-start[k]
                 
            else: # more than one hour
                for i in range(start_int[k],end_int[k]): # iterate over raster
                    con = getcon(act_num[k], f'{preddir}NL100_t{i}.tif', homework.loc[j], route.loc[j]['geometry'], ext = ext, extgaus = extgaus, sd = gaussd)
                    if i ==start_int[k]: # first hour may be from e.g. 7:20 instead of 7:00
                        if con is not None or con is not np.nan:
                            cons = con * (1- modf(start[k])[0]) # start percentage multiply by concentration of the hour, next hour will get the rest of the percentage
                            missingtime=0
                        else: 
                            cons = 0
                            missingtime = modf(start[k])[0]
                    elif i == end_int[k]: # last hour may be to e.g. 9:20 instead of 9:00
                        if con is not None or con is not np.nan:
                            cons = con * modf(end[k])[0] # end percentage
                            missingtime=0
                        else: # for none values or nan, assign valye 0 and note missing times
                            cons = 0
                            missingtime = modf(end[k])[0]               
                    else:
                        if con is not None or con is not np.nan:
                            cons = con # middle times
                            missingtime=0
                        else: # for none values or nan, assign valye 0 and note missing times
                            cons = 0
                            missingtime = end_int - start_int  -1          
    
                    
                # summing exposures
                    conh= conh +cons
                    #exp_each_hour.append(cons)
                    missingtimeh = missingtimeh + missingtime 
                    
            exp = conh/(end[k]-start[k]-missingtimeh+0.01) # average exp per activity
            exp_each_act.append(exp)
            #con_each_person.append(np.nanmean(remove_none(con_each_act[k*j : (k+1)*j ]) ))      
        exp_each_person.append(np.nanmean(remove_none(con_each_act[j*sched.shape[0]:(j+1)*sched.shape[0]])))
        print(j)
    
    if save_csv:
        exposuredir =f"{savedir}exposure/" 
        m.makefolder(exposuredir)    
        pd.DataFrame(exp_each_act).to_csv(f'{exposuredir}iter_{iteration}_act.csv')
        pd.DataFrame(exp_each_person).to_csv(f'{exposuredir}iter_{iteration}_person.csv')
    return (exp_each_act, exp_each_person)    
      
#act, person = cal_exp(filedir, savedir, iteration, save_csv = True)
act = pd.read_csv(f"{savedir}exposure/iter_{1}_act.csv").iloc[:,1]
person = pd.read_csv(f"{savedir}exposure/iter_{1}_person.csv").iloc[:,1]

# plot
def formattime(timeinput): 
  
    minute, hour = modf(timeinput)
    minute = np.floor(minute *60)     
    return "%02d:%02d" % (hour, minute) 

def plotact(sub1, sub2,  savename="1",act = act, simplify = True, select = 0):
    
    schedir = savedir+'gensche/'

    fig, ax = plt.subplots(sub1,sub2,figsize=(18, 56), sharey=True )
    axs = ax.flatten()
    for i1 in range (sub1*sub2):
            i = i1 + select
            sch = pd.read_csv(f'{schedir}ws_iter_1_id_{i}.csv')
            st = sch['start_time']
            et = sch['end_time']
            axs[i1].plot(list(st),act[i*7:(i+1)*7], "ko-") 
            #st= np.round(st,1)
            ind = np.where(np.diff(st)<1.5)[0]
            if simplify:
                xlabels = list(sch['activity'])
            else:
                xlabels = [f"{x3}: {x1} to {x2}" for x1, x2, x3, in zip(map(formattime,list(st)),map(formattime,list(et)),list(sch['activity']))]
            
            for j in range(7):
                x, y = st[j],list(act[i*7:(i+1)*7])[j] 
                t = axs[i1].text(x, y+2, xlabels[j] )
            #list(sch['activity'])[j] 
            et = et.drop(ind)
            st=st.drop(ind)
            #xlabels = [f"{x1} to \n{x2}" for x1, x2, in zip(map(formattime,st), map(formattime,et))]
            axs[i1].set_title(f'person ID: {i}')
            axs[i1].set_xlabel('hour')
            axs[i1].set_xticks(st) 
            axs[i1].set_xticklabels(map(formattime,st))
            axs[i1].tick_params(axis='x',labelrotation =45,bottom=True,length=5)
            axs[i1].set_ylabel("Exposure: " r'$ \mathrm{NO}_2$', fontsize=10)
    #fig.supxlabel("hour")
    #fig.supylabel("Exposure: " r'$ \mathrm{NO}_2$', fontsize=10)
    fig.tight_layout()      
    fig.savefig(f'{savedir}exposure_act{savename}.png') 
plotact(sub1 = 2, sub2 =4, savename="more", simplify=True, select = 2)    
# people
lat = np.array(homework.home_lat).astype(float)
lon = np.array(homework.home_lon).astype(float)
 
df1 = [person,lat, lon]
 
df2 = pd.DataFrame(data=df1).T
df2 = df2.rename (columns = {'0':"personal_exposure", "Unnamed 0": "lat", "Unnamed 1": "lon" })
exp_gdf = gpd.GeoDataFrame(df2["personal_exposure"], crs={'init': 'epsg:4326'},
                                     geometry=[Point(xy) for xy in zip(df2.lon, df2.lat)])

exp_gdf.to_file(f'{savedir}person_iter{iteration}.gpkg')
fig, ax = plt.subplots()
ax.set_aspect('equal')
exp_gdf.plot(ax=ax, column = 'personal_exposure',legend=True)
#
#src = rasterio.open(f'{preddir}NL100_t{i}.tif')
ax.imshow(src.read())
ax.set_axis_off()

plt.show()
 plt.close('all')
 
