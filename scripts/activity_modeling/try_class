#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:32:29 2021

@author: menglu
"""

class Gen_dest:
    def __init__(self, homedf, goal, n, csvname):
     self.homedf= homedf
     self.goal = goal
     self.n = n
     self.csvname =csvname
     
    def log_mean_sd (df):
     mu =  df.apply(lambda x: x+1). apply(lambda x: np.log(x)).apply(lambda x: np.mean(x))
     sd =  df.apply(lambda  x: x+1). apply(lambda x: np.log(x)).apply(lambda x: np.std(x))
      
     return (*mu, *sd)
    
    # Get the mean and standard deviation (SD) given a dataframe column. 
    def mean_sd (df):
     mu =  df.apply(lambda x: np.mean(x))
     sd =  df.apply(lambda x: np.std(x))
      
     return (*mu, *sd)
    
    # Log transform and then get mean and standard deviation (SD) given a dataframe column, using scipy, same as log_mean_sd
    def lognormal_mean_sd_scipy(df):
        shape, loc, scale = scipy.stats.lognorm.fit(df.apply(lambda x: x+1), floc=0)
        sd = shape
        mu = np.log(scale)
        return (mu, sd)
     
    # generate lognorm distribution, "manual" use log_mean_sd, else use scipy to get mean and sd.     
    def gen_lnorm(df, method="manual"):
      if method == "manual":
            mu, sd = log_mean_sd(df)   
      else:      
            mu, sd = lognormal_mean_sd_scipy(df)      
            #mu1, sd1 = log_mean_sd(df)
            #print(mu-mu1,sd-sd1) for checking purpose. same results
      gen = np.random.lognormal (mu, sd, 1)-1
      if gen < 0:
          gen = 0.1 
      return(gen)
     
    #based on social occupation and travel goal, output work, outdoor, shopping distances, based on the Ovin dataset paprameter name
    def distance(socialpartition="Scholier/student"):
        if socialpartition == "Scholier/student":# only consider eductation for students
            work_dist = Ovin.query('Doel == "Onderwijs/cursus volgen" & MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
        else:
            work_dist = Ovin.query('Doel == "Werken" & MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
        
        outdoor_dist = Ovin.query('Doel == "Sport/hobby" &  MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
        shopping_dist = Ovin.query('Doel =="Winkelen/boodschappen doen"& MaatsPart=="{}"'.format(socialpartition))[['KAf_mean']]
        return(work_dist,outdoor_dist,shopping_dist)
    
    # given two geopoints, home and work (origin and destination), output an array with [xcoord_home,ycoord_home,xcoord_work,ycoord_work]
    def disto2d(homeloc,workloc): # input geopoints. 
        xcoord_work = workloc.centroid.x
        ycoord_work = workloc.centroid.y
        xcoord_home = homeloc.centroid.x
        ycoord_home =homeloc.centroid.y
        p = [xcoord_home,ycoord_home,xcoord_work,ycoord_work]
        return(p)
    '''
    # input:  a dataframe of lon, lat
    # output: geopandas dataframe and a union of it. 
    # calculate once so make it a function. 
    '''
    def pot_dest(goal):
        if type(goal) is pd.core.frame.DataFrame: 
                w_gdf = gpd.GeoDataFrame(crs={'init': 'epsg:4326'},
                                         geometry=[Point(xy) for xy in zip(goal.lon, goal.lat)])
        else: # geopandas
            w_gdf = goal
        u = w_gdf.unary_union
        return (w_gdf, u)
    
    '''
    input: p: home point, w_gdf: destination (e.g. work) geopandas, u, union geopandas, 
           goal: activity for getting the distance, sopa: social participation for getting the distance
    output: home point, destinaton point, number of candidate points (to sample a destination point from).
    ''' 
    def getdestloc (p, w_gdf, u, goal = "work", sopa = "Scholier/student"):
        
        nearestpoint = nearest_points(p,u)[1] #get nearest point
        mindist = p.distance(nearestpoint) #find the distance to the nearest point
      
        work_dist, outdoor_dist, shopping_dist = distance(sopa) #get distance and generate the distribution
        # calculate distance (radius)
        if  goal  == "work": 
            sim_dist = gen_lnorm(work_dist, "")
            sim_distdeg = sim_dist/110.8 #convert to degree
        elif goal == "sports":
            sim_dist = gen_lnorm(outdoor_dist, "")
            sim_distdeg = sim_dist/110.8     
        elif goal == "shopping":
            sim_dist = gen_lnorm(shopping_dist, "")
            sim_distdeg = sim_dist/110.8 
        
        if sim_distdeg < mindist : # if the distance is shorter than the distance to the nearest points. take the nearest point
            sim_distdeg = mindist
            des_p= nearestpoint
            num_points = 0
            print(f'use nearest point as simulated distance is too short.')
        # else calculate a buffer and select points. 
        else:
            pbuf=p.buffer(sim_distdeg) # distance to degree`maybe better to project. 
            pbuf.crs={'init': 'epsg:4326', 'no_defs': True}
            worklocset =w_gdf[w_gdf.within(pbuf)]
            num_points = len(worklocset)
            print(f'sample from {num_points} points')
            workloc =  worklocset.sample(n = 1, replace=True, random_state=1)
            des_p = workloc.iloc[0]["geometry"] #get point out of the geopandas dataframe 
        return (p, des_p,num_points)
    
    '''
    # main function calculating all the destination points to dataframe. 
    input: homedf: dataframe of home (original) locations, names have to be "lat, lon". 
           goal:   dataframe or geopandas dataframe of work (destination) locations. names have to be "lat lon"    
           n: number of fisrt "n" points to calculate, e.g. n = 4, only calculate for the first 4 points. 
           csvname: name/dir for saving the csv file. if the csvname is None, not saving anything. 
           
    output: return the dataframe of lat lon of the original and destination locations. home_lon, home_lat, work_lon, work_lat, number of candicate destinations. 
    '''
    
    def storedf(self):
        homedf = self.homedf
        goal=self.goal
        n=self.n
        csvname = self.csvname    
        totalarray = [0,0,0,0,0]
        w_gdf, u  = pot_dest(goal) #get work(destination location)
        for id in range (n): 
            h =homedf.loc[id]
            p=Point(h.lon, h.lat) # distance to degree`maybe better to project. 
            
            op, dp, num_p  = getdestloc(p, w_gdf, u)
            parray = disto2d(op,dp)
            
            parray.insert(4,num_p)     
            
            #totalarray = totalarray.append(parray)
            #print(totalarray)
            totalarray = np.concatenate((totalarray, parray), axis=0)
        
        total = np.array (totalarray )
        
        totalre = total.reshape([-1, 5])[1:,:]
        totalre = pd.DataFrame(totalre)
        totalre = totalre.rename (columns = {0:"home_lon", 1: "home_lat", 2:"work_lon", 3:"work_lat", 4: "num_candi"})
        if not csvname is None:
            totalre.to_csv(f'{csvname}.csv')
        return 1
  
a = Gen_dest(homedf, Uni, n= 5, csvname= None)
a1 = a.storedf()
 