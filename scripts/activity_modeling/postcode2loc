#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:30:33 2021

@author: menglu
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pgeocode
import geopandas as gpd
from shapely.geometry import Point
filedir = "/Users/menglu/Documents/GitHub/mobiair/"
Ovin = pd.read_csv(filedir+'human_data/dutch_activity/Ovin.csv')
nomi = pgeocode.Nominatim('nl')
depzip = Ovin[["VertPC"]]
arrizip = Ovin[["AankPC"]]
depcoor = nomi.query_postal_code(np.array(depzip).squeeze())
arrcoor = nomi.query_postal_code(np.array(arrizip).squeeze())
Ovin[["dep_lat"]]=depcoor[["latitude"]]
Ovin[["dep_lon"]]=depcoor[["longitude"]]
Ovin[["arr_lat"]]=arrcoor[["latitude"]]
Ovin[["arr_lon"]]=arrcoor[["longitude"]]
Ovin.to_csv(filedir+'human_data/dutch_activity/Ovin.csv')
# I acutually know the exact work location, can be used for validation. not many if only for Utrecht though. 25 points
# Unistudent
Ovin.query('Doel == "Werken" & MaatsPart=="Scholier/student" & age_lb>=18 & VertProv == "Utrecht"')[["dep_lat","dep_lon"]]  
Ovin.query('Doel == "Werken" & MaatsPart=="Scholier/student"& age_lb>=18 & VertProv == "Utrecht"')[["arr_lat","arr_lon"]] 

#school students 
Ovin.query('Doel == "Werken" & MaatsPart=="Scholier/student" & age_lb<18')[["dep_lat","dep_lon"]] 
Ovin.query('Doel == "Werken" & MaatsPart=="Scholier/student"& age_lb<18')[["arr_lat","arr_lon"]] 
 