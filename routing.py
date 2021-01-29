import math
import json
import requests

from osgeo import ogr, gdal, osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()



class Routing(object):
  def __init__(self, server='127.0.0.1', port_car=None, port_bicycle=None, port_foot=None, port_train=None):
    self.server_ip = server
    self.port_car = port_car
    self.port_bicycle = port_bicycle
    self.port_foot = port_foot
    self.port_train = port_train


  def _profile_to_port(self, profile):

    if profile == 'car':
      return self.port_car
    elif profile == 'bicycle':
      return self.port_bicycle
    elif profile == 'foot':
      return self.port_foot
    elif profile == 'train':
      return self.port_train
    else:
      raise NotImplementedError


  def distance(self, lat1, lon1, lat2, lon2, profile):
    """ Returns distance in metres and estimated time in seconds
    """

    port = self._profile_to_port(profile)

    payload = {'steps':'false', 'alternatives':'false', 'annotations':'false', 'overview':'false'}

    query = f"http://{self.server_ip}:{port}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?"

    try:
      r = requests.get(query, params=payload)
    except requests.exceptions.RequestException as e:
      raise SystemExit(e)

    content = r.json()

    return_code = content['code']

    if return_code != 'Ok':
      msg = f'Failed to obtain route, response code is: "{return_code}"'
      raise RuntimeError(msg)


    route_content = content['routes'][0]
    distance = route_content['distance']
    duration = route_content['duration']

    return distance, duration


  def set_base_grid(self, filename):
    """ Setting a base grid. This will be used to align the raster cells of the returned route
    """

    raster = gdal.Open(filename, gdal.GA_ReadOnly)

    geoTransform = raster.GetGeoTransform()

    self.min_x = geoTransform[0]
    self.max_x = self.min_x + geoTransform[1] * raster.RasterXSize
    self.max_y = geoTransform[3]
    self.min_y = self.max_y + geoTransform[5] * raster.RasterYSize
    self.cellsize = geoTransform[1]

    proj = osr.SpatialReference(wkt=raster.GetProjection())
    self.epsg = int(proj.GetAttrValue('AUTHORITY', 1))



  def get_raster(self, lat1, lon1, lat2, lon2, profile):
    """ Returns rasterised route, matched to a base grid
    """

    port = self._profile_to_port(profile)

    payload = { 'steps':'true', 'alternatives':'false', 'annotations':'false', 'overview':'full', 'geometries':'geojson' }

    query = f"http://{self.server_ip}:{port}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?"

    try:
      r = requests.get(query, params=payload)
    except requests.exceptions.RequestException as e:
      raise SystemExit(e)

    content = r.json()

    return_code = content['code']

    if return_code != 'Ok':
      msg = f'Failed to obtain route, response code is: "{return_code}"'
      raise RuntimeError(msg)


    route_content = content['routes'][0]

    coordinates = route_content['geometry']['coordinates']

    # Make coordinate list suitable for GDAL
    content = \
"""{
"type": "FeatureCollection",
"name": "route",
"features": [
{ "type": "Feature", "properties": { }, "geometry":
{ "type": "LineString", "coordinates":""" +\
"{}".format(coordinates) +\
""" } }
]
}
"""

    # Route to GDAL dataset, route is in wgs84
    ds = gdal.OpenEx(content)
    assert ds is not None, 'Failed to open datasource'

    wgs_layer = ds.GetLayer()
    featureCount = wgs_layer.GetFeatureCount()
    assert wgs_layer.GetFeatureCount() == 1

    src_proj = wgs_layer.GetSpatialRef()

    assert src_proj.GetAuthorityCode(None) == '4326', src_proj.GetAuthorityCode(None)
    assert src_proj.GetDataAxisToSRSAxisMapping() == [2, 1]


    # Reproject route to the target CRS
    dst_ref = osr.SpatialReference()
    dst_ref.ImportFromEPSG(self.epsg)
    transformation = osr.CoordinateTransformation(src_proj, dst_ref)

    mem_ds = ogr.GetDriverByName('MEMORY').CreateDataSource('')
    dst_layer = mem_ds.CreateLayer('route', geom_type=ogr.wkbLineString, srs=dst_ref)

    wgs_layer.ResetReading()
    wgs_feature = wgs_layer.GetNextFeature()
    wgs_geometry = wgs_feature.GetGeometryRef()
    wgs_geometry.Transform(transformation)

    dst_geometry = ogr.CreateGeometryFromWkb(wgs_geometry.ExportToWkb())
    feature = ogr.Feature(dst_layer.GetLayerDefn())
    feature.SetGeometry(dst_geometry)
    dst_layer.CreateFeature(feature)
    feature = None

    assert int(dst_layer.GetSpatialRef().GetAuthorityCode(None)) == self.epsg, dst_layer.GetSpatialRef().GetAuthorityCode(None)

    assert dst_layer.GetFeatureCount() == 1
    dst_layer = None


    # Determine target raster extent, start with current extent
    dst_layer = mem_ds.GetLayer()
    source_feature = dst_layer.GetNextFeature()
    env = source_feature.GetGeometryRef().GetEnvelope()
    env_minX = env[0]
    env_maxX = env[1]
    env_minY = env[2]
    env_maxY = env[3]

    # Snap to base grid
    route_min_x = self.min_x + self.cellsize * math.floor(math.fabs(env_minX - self.min_x) / self.cellsize)
    route_max_x = self.max_x - self.cellsize * math.floor(math.fabs(env_maxX - self.max_x) / self.cellsize)
    route_max_y = self.max_y - self.cellsize * math.floor(math.fabs(env_maxY - self.max_y) / self.cellsize)
    route_min_y = self.min_y + self.cellsize * math.floor(math.fabs(env_minY - self.min_y) / self.cellsize)

    assert route_min_x <= env_minX
    assert route_max_x >= env_maxX
    assert route_min_y <= env_minY
    assert route_max_y >= env_maxY

    route_nr_rows, remainder = divmod(math.fabs(route_max_y - route_min_y), self.cellsize)
    assert remainder == 0, 'rows {} remainder {}'.format(nr_rows, remainder)
    route_nr_cols, remainder = divmod(math.fabs(route_max_x - route_min_x), self.cellsize)
    assert remainder == 0, 'rows {} remainder {}'.format(nr_rows, remainder)

    route_nr_rows = int(route_nr_rows)
    route_nr_cols = int(route_nr_cols)

    # Burn route into the raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xsize=route_nr_cols, ysize=route_nr_rows, bands=1, eType=gdal.GDT_Byte)
    target_ds.SetGeoTransform((route_min_x, self.cellsize, 0, route_max_y, 0, -self.cellsize))
    target_ds.SetProjection(dst_ref.ExportToWkt())

    err = gdal.RasterizeLayer(target_ds, [1], dst_layer, burn_values=[1], options=['ALL_TOUCHED=TRUE'])

    return target_ds





  def gpkg(self, lat1, lon1, lat2, lon2, profile, filename):
    """ Returns GeoPackage with route
    """

    port = self._profile_to_port(profile)

    payload = { 'steps':'true', 'alternatives':'false', 'annotations':'false', 'overview':'full', 'geometries':'geojson' }

    query = f"http://{self.server_ip}:{port}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?"

    try:
      r = requests.get(query, params=payload)
    except requests.exceptions.RequestException as e:
      raise SystemExit(e)

    content = r.json()

    return_code = content['code']

    if return_code != 'Ok':
      msg = f'Failed to obtain route, response code is: "{return_code}"'
      raise RuntimeError(msg)


    route_content = content['routes'][0]

    coordinates = route_content['geometry']['coordinates']

    # Make coordinate list suitable for GDAL
    content = \
"""{
"type": "FeatureCollection",
"name": "route",
"features": [
{ "type": "Feature", "properties": { }, "geometry":
{ "type": "LineString", "coordinates":""" +\
"{}".format(coordinates) +\
""" } }
]
}
"""

    # Route to GDAL dataset, route is in wgs84
    ds = ogr.GetDriverByName('GeoJSON').Open(content)
    assert ds is not None, 'Failed to open datasource'

    out = ogr.GetDriverByName('GPKG').CopyDataSource(ds, filename)

    out = None
